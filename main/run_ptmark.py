import argparse
import yaml
import os
from pathlib import Path
import logging
import shutil
import numpy as np
from PIL import Image 
from collections import defaultdict
from sklearn import metrics
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from ptmark_scheduling_ddim import DDIMScheduler
from datasets import load_dataset
# from diffusers.utils.torch_utils import randn_tensor
import sys
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import open_clip
import lpips

from statistics import mean, stdev

from ptmark_pipeline import WMDetectStableDiffusionPipeline
from ptmark_reference_pipeline import WMDetectStableDiffusionPipelinePtMark
from ptmark_watermark import GTWatermark, GTWatermarkMulti
from ptmark_utils import *
from ptmark_null_inversion import *
from pytorch_msssim import ms_ssim
from pytorch_ssim import ssim

parser = argparse.ArgumentParser(description='ptmark')
parser.add_argument('--original_image_path', default='./data/watermark_image', type=str, help='Path to the images folder')
parser.add_argument('--num_inner_steps',default=10, type=int)
parser.add_argument('--image_number',default=100, type=int)
parser.add_argument('--watarmark_loss_weight',default=0.0006, type=float)
args = parser.parse_args()

if 'coco' in args.original_image_path:
    dataset_name = 'coco_1000'
else:
    dataset_name = 'DiffusionDB'

# # 创建一个 SummaryWriter 对象，指定日志目录
# logger = setup_logging(log_dir='./logs_ptmark', log_name='double_watermark_channel_1')

# logger.info(f'===== Load Config =====')
device = torch.device('cuda')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
with open(PROJECT_ROOT / 'example/config/config.yaml', 'r') as file:
    cfgs = yaml.safe_load(file)
# logger.info(args)

# logger.info(f'===== Init Pipeline =====')
wm_pipe = GTWatermark(device, w_channel=3, w_radius=cfgs['w_radius'], generator=torch.Generator(device).manual_seed(cfgs['w_seed']))

scheduler = DDIMScheduler.from_pretrained(cfgs['model_id'], subfolder="scheduler")
pipe = WMDetectStableDiffusionPipeline.from_pretrained(cfgs['model_id'], scheduler=scheduler).to(device)
reference_pipe = WMDetectStableDiffusionPipelinePtMark.from_pretrained(cfgs['model_id'], scheduler=scheduler).to(device)
pipe.set_progress_bar_config(disable=True)
null_inversion = NullInversion(pipe)

tester_prompt = '' 
tester_embeddings = pipe.get_text_embedding(tester_prompt)

lpips_fn = lpips.LPIPS(net='alex').to(device)


wm_path = str(PROJECT_ROOT / 'outputs' / 'null_optimization_double_channel_1')
os.makedirs(wm_path, exist_ok=True)

# Diffusion DB
df = pd.read_parquet(PROJECT_ROOT / 'data' / 'prompts' / 'eval.parquet')

# dataset coco hard coding for now
with open(PROJECT_ROOT / 'data' / 'coco' / 'meta_data.json') as f:
    dataset = json.load(f)
    image_files = dataset['images']
    dataset = dataset['annotations']
    prompt_key = 'caption'

with open(PROJECT_ROOT / 'png_filenames.log', 'r', encoding='utf-8') as file:
    image_list_name = [line.strip() for line in file]

clean_path = str(PROJECT_ROOT / 'data' / 'no_watermark_imgs' / 'v2.1-base')


ssim_list = []
mssim_list = []
psnr_list = []
lpips_list = []
original_image_list = []
watermark_image_list = []
det_prob_list = defaultdict(list)
o_l1_list = defaultdict(list)
w_l1_list = defaultdict(list)
global_num = 0
image_list = natsorted(os.listdir(args.original_image_path))
for index in tqdm(range(100)):
    if 'coco' in args.original_image_path:
        prompt = dataset[index][prompt_key] 
        name = image_files[index]['file_name']
    else:
        name = image_list[index]
        prompt = df.Prompt[index] 
        if name not in image_list_name:
            continue
    cur_time = time.time()
    imagename = os.path.join(args.original_image_path, name)
    gt_img_tensor = get_img_tensor(imagename, device)

    # Step 1: Get init noise
    # empty_text_embeddings = pipe.get_text_embedding('')
    empty_text_embeddings = pipe.get_text_embedding(prompt)
    init_latents_approx = get_init_latent_list(gt_img_tensor, pipe, empty_text_embeddings)
    null_inversion.init_prompt(prompt=prompt)

    
    print(f'image {imagename}:')
    # logger.info(f'image {imagename}:')
    init_latents = init_latents_approx[-1].detach().clone()
    init_latents_wm = wm_pipe.inject_watermark(init_latents)
    wm_init_latents = pipe(prompt, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm, return_latents_list=True).init_latents
    wm_init_latents = [init_latents_wm] + wm_init_latents
    for step in range(len(wm_init_latents)):
        save_latents_singleChannel(wm_init_latents[step], channel=3, mask=wm_pipe.watermarking_mask, camp='plasma', save_path=str(PROJECT_ROOT / 'outputs' / 'saliency_list_forward' / f'{step}.png'))

    
    init_latents_approx[-1] = init_latents_wm.detach().clone()
    uncond_embeddings = null_inversion.null_optimization_double(init_latents_approx, wm_init_latents, num_inner_steps=args.num_inner_steps, epsilon=1e-5, wm_pipe=wm_pipe, watarmark_loss_weight=args.watarmark_loss_weight)
    pred_img_tensor = pipe(prompt, num_inference_steps=50, output_type='tensor', use_trainable_latents=True, init_latents=init_latents_wm, text_embeddings=uncond_embeddings).images

    cost = time.time() - cur_time
    print(cost)
    torch.cuda.empty_cache()
    
    # image_path = 'null_optimization_double.png'
    image_path = os.path.join(wm_path, name)
    pred_img_pil = save_img(image_path, pred_img_tensor, pipe)

    clean_name = os.path.join(clean_path, name)
    gt_image_pil = Image.open(clean_name)

    
    ssim_value = ssim(pred_img_tensor, gt_img_tensor).item()
    mssim_value = ms_ssim(gt_img_tensor, pred_img_tensor,data_range=1.).item()
    psnr_value = compute_psnr(pred_img_tensor, gt_img_tensor)
    lpips_value = float(lpips_fn(pred_img_tensor, gt_img_tensor).item())


    ssim_list.append(ssim_value)
    psnr_list.append(psnr_value)
    lpips_list.append(lpips_value)
    mssim_list.append(mssim_value)
    original_image_list.append(imagename)
    watermark_image_list.append(image_path)

    for attack in ['none', 'rotation', 'jpeg', 'cropping', 'blurring', 'noise', 'color_jitter']:
        print(f'attack: {attack}')
        # logger.info(f'attack: {attack}')
        # distortion
        gt_image_distortion, pred_img_distortion = image_distortion_attack(gt_image_pil, pred_img_pil, 0, attack)
        pred_img_distortion_tensor = (pil_to_tensor(pred_img_distortion)/255).unsqueeze(0).to(device)
        gt_img_distortion_tensor = (pil_to_tensor(gt_image_distortion)/255).unsqueeze(0).to(device)
        # pred_img_distortion_tensor = transform_img(pred_img_distortion).unsqueeze(0).to(empty_text_embeddings.dtype).to(device)
        # gt_img_distortion_tensor = transform_img(gt_image_distortion).unsqueeze(0).to(empty_text_embeddings.dtype).to(device)
        w_det_prob, w_l1 = watermark_prob(pred_img_distortion_tensor, pipe, wm_pipe, empty_text_embeddings, return_distance=True)
        o_det_prob, o_l1 = watermark_prob(gt_img_distortion_tensor, pipe, wm_pipe, empty_text_embeddings, return_distance=True)
        w_l1_list[attack].append(w_l1)
        o_l1_list[attack].append(o_l1)
        det_prob_list[attack].append(1 - w_det_prob)
        logger.info(f'w_l1 {w_l1}, o_l1 {o_l1}, det_prob {1 - w_det_prob}')
        print(f'w_l1 {w_l1}, o_l1 {o_l1}, det_prob {1 - w_det_prob}')

    logger.info(f'SSIM {ssim_value}, MSSIM {mssim_value}, PSNR {psnr_value}, LPIPS {lpips_value}')
    print(f'SSIM {ssim_value}, MSSIM {mssim_value}, PSNR {psnr_value}, LPIPS {lpips_value}')

for attack in ['none', 'rotation', 'jpeg', 'cropping', 'blurring', 'noise', 'color_jitter']:
    print(f'attack: {attack}')
    logger.info(f'attack: {attack}')
    preds = o_l1_list[attack] +  w_l1_list[attack]
    t_labels = [1] * len(o_l1_list[attack]) + [0] * len(w_l1_list[attack])

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low} ')
    logger.info(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low} ')
    print(f'mean det_prob_list: {mean(det_prob_list[attack])}, mean o_l1_list: {mean(o_l1_list[attack])}, mean w_l1_list: {mean(w_l1_list[attack])}')
    logger.info(f'mean det_prob_list: {mean(det_prob_list[attack])}, mean o_l1_list: {mean(o_l1_list[attack])}, mean w_l1_list: {mean(w_l1_list[attack])}')


print(f'mean SSIM = {mean(ssim_list)}, mean MSSIM {np.mean(mssim_list)}, mean PSNR = {mean(psnr_list)}, mean LPIPS {mean(lpips_list)}')
logger.info(f'mean SSIM = {mean(ssim_list)}, mean MSSIM {np.mean(mssim_list)}, mean PSNR = {mean(psnr_list)}, mean LPIPS {mean(lpips_list)}')



