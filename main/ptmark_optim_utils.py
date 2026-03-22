import torch
from torchvision import transforms
from datasets import load_dataset

from PIL import Image, ImageFilter
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy
import time
import logging
import os
import matplotlib.pyplot as plt
from skimage.transform import radon
from skimage import io
import cv2
import torch.nn as nn
import torchvision.models as models
import math
from typing import List, Optional, Tuple, Union
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'loss'))
from watson_vgg import WatsonDistanceVgg
from pytorch_ssim import SSIM


class RingDetectorCNN(nn.Module):
    def __init__(self):
        super(RingDetectorCNN, self).__init__()
        # 使用预训练的ResNet
        self.resnet = models.resnet18(pretrained=True)
        # 修改第一层以适应灰度图像输入
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后一层的输出为3（x, y, r）
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        return self.resnet(x)

def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)
    

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img2)
        
    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


# for one prompt to multiple images
def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.cat(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key


def circle_mask_steal(size=64, r=10, x0=0, x_offset=0, y0=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = np.round(x0)
    y0 = np.round(y0)
    r = np.round(r)
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2

def circle_mask(size=64, r=10, x0=0, x_offset=0, y0=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2


def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def get_watermarking_mask_steal(init_latents_w, x, y, r, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=r.cpu().numpy(), x0=x.cpu().numpy(), y0=y.cpu().numpy())
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')

    return watermarking_mask


def get_watermarking_pattern(pipe, args, device, shape=(1,4,64,64)):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device).to(torch.complex32)
    else:
        gt_init = pipe.get_random_latents()

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        

        gt_patch_tmp = copy.deepcopy(gt_patch)

        gt_patch_path = './data/gt_patch_image/demo_2'

        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            
            # gt_patch_visual
            # path = os.path.join(gt_patch_path, str(i))
            # os.makedirs(path, exist_ok=True)
            # a = gt_patch[0].cpu().numpy().astype(np.float32)
            # for k in range(4):
            #     plt.imshow(a[k], cmap='gray')
            #     plt.title(f'Channel {k + 1}')
            #     plt.axis('off')
            #     plt.savefig(os.path.join(path, 'channel_{}.png'.format(k+1)))
            #     plt.close()
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


def get_watermarking_pattern_testMultiRadiusACC(pipe, args, device, shape=(1,4,64,64)):
    set_random_seed(args.w_seed)
    gt_patchs = []
    for k in range(args.w_radius, 0, -1):
        if shape is not None:
            gt_init = torch.randn(*shape, device=device).to(torch.complex32)
        else:
            gt_init = pipe.get_random_latents()

        if 'seed_ring' in args.w_pattern:
            gt_patch = gt_init

            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(args.w_radius, 0, -1):
                tmp_mask = circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif 'seed_zeros' in args.w_pattern:
            gt_patch = gt_init * 0
        elif 'seed_rand' in args.w_pattern:
            gt_patch = gt_init
        elif 'rand' in args.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]
        elif 'zeros' in args.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        elif 'const' in args.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
            gt_patch += args.w_pattern_const
        elif 'ring' in args.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            
            gt_patch_tmp = copy.deepcopy(gt_patch)

            for i in range(k, 0, -1):
                tmp_mask = circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

        gt_patchs.append(gt_patch)
        # save_fft_image(gt_patch[0].cpu().float(), p=os.path.join('./outputs/save_fft_image', str(k)))

    return gt_patchs

def inject_watermark(init_latents_w, watermarking_mask, gt_patch, args, num):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w).to(torch.complex64), dim=(-1, -2)).to(torch.complex32)

    # bulid frequencyDistribution map
    init_latents_fft = init_latents_w_fft.clone()

    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {args.w_injection}')

    # bulid frequencyDistribution map
    # frequencyDistribution(init_latents_fft, init_latents_w_fft, num)

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft.to(torch.complex64), dim=(-1, -2))).to(torch.complex32).real

    return init_latents_w

def inject_watermark_steal(init_latents_w, watermarking_mask, gt_patch, args, num):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w).to(torch.complex64), dim=(-1, -2)).to(torch.complex32)

    # with torch.no_grad():
    #     predict_circle = model(init_latents_w.unsqueeze(0))

    # watermarking_mask = get_watermarking_mask(init_latents_w, predict_circle[0], predict_circle[1], predict_circle[2], args, device)
    

    if args.w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif args.w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {args.w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft.to(torch.complex64), dim=(-1, -2))).to(torch.complex32).real

    return init_latents_w


def frequencyDistribution(latents_o, latens_w, num):
    # 计算频域幅度谱
    magnitude_spectrum_o = np.log(np.abs(latents_o.real.cpu()) + 1).flatten()
    magnitude_spectrum_w = np.log(np.abs(latens_w.real.cpu()) + 1).flatten()
    magnitude_spectrum_d = magnitude_spectrum_w - magnitude_spectrum_o

    

    # 计算频域数据的直方图
    hist_o, bin_edges = np.histogram(magnitude_spectrum_o, bins=50)
    hist_w, bin_edges = np.histogram(magnitude_spectrum_w, bins=50)
    hist_d, bin_edges = np.histogram(magnitude_spectrum_d, bins=50)
    histograms = [hist_o, hist_w, hist_d]

    # 定义颜色和标签
    colors = ['blue', 'green', 'red']
    labels = ['magnitude_spectrum_o', 'magnitude_spectrum_w', 'magnitude_spectrum_d']

    # 绘制频域幅度谱
    plt.figure(figsize=(12, 6))

    for i, hist in enumerate(histograms):
        # 绘制柱状图
        plt.bar(bin_edges[:-1] + i * (bin_edges[1] - bin_edges[0]) / 4, hist, width=(bin_edges[1] - bin_edges[0]) / 4, 
                edgecolor='black', alpha=0.7, color=colors[i], label=f'{labels[i]} Histogram')
        
        # # 绘制折线图
        # plt.plot(bin_edges[:-1] + i * (bin_edges[1] - bin_edges[0]) / 4, hist, marker='o', color=colors[i], label=f'{labels[i]} Line Plot')

    # 添加标题和标签
    plt.title('Frequency Domain Histogram and Line Plot for Three Images')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.savefig('./outputs/magnitude_spectrum/{}.png'.format(num))
    plt.show() 
    plt.close() 


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w).to(torch.complex64), dim=(-1, -2)).to(torch.complex32)
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w).to(torch.complex64), dim=(-1, -2)).to(torch.complex32)
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric


def eval_watermark_noiseReplacement(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args, k):
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w).to(torch.complex64), dim=(-1, -2)).to(torch.complex32)
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w).to(torch.complex64), dim=(-1, -2)).to(torch.complex32)
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        gt_init = torch.randn((1,4,64,64)).cuda().to(torch.complex32)
        gt_init_fft= torch.fft.fftshift(torch.fft.fft2(gt_init),dim=(-1,-2))
        
        reversed_latents_w_fft_temp = copy.deepcopy(reversed_latents_w_fft)
        tmp_mask = circle_mask(reversed_latents_w.shape[-1], r=k)
        tmp_mask= torch.tensor(tmp_mask).cuda()
        reversed_latents_w_fft_temp[watermarking_mask] = gt_init_fft[watermarking_mask].clone()
        reversed_latents_w_fft_temp[:, args.w_channel, tmp_mask] = reversed_latents_w_fft[:, args.w_channel, tmp_mask].clone()
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft_temp[watermarking_mask] - target_patch[watermarking_mask]).mean().item()


    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric


def eval_watermark_noiseAdd(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args, k):
    g_noise = np.random.normal(0, k, reversed_latents_no_w.shape)
    g_noise = torch.tensor(g_noise).cuda().to(reversed_latents_w.dtype)
    reversed_latents_w = g_noise + reversed_latents_w
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w).to(torch.complex64), dim=(-1, -2)).to(torch.complex32)
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w).to(torch.complex64), dim=(-1, -2)).to(torch.complex32)
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()


    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric


def eval_watermark_ab(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch_1, gt_patch_2, args):
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - gt_patch_1[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - gt_patch_2[watermarking_mask]).mean().item()
    else:
        NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric

def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    # assume it's Fourier space wm
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])
    
    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def setup_logging(log_name=None, args=None, log_dir='logs'):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """

    os.makedirs(log_dir, exist_ok=True)

    # 设置日志文件名
    cur_time = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    if args is None and log_name is None:
        file_name = (
            cur_time 
            + '.log'
        )
    elif log_name:
        file_name = (
            cur_time + '_'
            + log_name
            + '.log'
        )
    elif args:
        file_name = (
            cur_time 
            + args.watermark_image_dir.split('/')[-1]
            + '.log'
        )
    log_file = os.path.join(log_dir, file_name)
    
    # 获取模块特定的 Logger 对象
    logger = logging.getLogger(__name__)
    
    # 配置日志记录器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger

def extract_ring_parameters(image):
    # image = io.imread('./outputs/reversed_latents_w_fft/channel_4.png', as_gray=True)
    image = image.to(torch.complex64).cpu().numpy().astype(np.float32)
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    
    peak_positions = np.argmax(sinogram, axis=0)
    center_position = np.mean(peak_positions)
    radius = (np.max(peak_positions) - np.min(peak_positions)) / 2
    
    return center_position, radius

def extract_ring_Canny(image, k, path):
    image = image.astype(np.uint8)
    # Apply Gaussian blur to reduce noise
    # blurred = cv2.GaussianBlur(image, (9, 9), 2).astype(np.uint8)

    # Apply edge detection
    edges = cv2.Canny(image, 200, 300)
    cv2.imwrite(os.path.join(path,'channel_{}_Canny.png'.format(k+1)),  edges)

    # Use the Hough Circle Transform to detect the ring
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                            param1=50, param2=30, minRadius=0, maxRadius=100)

    # If some circles are detected
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # Loop over the detected circles
        for (x, y, r) in circles:
            # Draw the circle in the output image
            cv2.circle(image, (x, y), r, (255, 0, 0), 2)
            # Draw a rectangle around the circle
            # cv2.rectangle(image, (x - r, y - r), (x + r, y + r), (0, 128, 255), 4)
            print(f"Circle found at (x={x}, y={y}) with radius={r}")
            
    # Show the result
    # cv2.imshow("Detected Circles", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(path,'channel_{}_extracted_ring.png'.format(k+1)),  image)


def save_fft_image(image, p=None):
    path = './outputs/save_fft_image'
    if p is not None:
        os.makedirs(p,exist_ok=True)
        path = p
    if image.shape[0] == 4:
        for k in range(4):
            plt.imshow(image[k], cmap='gray')
            plt.title(f'Channel {k + 1}')
            plt.axis('off')
            plt.savefig(os.path.join(path, 'channel_{}.png'.format(k+1)))
            plt.close()
    else:
        plt.imshow(image, cmap='gray')
        plt.title(f'fft_image')
        plt.axis('off')
        plt.savefig(os.path.join(path, 'fft_image.png'))
        plt.close()

def KL_loss(latents_prev_rec, latent_prev):
    # 计算 KL 散度损失
    mu_q = torch.mean(latents_prev_rec, dim=(2, 3), keepdim=True)  # (T,)
    sigma_q = torch.std(latents_prev_rec, dim=(2, 3), keepdim=True) + 1e-6  # 避免除 0

    mu_p = torch.mean(latent_prev, dim=(2, 3), keepdim=True)  # 原始高斯分布的均值
    sigma_p = torch.std(latent_prev, dim=(2, 3), keepdim=True) + 1e-6  # 原始高斯分布的方差

    kl_div = torch.sum(torch.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5)
    return kl_div

def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img2)
        
    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


def image_distortion_attack(img1, img2, seed, attack):
    if attack == 'rotation':
        img1 = transforms.RandomRotation((75, 75))(img1)
        img2 = transforms.RandomRotation((75, 75))(img2)

    if attack == 'jpeg':
        img1.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img1 = Image.open(f"tmp_{25}_{attack}.jpg")
        img2.save(f"tmp_{25}_{attack}.jpg", quality=25)
        img2 = Image.open(f"tmp_{25}_{attack}.jpg")

    if attack == 'cropping':
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(0.75, 0.75), ratio=(0.75, 0.75))(img2)
        
    if attack == 'blurring':
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=4))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=4))

    if attack == 'noise':
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, 0.1, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if attack == 'color_jitter':
        img1 = transforms.ColorJitter(brightness=6)(img1)
        img2 = transforms.ColorJitter(brightness=6)(img2)

    return img1, img2

class DistortionAttacks():
    def __init__(self):
        super(DistortionAttacks, self).__init__()
    def apply_rotation(img1, img2, params):
        img1 = transforms.RandomRotation((params, params))(img1)
        img2 = transforms.RandomRotation((params, params))(img2)
        return  img1, img2
    def apply_jpeg(img1, img2, params):
        img1.save(f"tmp_{params}.jpg", quality=params)
        img1 = Image.open(f"tmp_{params}.jpg")
        img2.save(f"tmp_{params}.jpg", quality=params)
        img2 = Image.open(f"tmp_{params}.jpg")
        return  img1, img2
    def apply_crop(img1, img2, params):
        seed = 0
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(params, params))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(params, params))(img2)
        return  img1, img2
    def apply_blur(img1, img2, params):
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=params))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=params))
        return  img1, img2
    def apply_noise(img1, img2, params):
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, params, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))
        return  img1, img2
    def apply_brightness(img1, img2, params):
        img1 = transforms.ColorJitter(brightness=params)(img1)
        img2 = transforms.ColorJitter(brightness=params)(img2)
        return  img1, img2

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def watermark_loss(latents_rec, latents_wm, wm_pipe):
    latents_rec_fft = torch.fft.fftshift(torch.fft.fft2(latents_rec), dim=(-1, -2))
    latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(latents_wm), dim=(-1, -2))
    loss = nn.L1Loss()
    lossW = loss(latents_rec_fft[wm_pipe.watermarking_mask], latents_wm_fft[wm_pipe.watermarking_mask])
    return lossW

def watermark_loss_wofft(latents_rec, latents_wm, wm_pipe):
    loss = nn.L1Loss()
    lossW = loss(latents_rec[wm_pipe.watermarking_mask], latents_wm[wm_pipe.watermarking_mask])
    return lossW

def watermark_loss_reverse(latents_rec, latents_wm, wm_pipe):
    latents_rec_fft = torch.fft.fftshift(torch.fft.fft2(latents_rec), dim=(-1, -2))
    latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(latents_wm), dim=(-1, -2))
    loss = nn.L1Loss()
    lossW = loss(latents_rec_fft[~wm_pipe.watermarking_mask], latents_wm_fft[~wm_pipe.watermarking_mask])
    return lossW

    

def save_latents_singleChannel(latent, channel=0, name='latent_color_singleChannel', camp='viridis', mask=None, save_path=None):
    # 可视化一个通道为伪彩色图
    latent = torch.fft.fftshift(torch.fft.fft2(latent), dim=(-1, -2)).cpu().float()
    img = latent[0][channel].cpu().numpy()# 你可以改为 1、2、3 看不同通道
    if mask is not None:
        mask = mask[0][channel].cpu()
        # img[mask[0][channel].cpu()] = 1
        img_clone = img.copy()
        # img_clone[mask] *= 7
        plt.imshow(img_clone, cmap=camp)  # 'viridis' 是类似你发的那张图的伪彩色风格
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        plt.imshow(img, cmap=camp)  # 'viridis' 是类似你发的那张图的伪彩色风格
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'./outputs/watermark_initial_radius_15.png', dpi=300, bbox_inches='tight', pad_inches=0)


# class LossProvider(nn.Module):
#     def __init__(self, device):
#         super(LossProvider, self).__init__()
        
#         self.loss_img, self.loss_w = nn.MSELoss(), nn.L1Loss()
#         self.loss_ssim = SSIM()

#         # add perceptive loss
#         loss_percep = WatsonDistanceVgg(reduction='sum')
#         loss_percep.load_state_dict(torch.load('./loss/rgb_watson_vgg_trial0.pth', map_location='cpu'))
#         loss_percep = loss_percep.to(device)
#         self.loss_per = lambda pred_img, gt_img: loss_percep((1+pred_img)/2.0, (1+gt_img)/2.0)/ pred_img.shape[0]

#     def __call__(self, pred_img_tensor, gt_img_tensor):
#         # init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
#         # lossW = self.loss_w(init_latents_fft[wm_pipe.watermarking_mask], wm_pipe.gt_patch[wm_pipe.watermarking_mask])*self.loss_weights[3]
#         lossI = self.loss_img(pred_img_tensor, gt_img_tensor)*10
#         lossP = self.loss_per(pred_img_tensor, gt_img_tensor)*0.1
#         lossS = (1-self.loss_ssim(pred_img_tensor, gt_img_tensor))
#         loss = lossI + lossP + lossS
#         return loss


