from PIL import Image
import math
import os
# from pytorch_msssim import ssim, ms_ssim
import matplotlib.pyplot as plt

from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
import torch
from typing import List, Optional, Tuple, Union
import time
import logging
import os
from diffusers import DDIMInverseScheduler, StableDiffusionXLPipeline
from torchvision import transforms as tvt
from ptmark_watermark import GTWatermark
import copy
import torch.nn.functional as F
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def show_images_side_by_side(images, titles=None, figsize=(8,4)):
    """
    Display a list of images side by side.
    
    Args:
    images (list of numpy arrays): List of images to display.
    titles (list of str, optional): List of titles for each image. Default is None.
    """
    num_images = len(images)
    
    if titles is not None:
        if len(titles) != num_images:
            raise ValueError("Number of titles must match the number of images.")
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i]) 
        ax.axis('off')
        
        if titles is not None:
            ax.set_title(titles[i])
    
    plt.tight_layout()
    plt.show()
    return

def show_latent_and_final_img(latent: torch.Tensor, img: torch.Tensor, pipe):
    with torch.no_grad():
        latents_pil_img = pipe.numpy_to_pil(pipe.decode_latents(latent.detach()))[0]
        pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    show_images_side_by_side([latents_pil_img, pil_img], ['Latent','Generated Image'])
    return

def save_img(path, img: torch.Tensor, pipe):
    pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    pil_img.save(path)
    return pil_img

def save_img_tensor(path, img: torch.Tensor):
    pil_img = numpy_to_pil(img_tensor_to_numpy(img))[0]
    pil_img.save(path)
    return pil_img


def img_tensor_to_numpy(tensor):
    return tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def get_img_tensor(img_path, device):
    img_tensor = pil_to_tensor(Image.open(img_path).convert("RGB"))/255
    return img_tensor.unsqueeze(0).to(device)

def create_output_folder(cfgs):
    parent = os.path.join(cfgs['save_img'], cfgs['dataset'])
    wm_path = os.path.join(parent, cfgs['method'], cfgs['case'])
    
    special_model = ['CompVis']
    for key in special_model:
        if key in cfgs['model_id']:
            wm_path = os.path.join(parent, cfgs['method'], '_'.join([cfgs['case'][:-1], key+'/']))
            break
        
    os.makedirs(wm_path, exist_ok=True)
    ori_path = os.path.join(parent, 'OriImgs/')
    os.makedirs(ori_path, exist_ok=True)
    return wm_path, ori_path

# Metrics for similarity
def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return 20 * math.log10(1.) - 10 * math.log10(mse)

# def compute_msssim(a, b):
#     return ms_ssim(a, b, data_range=1.).item()

# def compute_ssim(a, b):
#     return ssim(a, b, data_range=1.).item()

def load_img(img_path, device):
    img = Image.open(img_path).convert('RGB')
    x = (transforms.ToTensor()(img)).unsqueeze(0).to(device)
    return x

# def eval_psnr_ssim_msssim(ori_img_path, new_img_path, device):
#     ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
#     return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x), compute_msssim(ori_x, new_x)

def eval_lpips(ori_img_path, new_img_path, metric, device):
    ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
    return metric(ori_x, new_x).item()

# Detect watermark from one image
def watermark_prob(img, dect_pipe, wm_pipe, text_embeddings, tree_ring=True, device=torch.device('cuda'), return_distance=False, if_original=False):
    if isinstance(img, str):
        img_tensor = pil_to_tensor(Image.open(img).convert("RGB"))/255
        img_tensor = img_tensor.unsqueeze(0).to(device)
    elif isinstance(img, torch.Tensor):
        img_tensor = img

    img_latents = dect_pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = dect_pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1.0,
        num_inference_steps=50,
    )
    det_prob = wm_pipe.one_minus_p_value(reversed_latents) if not tree_ring else wm_pipe.tree_ring_p_value(reversed_latents)
    l1 = wm_pipe.eval_watermark(reversed_latents)
    if return_distance:
        return det_prob, l1
    else:
        return det_prob


# Detect watermark from one image
def watermark_prob_xl(img, dect_pipe, wm_pipe, text_embeddings, tree_ring=True, device=torch.device('cuda'), return_distance=False, if_original=False):
    if isinstance(img, str):
        img_tensor = pil_to_tensor(Image.open(img).convert("RGB"))/255
        img_tensor = img_tensor.unsqueeze(0).to(device)
    elif isinstance(img, torch.Tensor):
        img_tensor = img

    img_latents = dect_pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = dect_pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1.0,
        num_inference_steps=50,
    )
    det_prob = wm_pipe.one_minus_p_value(reversed_latents) if not tree_ring else wm_pipe.tree_ring_p_value(reversed_latents)
    l1 = wm_pipe.eval_watermark(reversed_latents)
    if return_distance:
        return det_prob, l1
    else:
        return det_prob


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

def get_init_latent(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
    # DDIM inversion from the given image
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )
    return reversed_latents

def get_init_latent_list(img_tensor, pipe, text_embeddings, guidance_scale=1.0):
    # DDIM inversion from the given image
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
        return_latents_list=True
    )
    reversed_latents = [img_latents] + reversed_latents
    return reversed_latents

def get_init_latent_list_xl(img_tensor, pipe, guidance_scale=1.0):
    # DDIM inversion from the given image
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        prompt="", negative_prompt="",
        latents=img_latents,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
        return_latents_list=True
    )
    reversed_latents = [img_latents] + reversed_latents
    return reversed_latents


def get_init_latent_xl(img_tensor, pipe, text_embeddings, guidance_scale=1.0, prompt=''):
    # DDIM inversion from the given image
    img_latents = pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = pipe.forward_diffusion(
        prompt="", negative_prompt="",
        latents=img_latents,
        guidance_scale=1.,
        num_inference_steps=50,
        output_type='latent', return_dict=False
    )[0]

    return reversed_latents




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
            cur_time 
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

def normalization(image):
    max_val = image.max()
    min_val = image.min()
    image = (image - min_val) / (max_val - min_val)
    return image

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


def blend(gt_img_tensor, wm_img_tensor, threshold=0.92, lower=0., upper=1., precision=1e-6, max_iter=1000):
      
    for i in range(max_iter):
        mid_theta = (lower + upper) / 2
        img_tensor = (gt_img_tensor-wm_img_tensor)*mid_theta+wm_img_tensor
        ssim_value = ssim(img_tensor, gt_img_tensor).item()

        if ssim_value <= threshold:
            lower = mid_theta
        else:
            upper = mid_theta
        if upper - lower < precision:
            break
        
    img_tensor = (gt_img_tensor-wm_img_tensor)*lower+wm_img_tensor
    return img_tensor

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)