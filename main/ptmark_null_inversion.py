from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptmark_prompt_utils as ptp_utils
# import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
from ptmark_optim_utils import *
from ptmark_watermark import GTWatermark
import yaml

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image



class LocalBlend:
    
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th


        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_word)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon, logger=None, writer=None):
        # self.init_prompt(prompt="")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1].clone()
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        global_step=0
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2].clone()
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # if writer is not None:
                #     writer.add_scalar('loss', loss_item, global_step)
                global_step += 1
                # print(f"Step {global_step}: Loss = {loss_item}")
                # if logger is not None:
                #     logger.info(f"Step {global_step}: Loss = {loss_item}")
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
                # latent_cur_noise = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                # latent_cur = self.prev_step(latent_cur_noise, t, latent_cur)
           
        bar.close()
        return uncond_embeddings_list
    

    def null_optimization_double(self, latents, wm_latents, num_inner_steps, epsilon, logger=None, writer=None, wm_pipe=None, watarmark_loss_weight=0, semantic_loss_weight=1):
        # self.init_prompt(prompt="")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1].clone()
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        global_step=0
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2].clone()
            wm_latents_prev = wm_latents[i].clone()
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = semantic_loss_weight * nnf.mse_loss(latents_prev_rec, latent_prev) + watarmark_loss_weight * watermark_loss(latents_prev_rec, wm_latents_prev, wm_pipe)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # if writer is not None:
                #     writer.add_scalar('loss', loss_item, global_step)
                global_step += 1
                # print(f"Step {global_step}: Loss = {loss_item}")
                # if logger is not None:
                #     logger.info(f"Step {global_step}: Loss = {loss_item}")
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
                # latent_cur_noise = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                # latent_cur = self.prev_step(latent_cur_noise, t, latent_cur)
           
        bar.close()
        return uncond_embeddings_list
    

    def null_optimization_double_fast(self, latents, wm_latents, num_inner_steps, epsilon, logger=None, writer=None, wm_pipe=None, watarmark_loss_weight=0, semantic_loss_weight=1, early_stop_patience=3):
        # self.init_prompt(prompt="")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1].clone()
        
        # 计算总进度条长度（考虑动态的num_inner_steps）
        total_steps = 0
        for i in range(NUM_DDIM_STEPS):
            # 根据timestep位置确定迭代次数
            if i < NUM_DDIM_STEPS * 0.2:  # 前20%
                n_steps = 10
            elif i < NUM_DDIM_STEPS * 0.7:  # 中间50%
                n_steps = 5
            else:  # 后30%
                n_steps = 1
            total_steps += n_steps
        
        bar = tqdm(total=total_steps)
        global_step=0
        for i in range(NUM_DDIM_STEPS):
            # 迭代次数调度：根据timestep位置动态调整num_inner_steps
            if i < NUM_DDIM_STEPS * 0.2:  # 前20%
                current_num_inner_steps = 10
            elif i < NUM_DDIM_STEPS * 0.7:  # 中间50%
                current_num_inner_steps = 5
            else:  # 后30%
                current_num_inner_steps = 1
            
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2].clone()
            wm_latents_prev = wm_latents[i].clone()
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            
            # Adaptive Early stopping: 自适应早停策略
            # 策略说明：
            # 1. 根据timestep位置动态调整patience和阈值（早期更宽松，后期更激进）
            # 2. 使用多重条件判断：绝对阈值、相对改进、损失趋势、无改进计数
            # 3. 跟踪损失历史，分析损失下降趋势
            loss_history = []  # 记录损失历史用于趋势分析
            min_loss = float('inf')
            no_improvement_count = 0
            # 根据timestep位置自适应调整patience和阈值
            if i < NUM_DDIM_STEPS * 0.2:  # 前20%: 更宽松的早停
                adaptive_patience = max(early_stop_patience, 5)
                adaptive_threshold = epsilon + i * 2e-5
                min_improvement_ratio = 0.001  # 最小改进比例
            elif i < NUM_DDIM_STEPS * 0.7:  # 中间50%: 中等早停
                adaptive_patience = max(early_stop_patience, 3)
                adaptive_threshold = epsilon + i * 2e-5
                min_improvement_ratio = 0.0005
            else:  # 后30%: 更激进的早停
                adaptive_patience = max(early_stop_patience, 2)
                adaptive_threshold = epsilon + i * 3e-5  # 稍微放宽阈值
                min_improvement_ratio = 0.0001
            
            actual_steps = 0
            initial_loss = None
            
            for j in range(current_num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = semantic_loss_weight * nnf.mse_loss(latents_prev_rec, latent_prev) + watarmark_loss_weight * watermark_loss(latents_prev_rec, wm_latents_prev, wm_pipe)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                
                # 记录初始损失
                if initial_loss is None:
                    initial_loss = loss_item
                
                # 更新最小损失
                if loss_item < min_loss:
                    min_loss = loss_item
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # 记录损失历史（保留最近几次用于趋势分析）
                loss_history.append(loss_item)
                if len(loss_history) > 5:
                    loss_history.pop(0)
                
                # if writer is not None:
                #     writer.add_scalar('loss', loss_item, global_step)
                global_step += 1
                actual_steps += 1
                # print(f"Step {global_step}: Loss = {loss_item}")
                # if logger is not None:
                #     logger.info(f"Step {global_step}: Loss = {loss_item}")
                bar.update()
                
                # Adaptive Early stopping: 多重条件判断
                should_stop = False
                
                # 条件1: 绝对阈值检查（连续低损失）
                if loss_item < adaptive_threshold:
                    # 检查上一次迭代是否也低于阈值（连续两次都低于阈值）
                    if len(loss_history) > 1 and loss_history[-2] < adaptive_threshold:
                        should_stop = True
                    # 或者第一次迭代就低于阈值且非常小
                    elif j == 0 and loss_item < adaptive_threshold * 0.5:
                        should_stop = True
                
                # 条件2: 相对改进检查（损失改进很小）
                if initial_loss is not None and j >= 2:
                    relative_improvement = (initial_loss - loss_item) / (initial_loss + 1e-8)
                    if relative_improvement < min_improvement_ratio and loss_item < initial_loss * 0.95:
                        should_stop = True
                
                # 条件3: 损失趋势检查（损失不再下降或下降很慢）
                if len(loss_history) >= 3:
                    recent_trend = loss_history[-1] - loss_history[-3]  # 最近3步的变化
                    if recent_trend > -min_improvement_ratio * initial_loss and loss_item < initial_loss * 0.9:
                        should_stop = True
                
                # 条件4: 无改进计数（连续多次无改进）
                if no_improvement_count >= adaptive_patience and j >= 2:
                    should_stop = True
                
                if should_stop:
                    break
            
            # 如果提前停止，更新进度条以反映跳过的步骤
            skipped_steps = current_num_inner_steps - actual_steps
            if skipped_steps > 0:
                bar.update(skipped_steps)
            
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
                # latent_cur_noise = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                # latent_cur = self.prev_step(latent_cur_noise, t, latent_cur)
           
        bar.close()
        return uncond_embeddings_list
    
    def null_optimization_double_fast_gemini(self, latents, wm_latents, num_inner_steps, epsilon, logger=None, writer=None, wm_pipe=None, watarmark_loss_weight=0, semantic_loss_weight=1):
        # self.init_prompt(prompt="")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1].clone()
        
        # 注意：这里的 total 只是一个估算，因为我们会动态调整步数，进度条可能会有些跳跃
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        global_step = 0
        
        for i in range(NUM_DDIM_STEPS):
            # ==========================================
            # 策略 2: 迭代次数调度 (Time-step Scheduling)
            # ==========================================
            step_ratio = i / NUM_DDIM_STEPS
            if step_ratio < 0.2:
                cur_num_inner_steps = num_inner_steps      # 前 20%: 保持 N=10 (全力优化)
            elif step_ratio < 0.7:
                cur_num_inner_steps = int(num_inner_steps / 2) # 中间 50%: N=5 (适度优化)
            else:
                cur_num_inner_steps = 1                    # 后 30%: N=1 (仅做一次微调或不做)
                # 如果想完全跳过优化（N=0），改为: cur_num_inner_steps = 0
            
            # 准备优化变量
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            
            # 学习率也可以随 timestep 衰减，保持原逻辑
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            
            latent_prev = latents[len(latents) - i - 2].clone()
            wm_latents_prev = wm_latents[i].clone()
            t = self.model.scheduler.timesteps[i]
            
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            
            # 用于早停的计数器
            consecutive_success_count = 0 
            prev_loss_item = float('inf')
            
            # 如果 cur_num_inner_steps 为 0，这个循环直接跳过，保留上一轮的 uncond_embeddings
            for j in range(cur_num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                
                loss = semantic_loss_weight * nnf.mse_loss(latents_prev_rec, latent_prev) + watarmark_loss_weight * watermark_loss(latents_prev_rec, wm_latents_prev, wm_pipe)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                
                # 更新全局步数
                global_step += 1
                bar.update() # 实际跑的步数
                
                # ==========================================
                # 策略 1: Adaptive Early Stopping (自适应早停)
                # ==========================================
                
                # 动态阈值计算 (保持你原有的逻辑，越靠后阈值越宽)
                current_epsilon = epsilon + i * 2e-5
                
                # 判断条件 A: Loss 绝对值达标
                is_low_loss = loss_item < current_epsilon
                
                # 判断条件 B: Loss 收敛 (下降幅度极小)
                # 例如：如果 Loss 变化小于 1e-5，说明优化不动了
                loss_diff = abs(prev_loss_item - loss_item)
                is_converged = loss_diff < 1e-5
                
                if is_low_loss or (is_converged and j > 2): # 至少跑2-3步再判断收敛，防止初始震荡
                    consecutive_success_count += 1
                else:
                    consecutive_success_count = 0 # 重置计数器
                
                prev_loss_item = loss_item
                
                # 连续 2 次满足条件即停止 (避免单次噪点导致的误判)
                if consecutive_success_count >= 2:
                    # 触发早停前，把剩余的进度条补齐，保持 bar 的显示美观（可选）
                    # bar.update(cur_num_inner_steps - j - 1) 
                    break

            # 如果因为 Scheduling 减少了步数，这里需要补齐 tqdm，
            # 或者直接忽略（因为我们在上面逻辑里只更新了实际跑的步数）
            # 这里保留你原本的逻辑来补齐进度条，防止 tqdm 报错或显示不准
            # 注意：这里要补齐的是 (num_inner_steps - 实际跑的 j)，而不是 cur_num_inner_steps
            # 但因为我们改变了总计划，建议不需要刻意补齐 num_inner_steps，
            # 只需要保证当前 step 结束即可。
            
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
                
        bar.close()
        return uncond_embeddings_list



    def null_optimization_double_xl(self, latents, wm_latents, num_inner_steps, epsilon, logger=None, writer=None, wm_pipe=None, watarmark_loss_weight=0, semantic_loss_weight=1, added_cond_kwargs=None):
        # self.init_prompt(prompt="")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1].clone()
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        global_step=0
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2].clone()
            wm_latents_prev = wm_latents[i].clone()
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = semantic_loss_weight * nnf.mse_loss(latents_prev_rec, latent_prev) + watarmark_loss_weight * watermark_loss(latents_prev_rec, wm_latents_prev, wm_pipe)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # if writer is not None:
                #     writer.add_scalar('loss', loss_item, global_step)
                global_step += 1
                # print(f"Step {global_step}: Loss = {loss_item}")
                # if logger is not None:
                #     logger.info(f"Step {global_step}: Loss = {loss_item}")
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
                # latent_cur_noise = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                # latent_cur = self.prev_step(latent_cur_noise, t, latent_cur)
           
        bar.close()
        return uncond_embeddings_list
    
    def null_optimization_double_MSE(self, latents, wm_latents, num_inner_steps, epsilon, logger=None, writer=None, wm_pipe=None, watarmark_loss_weight=0):
        # self.init_prompt(prompt="")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1].clone()
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        global_step=0
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2].clone()
            wm_latents_prev = wm_latents[i].clone()
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev) + watarmark_loss_weight * watermark_loss(latents_prev_rec, wm_latents_prev, wm_pipe)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # if writer is not None:
                #     writer.add_scalar('loss', loss_item, global_step)
                global_step += 1
                print(f"Step {global_step}: Loss = {loss_item}")
                # if logger is not None:
                #     logger.info(f"Step {global_step}: Loss = {loss_item}")
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
                # latent_cur_noise = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                # latent_cur = self.prev_step(latent_cur_noise, t, latent_cur)
           
        bar.close()
        return uncond_embeddings_list
    

    def null_optimization_reverse(self, latents, num_inner_steps, epsilon, logger=None, writer=None):
        # self.init_prompt(prompt="")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[0].clone()
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        global_step=0
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[i + 1].clone()
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if writer is not None:
                    writer.add_scalar('loss', loss_item, global_step)
                global_step += 1
                print(f"Step {global_step}: Loss = {loss_item}")
                if logger is not None:
                    logger.info(f"Step {global_step}: Loss = {loss_item}")
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
           
        bar.close()
        return uncond_embeddings_list

    def null_optimization_KL(self, latents, num_inner_steps, epsilon, logger=None, writer=None):
        # self.init_prompt(prompt="")
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1].clone()
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        global_step=0
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2].clone()
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev) + 10 * KL_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                # if writer is not None:
                #     writer.add_scalar('loss', loss_item, global_step)
                global_step += 1
                print(f"Step {global_step}: Loss = {loss_item}")
                if logger is not None:
                    logger.info(f"Step {global_step}: Loss = {loss_item}")
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
                # latent_cur_noise = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                # latent_cur = self.prev_step(latent_cur_noise, t, latent_cur)
           
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        # watermarking_image = load_512('./data/watermark_imgs/0.png', *offsets)
        # _, watermarking_ddim_latents = self.ddim_inversion(watermarking_image)
        # ddim_latents[-1] = watermarking_ddim_latents[-1]
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model, num_inner_steps=10, NUM_DDIM_STEPS=50, epsilon=1e-5):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False, )
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.num_inner_steps = num_inner_steps
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.epsilon = epsilon


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent



def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True):
    if run_baseline:
        print("w.o. zz")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t


# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# ldm_stable = StableDiffusionPipeline.from_pretrained("path/to/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
# try:
#     ldm_stable.disable_xformers_memory_efficient_attention()
# except AttributeError:
#     print("Attribute disable_xformers_memory_efficient_attention() is missing")
# tokenizer = ldm_stable.tokenizer

# null_inversion = NullInversion(ldm_stable)

# image_path = "./data/no_watermark_imgs/0.png"
# prompt = ""
# (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True, num_inner_steps=20)

# print("Modify or remove offsets according to your image!")

# prompts = [prompt]
# controller = AttentionStore()
# image_inv, x_t = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False)
# print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
# Image.fromarray(image_inv[0]).save('invert_image_embedding.png')
# pil_image = ptp_utils.view_images([image_gt, image_enc, image_inv[0]])
# pil_image.save('compare.png')
# show_cross_attention(controller, 16, ["up", "down"])

