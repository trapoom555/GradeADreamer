from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.grad_helper import SpecifyGradient

from peft import LoraConfig, get_peft_model


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        model_name='stabilityai/stable-diffusion-2-1-base',
        ckpt_path=None,
        opt=None,
    ):
        super().__init__()

        self.opt = opt
        self.device = device
        
        # LoRA
        config = LoraConfig(
            r=opt.lora_rank,
            lora_alpha=opt.lora_alpha,
            target_modules=["to_q", "to_v", "query", "value"],
            lora_dropout=opt.lora_dropout,
            bias="none",
        )

        self.dtype = torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=self.dtype
        )

        pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.model = pipe.unet

        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

        self.scheduler = DDIMScheduler.from_pretrained(
            model_name, subfolder="scheduler", torch_dtype=self.dtype
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_sampling[0])
        self.max_step = int(self.num_train_timesteps * opt.t_sampling[1])
        self.min_step_detail = int(self.num_train_timesteps * opt.detail_t_sampling[0])
        self.max_step_detail = int(self.num_train_timesteps * opt.detail_t_sampling[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step(
        self,
        pred_rgb, # [B, C, H, W], B is multiples of 4
        steps,
        as_latent=False
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        if self.opt.anneal_timestep:
            if steps <= self.opt.max_linear_anneal_iters:
                step_ratio = min(1, steps / self.opt.max_linear_anneal_iters)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            elif (steps > self.opt.max_linear_anneal_iters) and (steps <= self.opt.anneal_detail_iters):
                # t ~ U(0.02, 0.98)
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
            else:
                # t ~ U(0.02, 0.50)
                t = torch.randint(self.min_step_detail, self.max_step_detail + 1, (batch_size,), dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise ~ N(0,1)
            noise = torch.randn_like(latents)
            with self.model.disable_adapter():
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            with self.model.disable_adapter():
                pretrained_noise_pred = self.model(latent_model_input, tt, encoder_hidden_states=embeddings).sample

            pretrained_noise_pred_cond, pretrained_noise_pred_uncond = pretrained_noise_pred.chunk(2)
            pretrained_noise_pred = pretrained_noise_pred_uncond + self.opt.guidance_scale * (pretrained_noise_pred_cond - pretrained_noise_pred_uncond)
            
        noise_pred = self.model(latent_model_input, tt, encoder_hidden_states=embeddings).sample
        # perform guidance (high scale from paper!)
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.opt.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        w = (1 - self.alphas[t[0]])

        ## [HERE] Noise_pred - noise for guidance
        grad = w * (pretrained_noise_pred - noise_pred)
        grad = torch.nan_to_num(grad)
        # seems important to avoid NaN...
        grad = grad.clamp(-self.opt.grad_clip, self.opt.grad_clip)
        loss = SpecifyGradient.apply(latents, grad)

        # LoRA
        grad_lora = (noise_pred - noise)
        grad_lora = torch.nan_to_num(grad_lora)
        grad_lora = torch.clamp(grad_lora, -self.opt.lora_grad_clip, self.opt.lora_grad_clip)
        loss_lora = SpecifyGradient.apply(latents, grad_lora)

        return loss, loss_lora

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        with self.model.disable_adapter():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist

        with self.model.disable_adapter():
            latents = posterior.sample() * self.vae.config.scaling_factor

        return latents