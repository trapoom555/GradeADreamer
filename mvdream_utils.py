import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mvdream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from mvdream.ldm.models.diffusion.ddim import DDIMSampler

from diffusers import DDIMScheduler

from peft import LoraConfig, get_peft_model

from utils.grad_helper import SpecifyGradient

class MVDream(nn.Module):
    def __init__(
        self,
        device,
        model_name='sd-v2.1-base-4view',
        ckpt_path=None,
        opt=None,
    ):
        super().__init__()

        self.opt = opt
        self.device = device
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        self.model = build_model(self.model_name, ckpt_path=self.ckpt_path).eval().to(self.device)
        self.model.device = device

        # LoRA
        config = LoraConfig(
            r=opt.lora_rank,
            lora_alpha=opt.lora_alpha,
            target_modules=["to_q", "to_v", "query", "value"],
            lora_dropout=opt.lora_dropout,
            bias="none",
        )

        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

        self.dtype = torch.float32

        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=self.dtype
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * opt.t_sampling[0])
        self.max_step = int(self.num_train_timesteps * opt.t_sampling[1])
        self.min_step_detail = int(self.num_train_timesteps * opt.detail_t_sampling[0])
        self.max_step_detail = int(self.num_train_timesteps * opt.detail_t_sampling[1])
        self.min_step_refine = int(self.num_train_timesteps * 0.02)
        self.max_step_refine = int(self.num_train_timesteps * 0.30)

        self.embeddings = {}

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts).repeat(4,1,1)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts).repeat(4,1,1)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        with self.model.disable_adapter():
            embeddings = self.model.get_learned_conditioning(prompt).to(self.device)
        return embeddings
    
    def refine(
        self,
        pred_rgb, # [B, C, H, W], B is multiples of 4
        camera, # [B, 4, 4]
        steps,
        as_latent=False,
    ):
        batch_size = pred_rgb.shape[0]
        real_batch_size = batch_size // 4
        # interp to 256x256 to be fed into vae.
        pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_256)

        if self.opt.anneal_timestep:
            if steps <= self.opt.max_linear_anneal_iters:
                step_ratio = min(1, steps / self.opt.max_linear_anneal_iters)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            elif (steps > self.opt.max_linear_anneal_iters) and (steps <= self.opt.anneal_detail_iters):
                # t ~ U(0.02, 0.98)
                t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)
            else:
                # t ~ U(0.02, 0.50)
                t = torch.randint(self.min_step_detail, self.max_step_detail + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)
            
        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)
        camera = camera.repeat(2, 1)

        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera, "num_frames": 4}

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise ~ N(0,1)
            noise = torch.randn_like(latents)
            with self.model.disable_adapter():
                latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            with self.model.disable_adapter():
                pretrained_noise_pred = self.model.apply_model(latent_model_input, tt, context)
            
            pretrained_noise_pred_uncond, pretrained_noise_pred_pos = pretrained_noise_pred.chunk(2)
            pretrained_noise_pred = pretrained_noise_pred_uncond + self.opt.guidance_scale * (pretrained_noise_pred_pos - pretrained_noise_pred_uncond)
            
        noise_pred = self.model.apply_model(latent_model_input, tt, context)
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.opt.guidance_scale * (noise_pred_pos - noise_pred_uncond)

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
    
    def train_step(
        self,
        pred_rgb, # [B, C, H, W], B is multiples of 4
        camera, # [B, 4, 4]
        steps,
        as_latent=False,
    ):
        
        batch_size = pred_rgb.shape[0]
        ## 4 views per images
        real_batch_size = batch_size // 4
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 256x256 to be fed into vae.
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_256)

        if self.opt.anneal_timestep:
            if steps <= self.opt.max_linear_anneal_iters:
                step_ratio = min(1, steps / self.opt.max_linear_anneal_iters)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            elif (steps > self.opt.max_linear_anneal_iters) and (steps <= self.opt.anneal_detail_iters):
                # t ~ U(0.02, 0.98)
                t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)
            else:
                # t ~ U(0.02, 0.50)
                t = torch.randint(self.min_step_detail, self.max_step_detail + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (real_batch_size,), dtype=torch.long, device=self.device).repeat(4)

        camera = camera[:, [0, 2, 1, 3]] # to blender convention (flip y & z axis)
        camera[:, 1] *= -1
        camera = normalize_camera(camera).view(batch_size, 16)

        camera = camera.repeat(2, 1)
        embeddings = torch.cat([self.embeddings['neg'].repeat(real_batch_size, 1, 1), self.embeddings['pos'].repeat(real_batch_size, 1, 1)], dim=0)
        context = {"context": embeddings, "camera": camera, "num_frames": 4}

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise ~ N(0,1)
            noise = torch.randn_like(latents)
            with self.model.disable_adapter():
                latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            with self.model.disable_adapter():
                pretrained_noise_pred = self.model.apply_model(latent_model_input, tt, context)
            
            pretrained_noise_pred_uncond, pretrained_noise_pred_pos = pretrained_noise_pred.chunk(2)
            pretrained_noise_pred = pretrained_noise_pred_uncond + self.opt.guidance_scale * (pretrained_noise_pred_pos - pretrained_noise_pred_uncond)
            
        noise_pred = self.model.apply_model(latent_model_input, tt, context)
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.opt.guidance_scale * (noise_pred_pos - noise_pred_uncond)

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
        with self.model.disable_adapter():
            imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256]
        imgs = 2 * imgs - 1
        with self.model.disable_adapter():
            latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents # [B, 4, 32, 32]

    @torch.no_grad()
    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        elevation=0,
        azimuth_start=0,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        
        batch_size = len(prompts) * 4

        # Text embeds -> img latents
        sampler = DDIMSampler(self.model)
        shape = [4, height // 8, width // 8]
        c_ = {"context": self.encode_text(prompts).repeat(4,1,1)}
        uc_ = {"context": self.encode_text(negative_prompts).repeat(4,1,1)}

        camera = get_camera(4, elevation=elevation, azimuth_start=azimuth_start)
        camera = camera.repeat(batch_size // 4, 1).to(self.device)

        c_["camera"] = uc_["camera"] = camera
        c_["num_frames"] = uc_["num_frames"] = 4

        latents, _ = sampler.sample(S=num_inference_steps, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=guidance_scale,
                                        unconditional_conditioning=uc_,
                                        eta=0, x_T=None)

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [4, 3, 256, 256]

        return imgs


if __name__ == "__main__":
    import argparse
    import torchvision
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    device = torch.device("cuda")

    # seed
    seed = 7
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    sd = MVDream(device, opt=opt)

    imgs = sd.prompt_to_img(args.prompt, args.negative, num_inference_steps=args.steps)

    torchvision.utils.save_image(imgs, 'logs/img_2D_gen.jpg')
