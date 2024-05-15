import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mvdream.camera_utils import get_camera, convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model
from mvdream.ldm.models.diffusion.ddim import DDIMSampler

from diffusers import DDIMScheduler

from peft import LoraConfig, get_peft_model

class MVDream(nn.Module):
    def __init__(
        self,
        device,
        model_name='sd-v2.1-base-4view',
        ckpt_path=None,
        opt=None,
        t_range=[0.02, 0.98],
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
            target_modules=["to_q", "to_v"],
            lora_dropout=opt.lora_dropout,
            bias="none",
        )

        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

        self.dtype = torch.float32

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.embeddings = {}

        self.scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=self.dtype
        )

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

    def train_step(
        self,
        pred_rgb, # [B, C, H, W], B is multiples of 4
        camera, # [B, 4, 4]
        step_ratio=None,
        guidance_scale=100,
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

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
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
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            with self.model.disable_adapter():
                pretrained_noise_pred = self.model.apply_model(latent_model_input, tt, context)
            
            pretrained_noise_pred_uncond, pretrained_noise_pred_pos = pretrained_noise_pred.chunk(2)
            pretrained_noise_pred = pretrained_noise_pred_uncond + guidance_scale * (pretrained_noise_pred_pos - pretrained_noise_pred_uncond)
            
        noise_pred = self.model.apply_model(latent_model_input, tt, context)
        # assert torch.equal(noise_pred, pretrained_noise_pred) ## Assert error, cool ! lora makes sense
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        ## [HERE] Noise_pred - noise for guidance
        grad = (pretrained_noise_pred - noise_pred)
        grad = torch.nan_to_num(grad)
        # seems important to avoid NaN...
        grad = grad.clamp(-self.opt.grad_clip, self.opt.grad_clip)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        # LoRA
        grad_lora = (noise_pred - noise)
        grad_lora = torch.nan_to_num(grad_lora)
        grad_lora = grad.clamp(-self.opt.lora_grad_clip, self.opt.lora_grad_clip)
        target = (latents - grad_lora).detach()
        loss_lora = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss, loss_lora

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256]
        imgs = 2 * imgs - 1
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
        
        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument("--steps", type=int, default=30)
    opt = parser.parse_args()

    device = torch.device("cuda")

    sd = MVDream(device)

    while True:
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, num_inference_steps=opt.steps)

        grid = np.concatenate([
            np.concatenate([imgs[0], imgs[1]], axis=1),
            np.concatenate([imgs[2], imgs[3]], axis=1),
        ], axis=0)

        # visualize image
        plt.imshow(grid)
        plt.show()
