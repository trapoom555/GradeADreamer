from diffusers import (
    DDIMScheduler,
    StableDiffusion3Pipeline
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        model_name='stabilityai/stable-diffusion-2-1-base',
        opt=None,
    ):
        model_name = "stabilityai/stable-diffusion-3-medium-diffusers"

        super().__init__()

        self.opt = opt
        self.device = device

        self.dtype = torch.float32

        # Create model
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(device)
        # pipe.transformer.to(memory_format=torch.channels_last)
        # pipe.vae.to(memory_format=torch.channels_last)

        # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

        print(pipe.__dict__)

        self.pipe = pipe
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.model = pipe.transformer

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
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(prompt=prompts, prompt_2=None, prompt_3=None, negative_prompt=negative_prompts, num_images_per_prompt=1, do_classifier_free_guidance=True)
        self.embeddings['pos'] = prompt_embeds
        self.embeddings['neg'] = negative_prompt_embeds
        self.embeddings['pos_pooled'] = pooled_prompt_embeds
        self.embeddings['neg_pooled'] = negative_pooled_prompt_embeds
        # pos_embeds, pos_pooled_embeds = self.encode_text(prompts)
        # neg_embeds, neg_pooled_embeds = self.encode_text(negative_prompts)
        # self.embeddings['pos'] = pos_embeds
        # self.embeddings['neg'] = neg_embeds
        # self.embeddings['pos_pooled'] = pos_pooled_embeds
        # self.embeddings['neg_pooled'] = neg_pooled_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(prompt=[f'{p}, {d} view' for p in prompts], prompt_2=None, prompt_3=None, num_images_per_prompt=1, do_classifier_free_guidance=True)
            self.embeddings[d] = prompt_embeds
            self.embeddings[f'{d}_pooled'] = pooled_prompt_embeds
            # embeds, pooled_embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            # self.embeddings[d] = embeds
            # self.embeddings[f'{d}_pooled'] = pooled_embeds
    
    # def encode_text(self, prompt):
    #     # inputs = self.tokenizer(
    #     #     prompt,
    #     #     padding="max_length",
    #     #     max_length=self.tokenizer.model_max_length,
    #     #     return_tensors="pt",
    #     # )
    #     # outputs = self.text_encoder(inputs.input_ids.to(self.device))
    #     # embeddings = outputs[0]
    #     # pooled_embeddings = outputs[1]  # Assuming the second output is the pooled embeddings
    #     # return embeddings, pooled_embeddings
    #     (
    #         prompt_embeds,
    #         negative_prompt_embeds,
    #         pooled_prompt_embeds,
    #         negative_pooled_prompt_embeds,
    #     ) = self.pipe.encode_prompt(prompt=prompt, num_images_per_prompt=1, do_classifier_free_guidance=True)


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
        latents = latents.to(self.dtype)

        with torch.no_grad():
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            tt = tt.expand(latent_model_input.shape[0])

            embeddings = torch.cat([self.embeddings['pos'], self.embeddings['neg'].expand(batch_size, -1, -1)])
            pooled_embeddings = torch.cat([self.embeddings['pos_pooled'].expand(batch_size, -1), self.embeddings['neg_pooled'].expand(batch_size, -1)])

            noise_pred = self.model(
                hidden_states=latent_model_input,
                timestep=tt, 
                encoder_hidden_states=embeddings,
                pooled_projections=pooled_embeddings,
                return_dict=False
            )[0]

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.opt.guidance_scale * (noise_pred_text - noise_pred_uncond)

            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)
            # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents