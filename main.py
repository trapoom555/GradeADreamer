import os
import tqdm
import numpy as np

import torch
import torchvision
import torchvision.transforms.functional as T

from utils.cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam
from torch.optim import Adam

from mvdream_utils import MVDream
from utils.save_model import save_model

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None
        self.guidance_sd = None

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.optimizer = None
        self.lora_optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # initialize gaussians to a blob
        self.renderer.initialize(num_pts=self.opt.num_pts)

    def prepare_train(self):
        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # lazy load guidance model
        print(f"[INFO] loading MVDream...")
        self.guidance_sd = MVDream(self.device, opt=self.opt)
        print(f"[INFO] loaded MVDream!")
            
        self.lora_optimizer = Adam(self.guidance_sd.parameters(), lr=self.opt.lora_lr)

        # prepare embeddings
        with torch.no_grad():
            self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):
            self.step += 1

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### novel view (manual batch)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, self.opt.render_resolution, self.opt.render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

                # mvdream 4 views
                for view_i in range(1, 4):
                    pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                    poses.append(pose_i)

                    cur_cam_i = MiniCam(pose_i, self.opt.render_resolution, self.opt.render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                    # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                    out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                    image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)
                    
            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            path = os.path.join(self.opt.outdir, self.opt.outname)
            torchvision.utils.save_image(images, os.path.join(path, 'img.jpg'))
            if self.step % 100 == 0:
                torchvision.utils.save_image(images, os.path.join(path, f'{self.step}.jpg'))

            # guidance loss
            guide_loss, lora_loss = self.guidance_sd.train_step(images, poses, steps=self.step)
            loss = loss + self.opt.lambda_sd * guide_loss

            # lora step
            lora_loss.backward(retain_graph=True)
            self.lora_optimizer.step()
            self.lora_optimizer.zero_grad()
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)
    
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)

        # save model
        save_model(self)
        # save pointclouds
        path = os.path.join(self.opt.outdir, self.opt.outname)
        self.renderer.gaussians.save_ply(os.path.join(path, f'{self.step}.ply'))


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # seed
    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # train
    trainer = Trainer(opt)
    trainer.train(opt.iters)