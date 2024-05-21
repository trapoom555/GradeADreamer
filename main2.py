import os
import tqdm
import numpy as np

import torch
import torchvision
import torch.nn.functional as F

from utils.cam_utils import orbit_camera, OrbitCamera
from mesh_renderer import Renderer
from torch.optim import Adam

from mvdream_utils import MVDream

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None
        self.guidance_sd = None

        # renderer
        self.renderer = Renderer(opt).to(self.device)

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

    def save_model(self, path):    
        path = os.path.join(opt.outdir, opt.outname, opt.outname + '_refined_mesh.' + opt.mesh_format)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
              
    def prepare_train(self):
        
        self.step = 0

        # setup training
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

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
            step_ratio = min(1, self.step / self.opt.iters_refine)

            loss = 0

            ### novel view (manual batch)
            render_resolution = 512
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

                # no random render resolution
                ssaa = 1
                out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                image = out["image"] # [H, W, 3] in [0, 1]
                image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                images.append(image)

                # enable mvdream training
                for view_i in range(1, 4):
                    pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                    poses.append(pose_i)

                    out_i = self.renderer.render(pose_i, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                    image = out_i["image"].permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            path = os.path.join(self.opt.outdir, self.opt.outname)
            torchvision.utils.save_image(images, os.path.join(path, 'img_refine.jpg'))
            if self.step % 100 == 0:
                torchvision.utils.save_image(images, os.path.join(path, f'texture_{self.step}.jpg'))

            # guidance loss
            guide_loss, lora_loss = self.guidance_sd.refine(images, poses, steps=self.step)
            loss = loss + self.opt.lambda_sd * guide_loss

            # lora step
            lora_loss.backward(retain_graph=True)
            self.lora_optimizer.step()
            self.lora_optimizer.zero_grad()
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()

        # save model
        path = os.path.join(self.opt.outdir, self.opt.outname, self.opt.outname + '_refined_mesh.' + self.opt.mesh_format)
        self.save_model(path)


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

    # auto find mesh from stage 1
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.outname, opt.outname + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")

    # train
    trainer = Trainer(opt)
    trainer.train(opt.iters_refine)