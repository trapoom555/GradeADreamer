import os
import torch
import numpy as np
import nvdiffrast.torch as dr
from gradeadreamer.renderer.gs_renderer import MiniCam
import torch.nn.functional as F
from gradeadreamer.utils.cam_utils import orbit_camera
from gradeadreamer.utils.mesh import Mesh, safe_normalize
from gradeadreamer.utils.grid_put import mipmap_linear_grid_put_2d

@torch.no_grad()
def save_model(self, texture_size=1024):
    path = os.path.join(self.opt.outdir, self.opt.outname, self.opt.outname + '_mesh.' + self.opt.mesh_format)
    mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

    # perform texture extraction
    print(f"[INFO] unwrap uv...")
    h = w = texture_size
    mesh.auto_uv()
    mesh.auto_normal()

    albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
    cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

    vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
    hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

    render_resolution = 512

    glctx = dr.RasterizeCudaContext()

    for ver, hor in zip(vers, hors):
        # render image
        pose = orbit_camera(ver, hor, self.cam.radius)

        cur_cam = MiniCam(
            pose,
            render_resolution,
            render_resolution,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        
        cur_out = self.renderer.render(cur_cam)

        rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

        # enhance texture quality with zero123 [not working well]
        # if self.opt.guidance_model == 'zero123':
        #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
            # import kiui
            # kiui.vis.plot_image(rgbs)
            
        # get coordinate in texture image
        pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
        proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

        v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T
        rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
        depth = depth.squeeze(0) # [H, W, 1]

        alpha = (rast[0, ..., 3:] > 0).float()

        uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

        # use normal to produce a back-project mask
        normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
        normal = safe_normalize(normal[0])

        # rotated normal (where [0, 0, 1] always faces camera)
        rot_normal = normal @ pose[:3, :3]
        viewcos = rot_normal[..., [2]]

        mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
        mask = mask.view(-1)

        uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
        rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
        
        # update texture image
        cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
            h, w,
            uvs[..., [1, 0]] * 2 - 1,
            rgbs,
            min_resolution=256,
            return_count=True,
        )
        
        # albedo += cur_albedo
        # cnt += cur_cnt
        mask = cnt.squeeze(-1) < 0.1
        albedo[mask] += cur_albedo[mask]
        cnt[mask] += cur_cnt[mask]

    mask = cnt.squeeze(-1) > 0
    albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

    mask = mask.view(h, w)

    albedo = albedo.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    # dilate texture
    from sklearn.neighbors import NearestNeighbors
    from scipy.ndimage import binary_dilation, binary_erosion

    inpaint_region = binary_dilation(mask, iterations=32)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
        search_coords
    )
    _, indices = knn.kneighbors(inpaint_coords)

    albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

    mesh.albedo = torch.from_numpy(albedo).to(self.device)
    mesh.write(path)