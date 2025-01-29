import torch
import numpy as np
import math
from tqdm import tqdm
import torch.nn.functional as F

from raytracelib import RayTracer
from raytracelib.utils.texture import sample_texture
from raytracelib.renderers.base_renderer import BaseRenderer


def eval_sh(sh, dirs, degree):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.

    Args:
        sh (torch.tensor): SH coeffs [..., channels, (self.degree + 1) ** 2]
        dirs (torch.tensor): unit directions [..., 3]
        deg (int): SH deg. Currently, 0-4 supported

    Returns:
        [..., channels]
    """

    # hardcoded SH polynomials
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    # normalize dirs (just to be sure)
    dirs = F.normalize(dirs, dim=-1)

    result = C0 * sh[..., 0]

    if degree > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if degree > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )
                if degree > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


class MeshRenderer(BaseRenderer):
    def __init__(
        self,
        tensor_meshes=[],
        t_near=1e-3,
        t_far=100,
        profiler=None,
        render_all_shaders=False,
    ):
        # super
        super().__init__(
            renders_modes=["ray-traced"],
            renders_shaders=[
                "hits",
                "triangle_mesh_id",
                "triangle_id",
                "positions",
                "normals",
                "depth",
                "view_dirs",
            ],
            profiler=profiler,
        )

        if not isinstance(tensor_meshes, list):
            self.tensor_meshes = [tensor_meshes]
        else:
            self.tensor_meshes = tensor_meshes

        self.t_near = t_near
        self.t_far = t_far
        self.renders_options = {"render_second_hit": False}

        # bg
        self.default_bg_color = (255, 255, 255)  # rgb
        self.bg_color = (
            torch.tensor(self.default_bg_color, dtype=torch.float32, device="cuda")
            / 255.0
        )

        # prepare raytracer
        self.RT = RayTracer(self.tensor_meshes, self.t_far, pack_meshes=False)

        # which mesh should be rendered
        self.mesh_id = 0

        # activate optional shaders
        if self.tensor_meshes[self.mesh_id].vertices_uvs is not None:
            self.renders_shaders.append("uv")

        # TODO: update
        # if self.tensor_meshes[self.mesh_id].texture is not None:
        #     # RGBA
        #     self.renders_shaders.append("texture")
        #     if self.tensor_meshes[self.mesh_id].texture.shape[-1] > 4:
        #         # Features
        #         self.renders_shaders.append("sh_eval")
        #         if self.tensor_meshes[self.mesh_id].texture.shape[2] == 4:
        #             self.renders_shaders.append("sh_alpha")

        if render_all_shaders:
            self.active_shaders = self.renders_shaders

    @torch.no_grad()
    def shade(self, outputs):
        """convert outputs to frame buffer values"""

        if self.renders_options["render_second_hit"]:
            key_prefix = "sh_"
        else:
            key_prefix = ""

        faces_mesh_id = outputs[key_prefix + "triangles_mesh_id"]  # [N,]
        faces_id = outputs[key_prefix + "triangles_id"]  # [N,]
        depth = outputs[key_prefix + "depth"]  # [N,]
        positions = outputs[key_prefix + "positions"]  # [N, 3]
        normals = outputs[key_prefix + "normals"]  # [N, 3]
        is_hit = outputs[key_prefix + "is_hit"]  # [N,]
        barycentric = outputs[key_prefix + "barycentric"]  # [N, 3]
        view_dirs = outputs[key_prefix + "view_dirs"]  # [N, 3]

        res = {}

        if "uv" in self.active_shaders:
            # get ray intersection uv by interpolating triangle uvs
            # using barycentric coords
            vertices_uvs = self.tensor_meshes[self.mesh_id].get_faces_uvs()[faces_id]
            uvs = torch.sum(barycentric.unsqueeze(-1) * vertices_uvs, dim=1)
            # concatenate b channel (zeros)
            uvs = torch.cat((uvs, torch.zeros_like(uvs[:, :1], device="cuda")), dim=-1)
            uvs[~is_hit] = self.bg_color
            res[key_prefix + "uv"] = uvs

        if "texture" in self.active_shaders:
            # get ray intersection uv by interpolating triangle uvs
            # using barycentric coords
            vertices_uvs = self.tensor_meshes[self.mesh_id].get_faces_uvs()[faces_id]
            uvs = torch.sum(barycentric.unsqueeze(-1) * vertices_uvs, dim=1)
            textured = sample_texture(self.tensor_meshes[self.mesh_id].texture, uvs)[
                :, :3
            ]
            if textured.dim() == 3:
                # visualize first 3 components of feature texture
                textured = textured[:, :, 0]
            textured[~is_hit] = self.bg_color
            res[key_prefix + "texture"] = textured

        if "sh_eval" in self.active_shaders:
            # get ray intersection uv by interpolating triangle uvs
            # using barycentric coords
            vertices_uvs = self.tensor_meshes[self.mesh_id].get_faces_uvs()[faces_id]
            uvs = torch.sum(barycentric.unsqueeze(-1) * vertices_uvs, dim=1)
            sh_coeffs = sample_texture(self.tensor_meshes[self.mesh_id].texture, uvs)
            sh_raw_colors = eval_sh(sh_coeffs, view_dirs, 3)
            sh_colors = torch.nn.Sigmoid()(sh_raw_colors)
            # print(f"sh_colors: {sh_colors.shape}")
            # transparency = sh_colors[:, -1].unsqueeze(-1)
            # print(f"transparency: {transparency.shape}")
            # bg_transmittance = 1.0 - transparency
            sh_eval = sh_colors[:, :3]
            # sh_eval[~is_hit] = self.bg_color
            # print(f"color: {color.shape}")
            # final_color = (sh_colors[:, :3] * transparency) + (self.bg_color * bg_transmittance)
            # print(f"final_color: {final_color.shape}")
            # final_color[~is_hit] = self.bg_color
            res[key_prefix + "sh_eval"] = sh_eval

        if "sh_alpha" in self.active_shaders:
            # get ray intersection uv by interpolating triangle uvs
            # using barycentric coords
            vertices_uvs = self.tensor_meshes[self.mesh_id].get_faces_uvs()[faces_id]
            uvs = torch.sum(barycentric.unsqueeze(-1) * vertices_uvs, dim=1)
            sh_coeffs = sample_texture(self.tensor_meshes[self.mesh_id].texture, uvs)
            sh_raw_colors = eval_sh(sh_coeffs, view_dirs, 3)
            sh_colors = torch.nn.Sigmoid()(sh_raw_colors)
            transparency = sh_colors[:, -1].unsqueeze(-1)
            res[key_prefix + "sh_alpha"] = transparency

        if "hits" in self.active_shaders:
            # convert hit flag to float
            res[key_prefix + "hits"] = is_hit.float()

        if "triangle_mesh_id" in self.active_shaders:
            # assign a color to each face
            faces_id[~is_hit] = 0  # prevent overflow
            faces_color = self.tensor_meshes[self.mesh_id].mesh_color.repeat(
                is_hit.shape[0], 1
            )
            faces_color[~is_hit] = self.bg_color
            res[key_prefix + "triangle_mesh_id"] = faces_color

        if "triangle_id" in self.active_shaders:
            # assign a color to each face
            faces_id[~is_hit] = 0  # prevent overflow
            faces_color = self.tensor_meshes[self.mesh_id].faces_color[
                faces_id.squeeze(-1)
            ]
            faces_color[~is_hit] = self.bg_color
            res[key_prefix + "triangle_id"] = faces_color

        if "positions" in self.active_shaders:
            # normalize to [0, 1]
            if is_hit.sum() > 0:
                mn = positions[is_hit].min()
                mx = positions[is_hit].max()
                positions = (positions - mn) / (mx - mn + 1e-5)
            positions[~is_hit] = self.bg_color
            res[key_prefix + "positions"] = positions

        if "normals" in self.active_shaders:
            # already normalized to [-1, 1]
            normals = (normals + 1) * 0.5
            normals[~is_hit] = self.bg_color
            res[key_prefix + "normals"] = normals

        if "depth" in self.active_shaders:
            # mn = self.t_near
            # mx = self.t_far
            # depth = (depth - mn) / (mx - mn + 1e-5)
            # depth = depth / self.camera.t_far
            depth = torch.clip(depth, self.t_near, self.t_far)
            depth = depth / self.t_far
            depth = 1 - depth  # inverse
            depth[~is_hit] = 0
            res[key_prefix + "depth"] = depth

        if "view_dirs" in self.active_shaders:
            # already normalized to [-1, 1]
            view_dirs = (view_dirs + 1) * 0.5
            view_dirs[~is_hit] = self.bg_color
            res[key_prefix + "view_dirs"] = view_dirs

        return res

    @torch.no_grad()
    def render_rays(
        self,
        rays_o,
        rays_d,
        verbose=False,
        debug_ray_idx=None,
        **kwargs,
    ) -> dict:
        """Render rays

        Args:
            rays_o (torch.tensor): ray origins
            rays_d (torch.tensor): ray directions
            iter_nr (int): training iteration number
        """

        # trace
        outputs = self.RT.trace(
            rays_o,
            rays_d,
            verbose=verbose,
            mesh_id=self.mesh_id,
            debug_ray_idx=debug_ray_idx,
            check_second_intersection=self.renders_options["render_second_hit"],
        )

        # shade
        render_buffers = self.shade(outputs)

        res = {"ray-traced": render_buffers}

        return {"renders": res}
