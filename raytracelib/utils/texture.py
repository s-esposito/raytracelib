import numpy as np
import torch
from rich import print
from PIL import Image

from raytracelib.utils.images import image2numpy


def sample_texture(image, uvs):
    """
    Args:
        image: torch.tensor, [H, W, C] or [H, W, C, F]
        uvs: torch.tensor, [N, 2] in [0, 1]
    Out:
        vals: torch.tensor, [N, C] or [N, C, F]
    """
    # get image dims
    height, width = image.shape[:2]
    assert height == width, "only square textures are supported"
    texture_res = height
    # convert to uv coordinates
    uvs = uvs * torch.tensor([texture_res, texture_res], device=uvs.device)
    # convert to pixel coordinates
    uvs[:, 0] = torch.clamp(uvs[:, 0], 0, texture_res - 1)
    uvs[:, 1] = torch.clamp(uvs[:, 1], 0, texture_res - 1)
    uvs = uvs.int()
    # sample
    vals = image[uvs[:, 0], uvs[:, 1]]
    return vals


class RGBATexture:
    def __init__(self, texture_path):

        assert texture_path.endswith(
            ".png"
        ), f"texture {texture_path} must be a .png file"
        print(f"[INFO] loading texture {texture_path.split('/')[-1]}")
        texture = image2numpy(Image.open(texture_path)).astype(np.float32)
        print(f"[INFO] texture {texture_path.split('/')[-1]} loaded: {texture.shape}")
        # # if rgb, make it rgba
        # if C == 3:
        #     self.image = np.concatenate(
        #         [self.image, np.ones_like(self.image[..., :1])], axis=-1
        #     )
        self.image = texture
        self.height, self.width, self.nr_channels = self.image.shape


class SHTexture:
    def __init__(self, textures_paths=[], sh_deg=3, min_val=None, max_val=None):

        all_sh_coeffs = []
        channel_sh_coeffs = []
        sh_deg = 3
        nr_sh_coeffs = (sh_deg + 1) ** 2
        nr_read_sh_coeffs = 0

        for texture_path in textures_paths:
            assert texture_path.endswith(
                ".png"
            ), f"texture {texture_path} must be a .png file"
            print(f"[INFO] loading texture {texture_path.split('/')[-1]}")
            sh_coeffs = image2numpy(Image.open(texture_path)).astype(np.float32)
            sh_coeffs = np.rot90(sh_coeffs, k=3)  # fix alignement problem
            # sh_coeffs = np.random.rand(512, 512, 4).astype(np.float32)
            scale = max_val - min_val
            sh_coeffs = sh_coeffs * scale + min_val
            channel_sh_coeffs.append(sh_coeffs)
            nr_read_sh_coeffs += 4
            print(
                f"[INFO] texture {texture_path.split('/')[-1]} loaded: {sh_coeffs.shape}"
            )

            if nr_read_sh_coeffs == nr_sh_coeffs:
                image = np.concatenate(channel_sh_coeffs, axis=-1)
                all_sh_coeffs.append(image)
                channel_sh_coeffs = []
                nr_read_sh_coeffs = 0

        texture = np.stack(all_sh_coeffs, axis=2)
        print(f"[INFO] sh coeffs params {texture.shape} loaded")
        self.image = texture
        self.height, self.width, self.nr_channels, _ = self.image.shape
