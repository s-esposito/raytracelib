import torch


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, height, width):  # N=-1, error_map=None):
    """get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        height, width, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(
        torch.linspace(0, width - 1, width, device=device),
        torch.linspace(0, height - 1, height, device=device),
        indexing="ij",
    )
    i = i.t().reshape([1, height * width]).expand([B, height * width]) + 0.5
    j = j.t().reshape([1, height * width]).expand([B, height * width]) + 0.5

    # results = {}

    # if N > 0:
    #     N = min(N, height * width)

    #     if error_map is None:
    #         inds = torch.randint(0, height * width, size=[N], device=device)  # may duplicate
    #         inds = inds.expand([B, N])
    #     else:
    #         # weighted sample on a low-reso grid
    #         inds_coarse = torch.multinomial(
    #             error_map.to(device), N, replacement=False
    #         )  # [B, N], but in [0, 128*128)

    #         # map to the original resolution with random perturb.
    #         inds_x, inds_y = (
    #             inds_coarse // 128,
    #             inds_coarse % 128,
    #         )  # `//` will throw a warning in torch 1.10... anyway.
    #         sx, sy = height / 128, width / 128
    #         inds_x = (
    #             (inds_x * sx + torch.rand(B, N, device=device) * sx)
    #             .long()
    #             .clamp(max=height - 1)
    #         )
    #         inds_y = (
    #             (inds_y * sy + torch.rand(B, N, device=device) * sy)
    #             .long()
    #             .clamp(max=width - 1)
    #         )
    #         inds = inds_x * width + inds_y

    #         results["inds_coarse"] = inds_coarse  # need this when updating error_map

    #     i = torch.gather(i, -1, inds)
    #     j = torch.gather(j, -1, inds)

    #     results["inds"] = inds

    # else:
    #     inds = torch.arange(height * width, device=device).expand([B, height * width])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    # results["rays_o"] = rays_o
    # results["rays_d"] = rays_d

    return rays_o, rays_d
