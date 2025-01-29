from abc import ABC, abstractmethod
import torch
import numpy as np
import math
from tqdm import tqdm

from mvdatasets.utils.raycasting import get_camera_rays


class BaseRenderer(ABC):
    def __init__(self, renders_modes=[], renders_shaders=[], profiler=None):
        self.profiler = profiler
        self.renders_options = {}

        # shading
        self.renders_modes = renders_modes
        self.renders_shaders = renders_shaders
        if len(self.renders_modes) > 0:
            self.active_renders_modes = [self.renders_modes[0]]
        else:
            self.active_renders_modes = []
        if len(self.renders_shaders) > 0:
            self.active_shaders = [self.renders_shaders[0]]
        else:
            self.active_shaders = []

    @abstractmethod
    def shade(self, outputs):
        """convert outputs to frame buffer values"""
        pass

    @abstractmethod
    def render_rays(
        self,
        rays_o,
        rays_d,
        iter_nr=999999,
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
        pass

    @torch.no_grad()
    def render(
        self,
        camera,
        chunk_size=None,  # None or positive int (nr of rays per chunk)
        iter_nr=999999,
        verbose=False,
        debug_pixel=None,  # (y, x)
        **kwargs,
    ) -> dict:
        """base renderer

        Args:
            camera (_type_): _description_
            chunk_size (_type_): _description_
            iter_nr (int, optional): _description_. Defaults to 999999.

        Returns:
            pred (dict): dictionary of renders for each rendering mode.
            Renders are np.ndarray images.
        """

        # # chunk_size
        # print("chunk_size", chunk_size)

        # gen rays

        if self.profiler is not None:
            self.profiler.start("ray_gen")

        jitter_pixels = False
        rays_o_all, rays_d_all, points_2d = get_camera_rays(
            camera, jitter_pixels=jitter_pixels, device="cuda"
        )
        if verbose:
            print("rays_o_all", rays_o_all.shape)
            print("rays_d_all", rays_d_all.shape)
        rays_o_all = rays_o_all.contiguous()
        rays_d_all = rays_d_all.contiguous()

        if self.profiler is not None:
            self.profiler.end("ray_gen")

        debug_pixel_idx = None
        debug_pixel_batch_idx = 0
        if chunk_size is not None:
            # split rays in chunks
            nr_chunks = math.ceil(rays_o_all.shape[0] / chunk_size)
            rays_o_list = torch.chunk(rays_o_all, nr_chunks)
            rays_d_list = torch.chunk(rays_d_all, nr_chunks)
            if debug_pixel is not None:
                y = debug_pixel[0]
                x = debug_pixel[1]
                debug_pixel_batch_idx = (y * self.width + x) // chunk_size
                debug_pixel_idx = debug_pixel_batch_idx % chunk_size
        else:
            rays_o_list = [rays_o_all]
            rays_d_list = [rays_d_all]
            if debug_pixel is not None:
                y = debug_pixel[0]
                x = debug_pixel[1]
                debug_pixel_idx = y * self.width + x

        if debug_pixel is not None:
            print("debug_pixel_idx", debug_pixel_idx)
            print("debug_pixel_batch_idx", debug_pixel_batch_idx)

        renders_batches_lists = {}

        if self.profiler is not None:
            self.profiler.start("render")

        pbar = tqdm(rays_o_list, desc="rendering rays batches", ncols=100, leave=False)
        for i, _ in enumerate(pbar):
            # render batch

            batch_res = self.render_rays(
                rays_o=rays_o_list[i],
                rays_d=rays_d_list[i],
                iter_nr=iter_nr,
                debug_ray_idx=debug_pixel_idx if i == debug_pixel_batch_idx else None,
                verbose=verbose,
            )

            # append batch rendered rays to a list of batches results

            # iterate over render modes
            renders_batch = batch_res["renders"]
            for render_mode, renders_dict in renders_batch.items():
                if renders_dict is not None:
                    # check if mode is already in dict
                    if render_mode not in renders_batches_lists.keys():
                        # init
                        renders_batches_lists[render_mode] = {}
                    # iterate over render keys in batch
                    for render_key, render in renders_dict.items():
                        if render is not None:
                            if (
                                render_key
                                not in renders_batches_lists[render_mode].keys()
                            ):
                                # init
                                renders_batches_lists[render_mode][render_key] = []
                            renders_batches_lists[render_mode][render_key].append(
                                render.detach().cpu().numpy()
                            )
                            # print(
                            #     f"render_mode: {render_mode}, render_key: {render_key}, shape: {render.shape}"
                            # )

        # concat lists to np.ndarrays
        renders = {}
        for render_mode, renders_dict in renders_batches_lists.items():
            renders[render_mode] = {}
            for render_key, render_batches_list in renders_dict.items():
                renders[render_mode][render_key] = np.concatenate(
                    render_batches_list, axis=0
                )

                # print(
                #     f"render_mode: {render_mode}, render_key: {render_key}, shape: {renders[render_mode][render_key].shape}")

        if self.profiler is not None:
            self.profiler.end("render")

        return renders
