import argparse
import torch
import os
import numpy as np
from PIL import Image


import mvdatasets as mvds

from raytracelib.gui import GUI
from mvdatasets.utils.mesh import Mesh
from raytracelib.utils.scene import Scene

from raytracelib.utils.images import image2numpy
from raytracelib.utils.texture import RGBATexture, SHTexture, sample_texture

# Set datasets path
datasets_path = "/home/stefano/Data"

# test DTU
dataset_name = "dtu"
scene_name = "dtu_scan83"
config = {"load_mask": False}
mesh_path = "meshes/smurf.obj"

# # test DTU
# dataset_name = "blendernerf"
# scene_name = "plushy"
# config = {"load_mask": False}
# mesh_path = "meshes/plushy.obj"

nr_meshes = 1
textures_folder_files = os.listdir(os.path.join("textures", scene_name))
# get file ending with .csv in the textures folder
for nr_mesh in range(nr_meshes):
    textures_paths = [
        os.path.join("textures", scene_name, file)
        for file in textures_folder_files
        if file.endswith(".png") and file.startswith(f"mesh_{nr_mesh}")
    ]
    csv_file = [
        os.path.join("textures", scene_name, file)
        for file in textures_folder_files
        if file.endswith(".csv") and file.startswith(f"mesh_{nr_mesh}")
    ][0]
textures_paths = sorted(
    textures_paths, key=lambda x: int(x.split("_")[-1].split(".")[0])
)
print(f"csv_files: {csv_file}")
print(f"textures_paths: {textures_paths}")

# open csv file
min_max = []
with open(csv_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        min_max.append(float(line))
print(f"min_max: {min_max}")

if len(textures_paths) > 1:
    texture = SHTexture(
        textures_paths, sh_deg=3, min_val=min_max[0], max_val=min_max[1]
    )
else:
    texture = RGBATexture(textures_paths[0])
print(f"texture: {texture.image.shape}")

# uvs = np.random([2, 3])
# sh_coeffs = sample_texture(self.scene.get_mesh_texture(), uvs)
# sh_colors = eval_sh(sh_coeffs, -view_dirs, 3)
# transparency = sh_colors[-1].item()
# bg_transmittance = 1.0 - transparency
# color = sh_colors[:3] * transparency + self.bg_color * bg_transmittance
