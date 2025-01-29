import argparse
import torch
import os
import numpy as np

import mvdatasets as mvds

from raytracelib.gui import GUI
from mvdatasets.utils.mesh import Mesh
from mvdatasets.utils.tensor_mesh import TensorMesh
from raytracelib.renderers.mesh_renderer import MeshRenderer

from mvdatasets.utils.profiler import Profiler

# # Set datasets path
# datasets_path = "/home/stefano/Data"

# # test DTU
# dataset_name = "dtu"
# scene_name = "dtu_scan83"
# config = {"load_mask": False}
# mesh_path = "meshes/smurf.obj"

# # test BlenderNeRF
# dataset_name = "blendernerf"
# scene_name = "plushy"
# config = {"load_mask": False}
# mesh_path = "meshes/plushy.obj"

# nr_meshes = 1
# textures_folder_files = os.listdir(os.path.join("textures", scene_name))
# # get file ending with .csv in the textures folder
# for nr_mesh in range(nr_meshes):
#     textures_paths = [os.path.join("textures", scene_name, file) for file in textures_folder_files if file.endswith(".png") and file.startswith(f"mesh_{nr_mesh}")]
#     csv_file = [os.path.join("textures", scene_name, file) for file in textures_folder_files if file.endswith(".csv") and file.startswith(f"mesh_{nr_mesh}")][0]
# textures_paths = sorted(textures_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
# print(f"csv_files: {csv_file}")
# print(f"textures_paths: {textures_paths}")

# textures_paths = ["textures/chessboard.png"]

# # open csv file
# min_max = []
# with open(csv_file, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         min_max.append(float(line))

meshes_paths = ["meshes/smurf.obj", "meshes/plushy.obj"]

meshes = []
for mesh_path in meshes_paths:
    mesh = Mesh(mesh_meta={"mesh_path": mesh_path, "textures": []})
    meshes.append(mesh)

meshes_tensor = []
for mesh in meshes:
    mesh_tensor = TensorMesh(mesh, device="cuda")
    meshes_tensor.append(mesh_tensor)

# profiler
profiler = Profiler()

# # dataset loading
# print("\nloading data")
# mv_data = mvds.MVDataset(
#     dataset_name,
#     scene_name,
#     datasets_path,
#     splits=["train", "test"],
#     config=config,
#     verbose=True
# )

# renderer
t_near = 0.5  # near plane
t_far = 3  # far plane
renderer = MeshRenderer(meshes_tensor, t_near, t_far, profiler=profiler)

gui = GUI(
    renderer,
    # mv_cameras=mv_data
)
gui.render()
