# RayTracer

A CUDA Mesh RayTracer with BVH acceleration.

## Install

```bash
conda create -n raytracing python=3.8
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/avg-dev/raytrace-mesh
cd raytrace-mesh
```

raytracelib also depends on [mvdatasets](https://github.com/s-esposito/mv_datasets), which can be easily installed as follows:

```bash
# install mv_datasets
git clone --recursive https://github.com/s-esposito/mv_datasets
cd mv_datasets && python setup.py develop

# deps
pip install ninja trimesh opencv-python numpy tqdm matplotlib dearpygui

# install
python setup.py develop
``` 

### Usage

Example for a mesh normal renderer:

```bash
python viewer.py
```

https://user-images.githubusercontent.com/25863658/183238748-7ac82808-6cd3-4bb6-867a-9c22f8e3f7dd.mp4

Example code:

```python
import torch
from raytracelib import RayTracer
from mvdatasets.utils.mesh import Mesh

mesh_path = "meshes/smurf.obj"
texture_path = "textures/chessboard.png"

# load mesh
mesh = Mesh(
        mesh_meta={
            "mesh_path": mesh_path,
            "textures": [
                {
                    "texture_path": texture_path,
                }
            ]
        })

# build BVH from mesh
RT = RayTracer(mesh.vertices, mesh.faces)

# get rays [N, 3], [N, 3], torch.Tensor (cuda)
rays_o, rays_d = get_rays(pose, intrinsics, height, width)

# query ray-mesh intersection
res = RT.trace(rays_o, rays_d)

ids = res["ids"]  # [N, 1]
depth = res["depth"]  # [N, 1]
positions = res["positions"]  # [N, 3]
normals = res["normals"]  # [N, 3]
is_hit = res["is_hit"]  # [N,]
```

## Acknowledgement

- This repo is based on [nanashi kiui](https://github.com/ashawkey)'s [raytracing](https://github.com/ashawkey/raytracing) one.
- Credits to [Thomas Müller](https://tom94.net/)'s amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp).
