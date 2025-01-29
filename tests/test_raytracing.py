import torch
from raytracelib import RayTracer
from mvdatasets.utils.mesh import Mesh

meshes_paths = ["meshes/smurf.obj", "meshes/plushy.obj"]

meshes = []
for mesh_path in meshes_paths:
    mesh = Mesh(mesh_meta={"mesh_path": mesh_path, "textures": []})
    meshes.append(mesh)

# build BVHs from meshes
RT = RayTracer(meshes, t_far=3)

# get rays [N, 3], [N, 3], torch.Tensor (cuda)
# rays_o, rays_d = get_rays(pose, intrinsics, height, width)

rays_o = torch.tensor([[2, 0, 0]], device="cuda")
rays_d = torch.tensor([[-1, 0, 0]], device="cuda")

# query ray-mesh intersection
check_second_intersection = True
results = RT.trace_all(rays_o, rays_d, verbose=True, check_second_intersection=True)
print(results)
