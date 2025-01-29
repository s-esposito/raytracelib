import numpy as np
from PIL import Image

from mvdatasets.utils.mesh import Mesh
from mvdatasets.utils.tensor_mesh import TensorMesh
from raytracelib.renderers.mesh_renderer import MeshRenderer

from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.profiler import Profiler

meshes_paths = ["meshes/smurf.obj", "meshes/plushy.obj"]
textures_paths = ["textures/chessboard.png"]

meshes = []
for mesh_path in meshes_paths:
    mesh = Mesh(
        mesh_meta={
            "mesh_path": mesh_path,
            "textures": [
                {
                    "texture_path": textures_paths[0],
                }
            ],
        }
    )
    meshes.append(mesh)

meshes_tensor = []
for mesh in meshes:
    mesh_tensor = TensorMesh(mesh, device="cuda")
    meshes_tensor.append(mesh_tensor)

profiler = Profiler()
t_near = 0.5  # near plane
t_far = 3  # far plane
width = 512
height = 512
renderer = MeshRenderer(
    meshes_tensor, t_near, t_far, render_all_shaders=True, profiler=profiler
)

fov_deg = 30
fov_rad = fov_deg * np.pi / 180
instrinsics = np.eye(3)
instrinsics[0, 0] = width / (2 * np.tan(fov_rad / 2))
instrinsics[1, 1] = height / (2 * np.tan(fov_rad / 2))
instrinsics[0, 2] = width / 2 + 0.5
instrinsics[1, 2] = height / 2 + 0.5
# +0.5 because rays are casted from pixel centers
pose = np.eye(4)
pose[:3, 3] = np.array([1.5, 0, 0])
local_transform = np.array([[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
camera = Camera(
    intrinsics=instrinsics,
    pose=pose,
    local_transform=local_transform,
    height=height,
    width=width,
)
print(camera)

# first hit

for i in range(len(meshes_tensor)):
    renderer.mesh_id = i

    renders = renderer.render(camera, verbose=False, debug=True)

    for render_key, render in renders["ray-traced"].items():

        print(render_key, render.shape)
        render = render.reshape(height, width, -1)
        # save render_buffer as image
        if render.shape[-1] == 1:
            # repeat third channel 3 times
            render = np.repeat(render, 3, axis=-1)
        img = Image.fromarray((render * 255).astype(np.uint8))
        img.save(f"plots/{render_key}_{i}.png")

# # second hit

# renderer.renders_options["render_second_hit"] = True
# renders = renderer.render(camera, verbose=False, debug=True)

# for render_key, render in renders["ray-traced"].items():

#     print(render_key, render.shape)
#     render = render.reshape(height, width, -1)
#     # save render_buffer as image
#     if render.shape[-1] == 1:
#         # repeat third channel 3 times
#         render = np.repeat(render, 3, axis=-1)
#     img = Image.fromarray((render * 255).astype(np.uint8))
#     img.save(f"plots/{render_key}.png")

profiler.print_avg_times()
