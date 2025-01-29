# CUDA extension
import torch
import numpy as np
import raytracing_gpu as backend


class RayTracer:
    def __init__(self, meshes, t_near=1e-3, t_far=100, pack_meshes=False):

        self.t_near = t_near
        self.t_far = t_far
        self.mesh_bvhs = []
        self.nr_meshes = len(meshes)

        # for each mesh in the scene
        for mesh in meshes:
            assert mesh.faces.shape[0] > 8, "BVH needs at least 8 triangles."

            vertices = mesh.vertices
            faces = mesh.faces

            # create raytracer (with mesh-specific BVH)
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()

            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()

            # triangles_mesh_id = np.ones((faces.shape[0], 1), dtype=np.int32) * i

            mesh_bvh = backend.create_raytracer(vertices, faces)  # , triangles_mesh_id)
            self.mesh_bvhs.append(mesh_bvh)

    @torch.no_grad()
    def trace(
        self,
        rays_o,
        rays_d,
        mesh_id=0,
        verbose=False,
        debug_ray_idx=None,
        check_second_intersection=False,
    ):
        """
        rays_o (torch.Tensor, cuda, float) : [nr_rays, 3]
        rays_d (torch.Tensor, cuda, float) : [nr_rays, 3]
        """

        # assert rays data type
        assert (
            torch.is_tensor(rays_o) and rays_o.is_cuda
        ), "rays_o must be a torch.Tensor on cuda"
        assert (
            torch.is_tensor(rays_d) and rays_d.is_cuda
        ), "rays_d must be a torch.Tensor on cuda"
        assert (
            mesh_id < self.nr_meshes
        ), "mesh_id must be smaller than the number of meshes in the scene"

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        # prefix = rays_o.shape[:-1]
        # nr_rays = rays_o.shape[0]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        # TODO: fix
        # min_depth = torch.ones_like(rays_o[:, 0]).contiguous() * self.t_near
        min_depth = torch.zeros_like(rays_o[:, 0]).contiguous()
        # first hit buffers
        positions = torch.zeros_like(rays_o).contiguous()
        normals = torch.zeros_like(rays_d).contiguous()
        depth = torch.zeros_like(rays_o[:, 0]).contiguous()
        triangles_id = torch.zeros_like(rays_o[:, 0], dtype=torch.int64).contiguous()
        triangles_mesh_id = torch.zeros_like(
            rays_o[:, 0], dtype=torch.int64
        ).contiguous()
        barycentric = torch.zeros_like(rays_o).contiguous()

        # inplace write intersections back to rays_o
        mesh_bvh = self.mesh_bvhs[mesh_id]
        mesh_bvh.trace(
            # inputs
            rays_o,
            rays_d,
            min_depth,
            # outputs
            positions,
            normals,
            depth,
            triangles_mesh_id,
            triangles_id,
            barycentric,
        )

        # wait for cuda to finish
        torch.cuda.synchronize()

        is_hit = (depth <= self.t_far).squeeze(-1)
        any_hit = is_hit.any(dim=-1)

        res = {
            "any_hit": any_hit,  # [N,]
            "is_hit": is_hit,  # [N,]
            "positions": positions,  # [N, 3]
            "triangles_mesh_id": triangles_mesh_id,  # [N,]
            "triangles_id": triangles_id,  # [N,]
            "depth": depth,  # [N,]
            "normals": normals,  # [N, 3]
            "barycentric": barycentric,  # [N, 3]
            "view_dirs": rays_d,  # [N, 3]
        }

        # (optional) check second intersection
        if check_second_intersection:
            # second hit buffers
            sh_is_hit = torch.zeros_like(is_hit).contiguous()
            sh_any_hit = torch.zeros_like(any_hit).contiguous()
            sh_positions = torch.zeros_like(rays_o).contiguous()
            sh_normals = torch.zeros_like(rays_d).contiguous()
            sh_depth = torch.zeros_like(rays_o[:, 0]).contiguous()
            sh_triangles_mesh_id = torch.zeros_like(
                rays_o[:, 0], dtype=torch.int64
            ).contiguous()
            sh_triangles_id = torch.zeros_like(
                rays_o[:, 0], dtype=torch.int64
            ).contiguous()
            sh_barycentric = torch.zeros_like(rays_o).contiguous()

            # only trace rays that hit the mesh
            if any_hit:
                mesh_bvh.trace(
                    # inputs
                    rays_o,
                    rays_d,
                    depth,
                    # outputs
                    sh_positions,
                    sh_normals,
                    sh_depth,
                    sh_triangles_mesh_id,
                    sh_triangles_id,
                    sh_barycentric,
                )

                # wait for cuda to finish
                torch.cuda.synchronize()

                sh_is_hit = (sh_depth <= self.t_far).squeeze(-1)
                sh_is_hit = sh_is_hit & is_hit  # (just to be extra sure)
                sh_depth = torch.clamp(sh_depth, 0, self.t_far)
                sh_any_hit = sh_is_hit.any(dim=-1)

            sh_res = {
                "sh_any_hit": sh_any_hit,  # [N,]
                "sh_is_hit": sh_is_hit,  # [N,]
                "sh_positions": sh_positions,  # [N, 3]
                "sh_triangles_mesh_id": sh_triangles_mesh_id,  # [N,]
                "sh_triangles_id": sh_triangles_id,  # [N,]
                "sh_depth": sh_depth,  # [N,]
                "sh_normals": sh_normals,  # [N, 3]
                "sh_barycentric": sh_barycentric,  # [N, 3]
                "sh_view_dirs": rays_d,  # [N, 3]
            }

            # append second hit results to first hit results
            res.update(sh_res)

        if verbose:
            for k, v in res.items():
                print(
                    k,
                    "\n shape:",
                    v.shape,
                    "\n data:",
                    v,
                    "\n min val, max val:",
                    v.min(),
                    v.max(),
                    "\n dtype:",
                    v.dtype,
                    "\n",
                )

        if debug_ray_idx is not None:
            print(f"debug_ray_idx {debug_ray_idx}")
            res_debug = {}
            for k, v in res.items():
                if k == "any_hit" or k == "sh_any_hit":
                    continue
                res_debug[k] = v[debug_ray_idx]

            for k, v in res_debug.items():
                print(k, v)

        return res

    def trace_all(self, rays_o, rays_d, verbose=False, check_second_intersection=False):
        """trace all meshes in the scene

        args:
            rays_o (torch.Tensor, cuda, float) : [nr_rays, 3]
            rays_d (torch.Tensor, cuda, float) : [nr_rays, 3]

        returns:
            results (list): list of dicts of hit results
        """
        results = []
        for mesh_id, _ in enumerate(self.mesh_bvhs):
            if verbose:
                print(f"tracing mesh {mesh_id}")

            res = self.trace(
                rays_o,
                rays_d,
                mesh_id=mesh_id,
                verbose=verbose,
                check_second_intersection=check_second_intersection,
            )
            results.append(res)

        return results
