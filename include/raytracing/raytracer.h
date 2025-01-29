#pragma once

#include <Eigen/Dense>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;
using TrigsMeshIdx = Matrix<uint32_t, Dynamic, 1, RowMajor>;

namespace raytracing
{

    // abstract class of raytracer
    class RayTracer
    {
    public:
        RayTracer() {}
        virtual ~RayTracer() {}

        virtual void trace(
            at::Tensor rays_o,
            at::Tensor rays_d,
            at::Tensor min_depth,
            at::Tensor positions,
            at::Tensor normals,
            at::Tensor depth,
            at::Tensor triangles_mesh_id,
            at::Tensor triangles_id,
            at::Tensor barycentric
        ) = 0;
    };

    // function to create an implementation of raytracer
    RayTracer *create_raytracer(Ref<const Verts> vertices, Ref<const Trigs> triangles);
    // RayTracer *create_raytracer(Ref<const Verts> vertices, Ref<const Trigs> triangles, Ref<const TrigsMeshIdx> triangles_mesh_id);

} // namespace raytracing