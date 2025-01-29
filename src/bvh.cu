#include <raytracing/common.h>
#include <raytracing/triangle.cuh>
#include <raytracing/bvh.cuh>

#include <stack>
#include <iostream>
#include <cstdio>

using namespace Eigen;
using namespace raytracing;

namespace raytracing
{
    __global__ void raytrace_kernel(
        uint32_t n_elements,
        const Vector3f *__restrict__ rays_o,
        const Vector3f *__restrict__ rays_d,
        const float *__restrict__ min_depth, 
        Vector3f *__restrict__ positions,
        Vector3f *__restrict__ normals, 
        float *__restrict__ depth,
        int64_t *__restrict__ triangles_mesh_id, 
        int64_t *__restrict__ triangles_id, 
        Vector3f *__restrict__ barycentric,
        const TriangleBvhNode *__restrict__ nodes, 
        const Triangle *__restrict__ triangles
    );

    struct DistAndIdx
    {
        float dist;
        uint32_t idx;

        __host__ __device__ bool operator<(const DistAndIdx &other)
        {
            return dist < other.dist;
        }

        // __host__ __device__ bool operator>(const DistAndIdx &other)
        // {
        //     return dist > other.dist;
        // }
    };

    template <typename T>
    __host__ __device__ void inline compare_and_swap(T &t1, T &t2)
    {
        if (t1 < t2)
        {
            T tmp{t1};
            t1 = t2;
            t2 = tmp;
        }
    }

    // template <typename T>
    // __host__ __device__ void inline compare_and_swap_reversed(T &t1, T &t2)
    // {
    //     if (t1 > t2)
    //     {
    //         T tmp{t1};
    //         t1 = t2;
    //         t2 = tmp;
    //     }
    // }

    // Sorting networks from http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N4L5D3
    template <uint32_t N, typename T>
    __host__ __device__ void sorting_network(T values[N])
    {
        static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
        if (N <= 1)
        {
            return;
        }
        else if (N == 2)
        {
            compare_and_swap(values[0], values[1]);
        }
        else if (N == 3)
        {
            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[1], values[2]);
        }
        else if (N == 4)
        {
            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[1], values[3]);
            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[1], values[2]);
        }
        else if (N == 5)
        {
            compare_and_swap(values[0], values[3]);
            compare_and_swap(values[1], values[4]);

            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[1], values[3]);

            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[4]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);

            compare_and_swap(values[2], values[3]);
        }
        else if (N == 6)
        {
            compare_and_swap(values[0], values[5]);
            compare_and_swap(values[1], values[3]);
            compare_and_swap(values[2], values[4]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);

            compare_and_swap(values[0], values[3]);
            compare_and_swap(values[2], values[5]);

            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[4], values[5]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);
        }
        else if (N == 7)
        {
            compare_and_swap(values[0], values[6]);
            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[4], values[5]);

            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[1], values[4]);
            compare_and_swap(values[3], values[6]);

            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[5]);
            compare_and_swap(values[3], values[4]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[4], values[6]);

            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[4], values[5]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);
            compare_and_swap(values[5], values[6]);
        }
        else if (N == 8)
        {
            compare_and_swap(values[0], values[2]);
            compare_and_swap(values[1], values[3]);
            compare_and_swap(values[4], values[6]);
            compare_and_swap(values[5], values[7]);

            compare_and_swap(values[0], values[4]);
            compare_and_swap(values[1], values[5]);
            compare_and_swap(values[2], values[6]);
            compare_and_swap(values[3], values[7]);

            compare_and_swap(values[0], values[1]);
            compare_and_swap(values[2], values[3]);
            compare_and_swap(values[4], values[5]);
            compare_and_swap(values[6], values[7]);

            compare_and_swap(values[2], values[4]);
            compare_and_swap(values[3], values[5]);

            compare_and_swap(values[1], values[4]);
            compare_and_swap(values[3], values[6]);

            compare_and_swap(values[1], values[2]);
            compare_and_swap(values[3], values[4]);
            compare_and_swap(values[5], values[6]);
        }
    }

    template <uint32_t BRANCHING_FACTOR>
    class TriangleBvhWithBranchingFactor : public TriangleBvh
    {
    public:
        __host__ __device__ static std::tuple<int, int, float, float, float> ray_intersect(Ref<const Vector3f> ray_o, Ref<const Vector3f> ray_d, const float min_t, const TriangleBvhNode *__restrict__ bvhnodes, const Triangle *__restrict__ triangles)
        {
            // returns closest or farthest intersection
            FixedIntStack query_stack;
            query_stack.push(0);

            float curr_t = MAX_DIST;

            int tri_idx = -1;
            int mesh_idx = -1;
            float curr_u = 0.0f;
            float curr_v = 0.0f;

            while (!query_stack.empty())
            {
                int idx = query_stack.pop();
                const TriangleBvhNode &node = bvhnodes[idx];

                if (node.left_idx < 0)
                {
                    // printf("leaf node\n");
                    // printf("left_idx: %d\n", node.left_idx);

                    int end = -node.right_idx - 1;
                    for (int i = -node.left_idx - 1; i < end; ++i)
                    {
                        TriangleHit hit_res;
                        int is_hit = triangles[i].ray_intersect(ray_o, ray_d, hit_res);
                        float t = hit_res.t;
                        float u = hit_res.uv.x();
                        float v = hit_res.uv.y();
                        int m_idx = triangles[i].mesh_idx;

                        // printf("curr_t: %f\n", curr_t);
                        // printf("tri_idx: %d\n", tri_idx);
                        // printf("mesh_idx: %d\n", mesh_idx);

                        if (is_hit && t > min_t && t < curr_t)
                        {
                            // printf("hit: %d, t: %f\n", is_hit, t);
                            curr_t = t;
                            mesh_idx = m_idx;
                            tri_idx = i;
                            curr_u = u;
                            curr_v = v;
                        }
                    }
                }
                else
                {
                    DistAndIdx children[BRANCHING_FACTOR];
                    uint32_t first_child = node.left_idx;

#pragma unroll
                    for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i)
                    {
                        children[i] = {
                            bvhnodes[i + first_child].bb.ray_intersect(ray_o, ray_d).x(), // t_near
                            i + first_child                                         // node idx
                        };
                    }
                    sorting_network<BRANCHING_FACTOR>(children);

#pragma unroll
                    for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i)
                    {
                        // looking for closest intersection
                        if (children[i].dist < curr_t)
                        {
                            // printf("push: %d\n", children[i].idx);
                            query_stack.push(children[i].idx);
                        }
                    }
                }
            }

            return std::make_tuple(mesh_idx, tri_idx, curr_t, curr_u, curr_v);
        }

        void ray_trace_gpu(
            uint32_t n_elements,
            const float *rays_o,
            const float *rays_d,
            const float *min_depth,
            float *positions,
            float *normals,
            float *depth,
            int64_t *triangles_mesh_id,
            int64_t *triangles_id,
            float *barycentric,
            const Triangle *gpu_triangles,
            cudaStream_t stream
        ) override
        {

            // cast float* to Vector3f*
            const Vector3f *rays_o_vec = (const Vector3f *)rays_o;
            const Vector3f *rays_d_vec = (const Vector3f *)rays_d;
            Vector3f *positions_vec = (Vector3f *)positions;
            Vector3f *normals_vec = (Vector3f *)normals;
            Vector3f *barycentric_vec = (Vector3f *)barycentric;

            linear_kernel(
                raytrace_kernel,
                0,
                stream,
                n_elements,
                rays_o_vec,
                rays_d_vec,
                min_depth,
                positions_vec,
                normals_vec,
                depth,
                triangles_mesh_id,
                triangles_id,
                barycentric_vec,
                m_nodes_gpu.data(),
                gpu_triangles);

            // cudaDeviceSynchronize();
            // cudaCheckError();
        }

        void build(std::vector<Triangle> &triangles, uint32_t n_primitives_per_leaf) override
        {
            m_nodes.clear();

            // Root
            m_nodes.emplace_back();
            m_nodes.front().bb = BoundingBox(std::begin(triangles), std::end(triangles));

            struct BuildNode
            {
                int node_idx;
                std::vector<Triangle>::iterator begin;
                std::vector<Triangle>::iterator end;
            };

            std::stack<BuildNode> build_stack;
            build_stack.push({0, std::begin(triangles), std::end(triangles)});

            while (!build_stack.empty())
            {
                const BuildNode &curr = build_stack.top();
                size_t node_idx = curr.node_idx;

                std::array<BuildNode, BRANCHING_FACTOR> children;
                children[0].begin = curr.begin;
                children[0].end = curr.end;

                build_stack.pop();

                // Partition the triangles into the children
                int n_children = 1;
                while (n_children < BRANCHING_FACTOR)
                {
                    for (int i = n_children - 1; i >= 0; --i)
                    {
                        auto &child = children[i];

                        // Choose axis with maximum standard deviation
                        Vector3f mean = Vector3f::Zero();
                        for (auto it = child.begin; it != child.end; ++it)
                        {
                            mean += it->centroid();
                        }
                        mean /= (float)std::distance(child.begin, child.end);

                        Vector3f var = Vector3f::Zero();
                        for (auto it = child.begin; it != child.end; ++it)
                        {
                            Vector3f diff = it->centroid() - mean;
                            var += diff.cwiseProduct(diff);
                        }
                        var /= (float)std::distance(child.begin, child.end);

                        Vector3f::Index axis;
                        var.maxCoeff(&axis);

                        auto m = child.begin + std::distance(child.begin, child.end) / 2;
                        std::nth_element(
                            child.begin, m, child.end,
                            [&](const Triangle &tri1, const Triangle &tri2)
                            {
                                return tri1.centroid(axis) < tri2.centroid(axis);
                            });

                        children[i * 2].begin = children[i].begin;
                        children[i * 2 + 1].end = children[i].end;
                        children[i * 2].end = children[i * 2 + 1].begin = m;
                    }

                    n_children *= 2;
                }

                // Create next build nodes
                m_nodes[node_idx].left_idx = (int)m_nodes.size();
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i)
                {
                    auto &child = children[i];
                    assert(child.begin != child.end);
                    child.node_idx = (int)m_nodes.size();

                    m_nodes.emplace_back();
                    m_nodes.back().bb = BoundingBox(child.begin, child.end);

                    if (std::distance(child.begin, child.end) <= n_primitives_per_leaf)
                    {
                        m_nodes.back().left_idx = -(int)std::distance(std::begin(triangles), child.begin) - 1;
                        m_nodes.back().right_idx = -(int)std::distance(std::begin(triangles), child.end) - 1;
                    }
                    else
                    {
                        build_stack.push(child);
                    }
                }
                m_nodes[node_idx].right_idx = (int)m_nodes.size();
            }

            m_nodes_gpu.resize_and_copy_from_host(m_nodes);

            std::cout << "[INFO] Built TriangleBvh: nodes=" << m_nodes.size() << std::endl;
        }

        TriangleBvhWithBranchingFactor() {}
    };

    using TriangleBvh4 = TriangleBvhWithBranchingFactor<4>;

    std::unique_ptr<TriangleBvh> TriangleBvh::make()
    {
        return std::unique_ptr<TriangleBvh>(new TriangleBvh4());
    }

    __global__ void raytrace_kernel(
        uint32_t n_elements,
        const Vector3f *__restrict__ rays_o,
        const Vector3f *__restrict__ rays_d,
        const float *__restrict__ min_depth,
        Vector3f *__restrict__ positions,
        Vector3f *__restrict__ normals,
        float *__restrict__ depth,
        int64_t *__restrict__ triangles_mesh_id,
        int64_t *__restrict__ triangles_id,
        Vector3f *__restrict__ barycentric,
        const TriangleBvhNode *__restrict__ nodes,
        const Triangle *__restrict__ triangles
    )
    {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_elements)
            return;

        Vector3f ray_o = rays_o[i];
        Vector3f ray_d = rays_d[i];
        float min_d = min_depth[i];

        std::tuple<int, int, float, float, float> res = TriangleBvh4::ray_intersect(ray_o, ray_d, min_d, nodes, triangles);

        // write depth
        depth[i] = std::get<2>(res);

        // intersection point is written back to positions.
        // non-intersect point reaches at most 10 depth
        positions[i] = ray_o + depth[i] * ray_d;

        // face normal is written to directions.
        if (std::get<0>(res) >= 0)
        {
            triangles_mesh_id[i] = triangles[std::get<0>(res)].mesh_idx;
            triangles_id[i] = triangles[std::get<1>(res)].idx;
            normals[i] = triangles[std::get<1>(res)].normal();
            float u = std::get<3>(res);
            float v = std::get<4>(res);
            barycentric[i] = Vector3f(1 - (u + v), u, v);
        }
        else
        {
            normals[i].setZero();
            triangles_mesh_id[i] = -1;
            triangles_id[i] = -1;
            barycentric[i].setZero();
        }
    }
}