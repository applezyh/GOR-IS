#include "auxiliary.h"
#include "sample_backward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// refer to gsplat https://github.com/nerfstudio-project/gsplat/blob/65042cc501d1cdbefaf1d6f61a9a47575eec8c71/gsplat/cuda/include/Utils.cuh#L94
template <uint32_t DIM, class WarpT>
__forceinline__ __device__ void warpSum(float* val, WarpT& warp) {
#pragma unroll
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<float>());
    }
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float& val, WarpT& warp) {
    val = cg::reduce(warp, val, cg::plus<float>());
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float2& val, WarpT& warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float3& val, WarpT& warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
}

template <class WarpT>
__forceinline__ __device__ void warpSum(float4& val, WarpT& warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
    val.w = cg::reduce(warp, val.w, cg::plus<float>());
}

__global__ void preprocessPointsCUDA(
    int P,
    const float3* points3D,
    const float* viewmatrix,
    const float* proj,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx,
    const float tan_fovy,
    const uint32_t* tiles_touched,
    const float2* dL_dpoints2D,
    float3* dL_dpoints3D) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || tiles_touched[idx] == 0)
        return;

    float3 m = points3D[idx];

    // Taking care of gradients from the screenspace points
    float4 m_hom = transformPoint4x4(m, proj);
    float m_w    = 1.0f / (m_hom.w + 0.0000001f);

    // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
    // from rendering procedure
    float3 dL_dpoints;
    float mul1   = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
    float mul2   = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
    dL_dpoints.x = (proj[0] * m_w - proj[3] * mul1) * dL_dpoints2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dpoints2D[idx].y;
    dL_dpoints.y = (proj[4] * m_w - proj[7] * mul1) * dL_dpoints2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dpoints2D[idx].y;
    dL_dpoints.z = (proj[8] * m_w - proj[11] * mul1) * dL_dpoints2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dpoints2D[idx].y;

    // That's the third part of the mean gradient.
    dL_dpoints3D[idx] = dL_dpoints;
}

template <int SAMPLES_PRE_ROUND>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    sampleDepthCUDA(
        const uint2* __restrict__ gaussian_ranges,
        const uint2* __restrict__ point_ranges,
        const uint32_t* __restrict__ gaussian_list,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        float focal_x, float focal_y,
        const float2* __restrict__ points2D,
        const float2* __restrict__ gaussians2D,
        const float4* __restrict__ ray_planes,
        const float4* __restrict__ conic_opacity,
        const uint32_t* __restrict__ n_contrib,
        const float* __restrict__ accum_depth,
        const float* __restrict__ final_T,
        const float3* __restrict__ dL_doutputs,
        float3* __restrict__ dL_dgaussians2D,
        float4* __restrict__ dL_dconic2D,
        float4* __restrict__ dL_dray_planes,
        float2* __restrict__ dL_dpoints2D) {
    // We rasterize again. Compute necessary block info.
    auto block                       = cg::this_thread_block();
    cg::thread_block_tile<32> warp   = cg::tiled_partition<32>(block);
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 range                = gaussian_ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

    // Gradient of pixel coordinate w.r.t. normalized
    // screen-space viewport corrdinates (-1 to 1)
    const float ddelx_dx = 0.5f * W;
    const float ddely_dy = 0.5f * H;

    constexpr int BLOCK_SAMPLES_PRE_ROUND = BLOCK_SIZE * SAMPLES_PRE_ROUND;
    uint2 p_range                         = point_ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int p_rounds                    = ((p_range.y - p_range.x + BLOCK_SAMPLES_PRE_ROUND - 1) / BLOCK_SAMPLES_PRE_ROUND);
    int p_toDo                            = p_range.y - p_range.x;

    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float3 collected_ray_planes[BLOCK_SIZE];
    for (int p_round = 0; p_round < p_rounds; p_round++, p_toDo -= BLOCK_SAMPLES_PRE_ROUND) {
        float T[SAMPLES_PRE_ROUND];
        float T_final[SAMPLES_PRE_ROUND];
        uint32_t point_idx[SAMPLES_PRE_ROUND];
        float2 point_xy[SAMPLES_PRE_ROUND];
        float dL_dDepth[SAMPLES_PRE_ROUND];
        float dL_dfinalT_times_finalT[SAMPLES_PRE_ROUND];
        float2 dL_dpoint_xy[SAMPLES_PRE_ROUND];
        uint32_t last_contributor[SAMPLES_PRE_ROUND] = {0};
        float last_alpha[SAMPLES_PRE_ROUND]          = {0};
        float last_t[SAMPLES_PRE_ROUND]              = {0};
        float accum_t_rec[SAMPLES_PRE_ROUND]         = {0};

        int toDo             = range.y - range.x;
        uint32_t contributor = toDo;
        int point_num_round  = 0;
#pragma unroll
        for (int p = 0; p < SAMPLES_PRE_ROUND; p++) {
            int progress = (p_round * SAMPLES_PRE_ROUND + p) * BLOCK_SIZE + block.thread_rank();
            if (p_range.x + progress < p_range.y) {
                int pid      = point_list[p_range.x + progress];
                point_idx[p] = pid;

                // We start from the back. The ID of the last contributing
                // Gaussian is known from each pixel from the forward.
                last_contributor[p] = n_contrib[pid];
                float3 dL_doutput   = dL_doutputs[pid];
                float final_depth   = accum_depth[pid];
                point_xy[p]         = points2D[pid];
                T_final[p]          = final_T[pid];
                T[p]                = T_final[p];

                float2 pixnf      = {(point_xy[p].x - static_cast<float>(W - 1) / 2.f) / focal_x,
                                     (point_xy[p].y - static_cast<float>(H - 1) / 2.f) / focal_y};
                const float rln   = rnorm3df(pixnf.x, pixnf.y, 1.f);
                const float rln2  = 1.f / (pixnf.x * pixnf.x + pixnf.y * pixnf.y + 1.f);
                const float depth = final_depth * rln / fmaxf(1.f - T_final[p], 1e-7f);
                // output[point_idx[p]]        = {pixnf.x * depth, pixnf.y * depth, depth};

                float dL_ddepth            = dL_doutput.x * pixnf.x + dL_doutput.y * pixnf.y + dL_doutput.z;
                dL_dfinalT_times_finalT[p] = final_depth * rln / fmaxf((1.f - T_final[p]) * (1.f - T_final[p]), 1e-7f) * dL_ddepth * T_final[p];
                dL_dDepth[p]               = rln / fmaxf(1.f - T_final[p], 1e-7f) * dL_ddepth;
                float aux                  = (dL_doutput.x * pixnf.x + dL_doutput.y * pixnf.y + dL_doutput.z) * rln2;
                float2 dL_dpixnf           = {(dL_doutput.x - aux * pixnf.x) * depth,
                                              (dL_doutput.y - aux * pixnf.y) * depth};
                dL_dpoint_xy[p]            = {dL_dpixnf.x / focal_x, dL_dpixnf.y / focal_y};
                point_num_round++;
            }
        }

        // Traverse all Gaussians
        for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
            // Load auxiliary data into shared memory, start in the BACK
            // and load them in revers order.
            block.sync();
            const int progress = i * BLOCK_SIZE + block.thread_rank();
            if (range.x + progress < range.y) {
                const int coll_id                            = gaussian_list[range.y - progress - 1];
                collected_id[block.thread_rank()]            = coll_id;
                collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
                collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
                float4 ray_plane = ray_planes[coll_id];
                collected_ray_planes[block.thread_rank()]    = {ray_plane.x, ray_plane.y, ray_plane.z};
            }
            block.sync();

            // Iterate over Gaussians
            for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
                // refer to gsplat that uses warp-wise reduction before atomicAdd
                // Keep track of current Gaussian ID. Skip, if this one
                // is behind the last contributor for this pixel.

                contributor--;

                // Compute blending values, as before.
                const float2 xy    = collected_xy[j];
                const float4 con_o = collected_conic_opacity[j];
                float G[SAMPLES_PRE_ROUND];
                bool valid[SAMPLES_PRE_ROUND] = {false}; // initialization for safety
                bool any_valid                = false;
                for (int p = 0; p < point_num_round; p++) {
                    const float2 d = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                    float power    = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                    G[p]           = expf(power);
                    float alpha    = fminf(0.99f, con_o.w * G[p]);
                    valid[p]       = !((contributor >= last_contributor[p]) || (power > 0.0f) || (alpha < 1.0f / 255.0f));
                    any_valid      = any_valid || valid[p];
                }
                if (!warp.any(any_valid))
                    continue;

                float3 ray_plane            = collected_ray_planes[j];
                float dL_dt_local           = 0.f;
                float2 dL_dray_planes_local = {0};
                float2 dL_dmean2D_local     = {0};
                float4 dL_dconic2D_local    = {0};
                for (int p = 0; p < point_num_round; p++) {
                    if (valid[p]) {
                        const float2 d              = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                        float alpha                 = fminf(0.99f, con_o.w * G[p]);
                        T[p]                        = T[p] / (1.f - alpha);
                        const float blending_weight = alpha * T[p];

                        // Propagate gradients to per-Gaussian colors and keep
                        // gradients w.r.t. alpha (blending factor for a Gaussian/pixel
                        // pair).
                        float t        = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                        accum_t_rec[p] = last_alpha[p] * last_t[p] + (1.f - last_alpha[p]) * accum_t_rec[p];
                        last_t[p]      = t;
                        last_alpha[p]  = alpha;
                        float dL_dopa  = (t - accum_t_rec[p]) * dL_dDepth[p];
                        dL_dopa *= T[p];
                        dL_dopa += -dL_dfinalT_times_finalT[p] / (1.f - alpha);
                        const float dL_dt = blending_weight * dL_dDepth[p];
                        dL_dt_local += dL_dt;
                        dL_dray_planes_local.x += dL_dt * d.x;
                        dL_dray_planes_local.y += dL_dt * d.y;

                        // Helpful reusable temporary variables
                        const float dL_dG    = con_o.w * dL_dopa;
                        const float gdx      = G[p] * d.x;
                        const float gdy      = G[p] * d.y;
                        const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
                        const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

                        // Update gradients w.r.t. 2D mean position of the Gaussian
                        float dL_ddelx = dL_dG * dG_ddelx + dL_dt * ray_plane.x;
                        float dL_ddely = dL_dG * dG_ddely + dL_dt * ray_plane.y;

                        dL_dmean2D_local.x += dL_ddelx;
                        dL_dmean2D_local.y += dL_ddely;

                        dL_dpoint_xy[p].x -= dL_ddelx;
                        dL_dpoint_xy[p].y -= dL_ddely;

                        dL_dconic2D_local.x += -0.5f * gdx * d.x * dL_dG;
                        dL_dconic2D_local.y += -0.5f * gdx * d.y * dL_dG;
                        dL_dconic2D_local.z += -0.5f * gdy * d.y * dL_dG;
                        dL_dconic2D_local.w += G[p] * dL_dopa;
                    }
                }
                dL_dmean2D_local.x *= ddelx_dx;
                dL_dmean2D_local.y *= ddely_dy;
                warpSum(dL_dt_local, warp);
                warpSum(dL_dray_planes_local, warp);
                warpSum(dL_dmean2D_local, warp);
                warpSum(dL_dconic2D_local, warp);
                if (warp.thread_rank() == 0) {
                    const int global_id = collected_id[j];
                    atomicAdd(&dL_dray_planes[global_id].x, dL_dray_planes_local.x);
                    atomicAdd(&dL_dray_planes[global_id].y, dL_dray_planes_local.y);
                    atomicAdd(&dL_dray_planes[global_id].z, dL_dt_local);
                    atomicAdd(&dL_dgaussians2D[global_id].x, dL_dmean2D_local.x);
                    atomicAdd(&dL_dgaussians2D[global_id].y, dL_dmean2D_local.y);
                    atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_local.x);
                    atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_local.y);
                    atomicAdd(&dL_dconic2D[global_id].z, dL_dconic2D_local.z);
                    atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_local.w);
                }
            }
        }
        for (int p = 0; p < point_num_round; p++) {
            dL_dpoints2D[point_idx[p]] = {dL_dpoint_xy[p].x * ddelx_dx,
                                          dL_dpoint_xy[p].y * ddely_dy};
        }
    }
}

void BACKWARD::preprocess_points(
    int P,
    const float3* points3D,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx,
    const float tan_fovy,
    const uint32_t* tiles_touched,
    const float2* dL_dpoints2D,
    float3* dL_dpoints3D) {
    preprocessPointsCUDA<<<(P + 255) / 256, 256>>>(
        P,
        points3D,
        viewmatrix,
        projmatrix,
        cam_pos,
        W, H,
        tan_fovx,
        tan_fovy,
        tiles_touched,
        dL_dpoints2D,
        dL_dpoints3D);
}

void BACKWARD::sampleDepth(
    const dim3 grid, dim3 block,
    const uint2* gaussian_ranges,
    const uint2* point_ranges,
    const uint32_t* gaussian_list,
    const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    const float2* points2D,
    const float2* gaussians2D,
    const float4* ray_planes,
    const float4* conic_opacity,
    const uint32_t* n_contrib,
    const float* accum_depth,
    const float* final_T,
    const float3* dL_doutput,
    float3* dL_dgaussians2D,
    float4* dL_dconic2D,
    float4* dL_dray_planes,
    float2* dL_dpoints2D) {
    sampleDepthCUDA<SAMPLE_BATCH_SIZE><<<grid, block>>>(
        gaussian_ranges,
        point_ranges,
        gaussian_list,
        point_list,
        W, H,
        focal_x, focal_y,
        points2D,
        gaussians2D,
        ray_planes,
        conic_opacity,
        n_contrib,
        accum_depth,
        final_T,
        dL_doutput,
        dL_dgaussians2D,
        dL_dconic2D,
        dL_dray_planes,
        dL_dpoints2D);
}