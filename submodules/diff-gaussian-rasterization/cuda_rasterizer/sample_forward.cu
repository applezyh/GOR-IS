#include "auxiliary.h"
#include "sample_forward.h"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
// adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/5245b20e5d11acd6d1ff5af4b890dc2bedd99693/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
__global__ void preprocessPointsCUDA(
    int P,
    const float* points3D,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    float2* points2D,
    float* depths,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Gaussian will not be processed further.
    tiles_touched[idx] = 0;

    // Perform near culling, quit if outside.
    const float3 p_orig = {points3D[3 * idx], points3D[3 * idx + 1], points3D[3 * idx + 2]};
    float3 p_view;
    if (!in_frustum(p_orig, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    // Transform point by projecting
    float4 p_hom  = transformPoint4x4(p_orig, projmatrix);
    float p_w     = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

    float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};

    // If the point is outside the image, quit.
    if (point_image.x < 0 || point_image.x > W - 1 || point_image.y < 0 || point_image.y > H - 1)
        return;

    // Store some useful helper data for the next steps.
    depths[idx]        = norm3df(p_view.x, p_view.y, p_view.z);
    points2D[idx]      = point_image;
    tiles_touched[idx] = 1;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS, int SAMPLES_PRE_ROUND>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    integrateCUDA(
        const uint2* __restrict__ gaussian_ranges,
        const uint2* __restrict__ point_ranges,
        const uint32_t* __restrict__ gaussian_list,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        float focal_x, float focal_y,
        const float2* __restrict__ points2D,
        const float2* __restrict__ gaussians2D,
        const float* __restrict__ features,
        const float4* __restrict__ ray_planes,
        const float* __restrict__ point_depths,
        const float4* __restrict__ conic_opacity,
        const float* __restrict__ bg_color,
        float* __restrict__ out_color,
        float* __restrict__ out_alpha,
        bool* __restrict__ inside) {
    auto block                 = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

    // Load start/end range of IDs to process in bit sorted list.
    const uint2 range = gaussian_ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    constexpr int BLOCK_SAMPLES_PRE_ROUND = BLOCK_SIZE * SAMPLES_PRE_ROUND;
    uint2 p_range                         = point_ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int p_rounds                    = ((p_range.y - p_range.x + BLOCK_SAMPLES_PRE_ROUND - 1) / BLOCK_SAMPLES_PRE_ROUND);
    int p_toDo                            = p_range.y - p_range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float4 collected_ray_planes[BLOCK_SIZE];
    for (int p_round = 0; p_round < p_rounds; p_round++, p_toDo -= BLOCK_SAMPLES_PRE_ROUND) {
        bool done[SAMPLES_PRE_ROUND]    = {false};
        float weight[SAMPLES_PRE_ROUND] = {0.f};
        float T[SAMPLES_PRE_ROUND];
        uint32_t point_idx[SAMPLES_PRE_ROUND];
        float2 point_xy[SAMPLES_PRE_ROUND];
        float point_depth[SAMPLES_PRE_ROUND];
        int point_done      = 0;
        int point_num_round = 0;
#pragma unroll
        for (int i = 0; i < SAMPLES_PRE_ROUND; i++) {
            int progress = (p_round * SAMPLES_PRE_ROUND + i) * BLOCK_SIZE + block.thread_rank();
            if (p_range.x + progress < p_range.y) {
                T[i]           = 1.f;
                int pid        = point_list[p_range.x + progress];
                point_idx[i]   = pid;
                point_xy[i]    = points2D[pid];
                point_depth[i] = point_depths[pid];
                point_num_round++;
            }
        }
        bool all_done = point_num_round == 0;
        int toDo      = range.y - range.x;
        for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
            // End if entire block votes that it is done rasterizing
            int num_done = __syncthreads_count(all_done);
            if (num_done == BLOCK_SIZE)
                break;
            // Collectively fetch per-Gaussian data from global to shared
            int progress = i * BLOCK_SIZE + block.thread_rank();
            if (range.x + progress < range.y) {
                int coll_id                                  = gaussian_list[range.x + progress];
                collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
                collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
                collected_ray_planes[block.thread_rank()]    = ray_planes[coll_id];
            }
            block.sync();

            // Iterate over current batch
            for (int j = 0; !all_done && j < min(BLOCK_SIZE, toDo); j++) {
                float2 xy    = collected_xy[j];
                float4 con_o = collected_conic_opacity[j];
                for (int p = 0; p < point_num_round; p++) {
                    if (done[p])
                        continue;
                    float2 d    = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                    if (power > 0.0f) {
                        continue;
                    }

                    // Eq. (2) from 3D Gaussian splatting paper.
                    // Obtain alpha by multiplying with Gaussian opacity
                    // and its exponential falloff from mean.
                    // Avoid numerical instabilities (see paper appendix).
                    float4 ray_plane = collected_ray_planes[j];
                    float t          = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                    float alpha      = fminf(0.99f, con_o.w * expf(power));
                    if (point_depth[p] < t) {
                        float delta_u = (t - point_depth[p]) * ray_plane.w;
                        float power   = -0.5f * (delta_u * delta_u);
                        alpha *= ray_plane.w > 0 ? expf(power) : 0.f;
                    }
                    if (alpha < 1.0f / 255.0f)
                        continue;
                    float test_T = T[p] * (1.f - alpha);
                    if (test_T < 0.0001f) {
                        done[p] = true;
                        point_done++;
                        all_done = point_done == point_num_round;
                        continue;
                    }

                    const float aT = alpha * T[p];
                    weight[p] += aT;
                    T[p] = test_T;
                }
            }
        }
        for (int i = 0; i < point_num_round; i++) {
            out_alpha[point_idx[i]] = weight[i];
            inside[point_idx[i]]    = true;
        }
    }
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
        uint32_t* __restrict__ n_contrib,
        float* __restrict__ accum_depth,
        float* __restrict__ final_T,
        float3* __restrict__ output,
        bool* __restrict__ inside_output) {
    auto block                 = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

    // Load start/end range of IDs to process in bit sorted list.
    const uint2 range = gaussian_ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds  = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

    constexpr int BLOCK_SAMPLES_PRE_ROUND = BLOCK_SIZE * SAMPLES_PRE_ROUND;
    uint2 p_range                         = point_ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int p_rounds                    = ((p_range.y - p_range.x + BLOCK_SAMPLES_PRE_ROUND - 1) / BLOCK_SAMPLES_PRE_ROUND);
    int p_toDo                            = p_range.y - p_range.x;

    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    __shared__ float3 collected_ray_planes[BLOCK_SIZE];

    for (int p_round = 0; p_round < p_rounds; p_round++, p_toDo -= BLOCK_SAMPLES_PRE_ROUND) {
        bool done[SAMPLES_PRE_ROUND]   = {false};
        float Depth[SAMPLES_PRE_ROUND] = {0.f};
        float T[SAMPLES_PRE_ROUND];
        uint32_t point_idx[SAMPLES_PRE_ROUND];
        float2 point_xy[SAMPLES_PRE_ROUND];
        uint32_t last_contributor[SAMPLES_PRE_ROUND] = {0};
        uint32_t contributor                         = 0;
        int point_done                               = 0;
        int point_num_round                          = 0;
#pragma unroll
        for (int p = 0; p < SAMPLES_PRE_ROUND; p++) {
            int progress = (p_round * SAMPLES_PRE_ROUND + p) * BLOCK_SIZE + block.thread_rank();
            if (p_range.x + progress < p_range.y) {
                T[p]         = 1.f;
                int pid      = point_list[p_range.x + progress];
                point_idx[p] = pid;
                point_xy[p]  = points2D[pid];
                point_num_round++;
            }
        }
        bool all_done = point_num_round == 0;
        int toDo      = range.y - range.x;
        for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
            // End if entire block votes that it is done rasterizing
            int num_done = __syncthreads_count(all_done);
            if (num_done == BLOCK_SIZE)
                break;
            // Collectively fetch per-Gaussian data from global to shared
            int progress = i * BLOCK_SIZE + block.thread_rank();
            if (range.x + progress < range.y) {
                int coll_id                                  = gaussian_list[range.x + progress];
                collected_xy[block.thread_rank()]            = gaussians2D[coll_id];
                collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
                float4 ray_plane                             = ray_planes[coll_id];
                collected_ray_planes[block.thread_rank()]    = {ray_plane.x, ray_plane.y, ray_plane.z};
            }
            block.sync();

            // Iterate over current batch
            for (int j = 0; !all_done && j < min(BLOCK_SIZE, toDo); j++) {
                contributor++;
                float2 xy        = collected_xy[j];
                float4 con_o     = collected_conic_opacity[j];
                float3 ray_plane = collected_ray_planes[j];
                for (int p = 0; p < point_num_round; p++) {
                    if (done[p])
                        continue;
                    float2 d    = {xy.x - point_xy[p].x, xy.y - point_xy[p].y};
                    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
                    if (power > 0.0f) {
                        continue;
                    }

                    // Eq. (2) from 3D Gaussian splatting paper.
                    // Obtain alpha by multiplying with Gaussian opacity
                    // and its exponential falloff from mean.
                    // Avoid numerical instabilities (see paper appendix).
                    float alpha = fminf(0.99f, con_o.w * expf(power));
                    if (alpha < 1.0f / 255.0f)
                        continue;
                    float test_T = T[p] * (1.f - alpha);
                    if (test_T < 0.0001f) {
                        done[p] = true;
                        point_done++;
                        all_done = point_done == point_num_round;
                        continue;
                    }

                    const float aT = alpha * T[p];
                    float t        = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                    Depth[p] += t * aT;
                    T[p]                = test_T;
                    last_contributor[p] = contributor;
                }
            }
        }
        for (int p = 0; p < point_num_round; p++) {
            float2 pixnf                = {(point_xy[p].x - static_cast<float>(W - 1) / 2.f) / focal_x,
                                           (point_xy[p].y - static_cast<float>(H - 1) / 2.f) / focal_y};
            const float rln             = rnorm3df(pixnf.x, pixnf.y, 1.f);
            float depth                 = Depth[p] * rln / fmaxf(1.f - T[p], 1e-7f);
            output[point_idx[p]]        = {pixnf.x * depth, pixnf.y * depth, depth};
            accum_depth[point_idx[p]]   = Depth[p];
            final_T[point_idx[p]]       = T[p];
            n_contrib[point_idx[p]]     = last_contributor[p];
            inside_output[point_idx[p]] = true;
            // inside_output[point_idx[i]] = T[i] < 1.f;
        }
    }
}

void FORWARD::preprocess_points(
    int PN,
    const float* points3D,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    float2* points2D,
    float* depths,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered) {
    preprocessPointsCUDA<<<(PN + 255) / 256, 256>>>(
        PN,
        points3D,
        viewmatrix,
        projmatrix,
        cam_pos,
        W, H,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        points2D,
        depths,
        grid,
        tiles_touched,
        prefiltered);
}

void FORWARD::integrate(
    const dim3 grid, dim3 block,
    const uint2* gaussian_ranges,
    const uint2* point_ranges,
    const uint32_t* gaussian_list,
    const uint32_t* point_list,
    int W, int H,
    float focal_x, float focal_y,
    const float2* points2D,
    const float2* gaussians2D,
    const float* colors,
    const float4* ray_planes,
    const float* point_depths,
    const float4* conic_opacity,
    const float* bg_color,
    float* out_color,
    float* out_alpha,
    bool* inside) {
    integrateCUDA<NUM_CHANNELS, SAMPLE_BATCH_SIZE><<<grid, block>>>(
        gaussian_ranges,
        point_ranges,
        gaussian_list,
        point_list,
        W, H,
        focal_x, focal_y,
        points2D,
        gaussians2D,
        colors,
        ray_planes,
        point_depths,
        conic_opacity,
        bg_color,
        out_color,
        out_alpha,
        inside);
}

void FORWARD::sampleDepth(
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
    uint32_t* n_contrib,
    float* accum_depth,
    float* final_T,
    float3* output,
    bool* inside) {
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
        output,
        inside);
}