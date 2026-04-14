/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "auxiliary.h"
#include "render_forward.h"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ __forceinline__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped) {
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    glm::vec3 pos = means[idx];
    glm::vec3 dir = pos - campos;
    dir           = dir / glm::length(dir);

    glm::vec3* sh    = ((glm::vec3*)shs) + idx * max_coeffs;
    glm::vec3 result = SH_C0 * sh[0];

    if (deg > 0) {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result  = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                     SH_C2[0] * xy * sh[4] +
                     SH_C2[1] * yz * sh[5] +
                     SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                     SH_C2[3] * xz * sh[7] +
                     SH_C2[4] * (xx - yy) * sh[8];

            if (deg > 2) {
                result = result +
                         SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
                         SH_C3[1] * xy * z * sh[10] +
                         SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
                         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                         SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
                         SH_C3[5] * z * (xx - yy) * sh[14] +
                         SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
            }
        }
    }
    result += 0.5f;

    // RGB colors are clamped to positive values. If values are
    // clamped, we need to keep track of this for the backward pass.
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
    return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
template <bool INTE = false>
__device__ __forceinline__ bool computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, float kernel_size, const float* cov3D, const float* viewmatrix,
                                             float* cov2D, float4* normal, float4* ray_plane, float& coef, const glm::vec3* scale = nullptr, const float4* rotation = nullptr, const float mod = 1.f) {
    // The following models the steps outlined by equations 29
    // and 31 in "EWA Splatting" (Zwicker et al., 2002).
    // Additionally considers aspect / scaling of viewport.
    // Transposes used to account for row-/column-major conventions.
    float3 t       = transformPoint4x3(mean, viewmatrix);
    const float tc = norm3df(t.x, t.y, t.z);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    float txtz       = t.x / t.z;
    float tytz       = t.y / t.z;
    t.x              = fminf(limx, fmaxf(-limx, txtz)) * t.z;
    t.y              = fminf(limy, fmaxf(-limy, tytz)) * t.z;
    txtz             = t.x / t.z;
    tytz             = t.y / t.z;

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.f, -(focal_x * t.x) / (t.z * t.z),
        0.f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.f, 0.f, 0.f);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 cov;
    glm::mat3 cov_cam_inv;

    bool well_conditioned;

    auto find_min_from_triple = [](auto v_list) -> unsigned int {
        int bigger_id = int(v_list[0] < v_list[1]);
        int idx[2]    = {(bigger_id + 1) % 3, (bigger_id + 2) % 3};
        return idx[int(v_list[idx[0]] > v_list[idx[1]])];
    };

    if (scale) {
        // Create scaling matrix
        glm::mat3 S                = glm::mat3(1.0f);
        glm::mat3 S_inv            = glm::mat3(1.0f);
        const float scale_local[3] = {mod * scale->x, mod * scale->y, mod * scale->z};
        S[0][0]                    = scale_local[0];
        S[1][1]                    = scale_local[1];
        S[2][2]                    = scale_local[2];

        S_inv[0][0] = __frcp_rn(scale_local[0]);
        S_inv[1][1] = __frcp_rn(scale_local[1]);
        S_inv[2][2] = __frcp_rn(scale_local[2]);

        unsigned int min_id = find_min_from_triple(scale_local);
        well_conditioned    = scale_local[min_id] > 1E-7;

        // Normalize quaternion to get valid rotation
        float4 rot = *rotation;
        float r    = rot.x;
        float x    = rot.y;
        float y    = rot.z;
        float z    = rot.w;

        // Compute rotation matrix from quaternion
        glm::mat3 R = glm::mat3(
            1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
            2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
            2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

        glm::mat3 M = S * R * T;
        // Compute 3D world covariance matrix Sigma
        cov = glm::transpose(M) * M;
        if (well_conditioned) {
            glm::mat3 M_inv = S_inv * R * W;
            cov_cam_inv     = glm::transpose(M_inv) * M_inv;
        } else {
            glm::vec3 r     = {R[0][min_id], R[1][min_id], R[2][min_id]};
            glm::vec3 M_inv = r * W;
            cov_cam_inv     = glm::outerProduct(M_inv, M_inv);
        }
    } else {
        glm::mat3 Vrk = glm::mat3(
            cov3D[0], cov3D[1], cov3D[2],
            cov3D[1], cov3D[3], cov3D[4],
            cov3D[2], cov3D[4], cov3D[5]);

        cov = glm::transpose(T) * glm::transpose(Vrk) * T;

        glm::mat3 Vrk_eigen_vector;
        glm::vec3 Vrk_eigen_value;
        int D = glm_modification::findEigenvaluesSymReal(Vrk, Vrk_eigen_value, Vrk_eigen_vector);

        unsigned int min_id = find_min_from_triple(Vrk_eigen_value);

        well_conditioned = Vrk_eigen_value[min_id] > 1E-8;
        glm::vec3 eigenvector_min;
        glm::mat3 Vrk_inv;
        if (well_conditioned) {
            glm::mat3 diag = glm::mat3(1 / Vrk_eigen_value[0], 0, 0,
                                       0, 1 / Vrk_eigen_value[1], 0,
                                       0, 0, 1 / Vrk_eigen_value[2]);
            Vrk_inv        = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
        } else {
            eigenvector_min = Vrk_eigen_vector[min_id];
            Vrk_inv         = glm::outerProduct(eigenvector_min, eigenvector_min);
        }
        cov_cam_inv = glm::transpose(W) * Vrk_inv * W;
    }

    // output[0] = { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
    cov2D[0]          = float(cov[0][0] + kernel_size);
    cov2D[1]          = float(cov[0][1]);
    cov2D[2]          = float(cov[1][1] + kernel_size);
    const float det_0 = fmaxf(1e-6f, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
    const float det_1 = fmaxf(1e-6f, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
    coef              = sqrtf(det_0 / det_1);

    // glm::mat3 testm = glm::mat3{
    // 	1,2,3,
    // 	4,5,6,
    // 	7,8,9,
    // };
    // glm::vec3 testv = {1,1,1};
    // glm::vec3 resultm = testm * testv;
    // printf("%f %f %f\n", resultm[0], resultm[1], resultm[2]); 12.000000 15.000000 18.000000

    glm::vec3 uvh    = {txtz, tytz, 1};
    glm::vec3 uvh_m  = cov_cam_inv * uvh;
    glm::vec3 uvh_mn = glm::normalize(uvh_m);

    if (isnan(uvh_mn.x)) {
        *ray_plane = {0};
        *normal    = {0, 0, -1, 0};
    } else {
        float u2 = txtz * txtz;
        float v2 = tytz * tytz;
        float uv = txtz * tytz;

        const float l = norm3df(t.x, t.y, t.z);

        glm::mat3 nJ_inv = glm::mat3(
            v2 + 1, -uv, 0,
            -uv, u2 + 1, 0,
            -txtz, -tytz, 0);

        float vbn           = glm::dot(uvh_mn, uvh);
        float ray_len2      = u2 + v2 + 1;
        float factor_normal = l / ray_len2;
        glm::vec3 plane     = nJ_inv * (uvh_mn / max(vbn, 1e-7f));

        if constexpr (INTE) {
            float rsigmat;
            if (well_conditioned) {
                float vb = glm::dot(uvh_m, uvh);
                rsigmat  = well_conditioned ? sqrtf(vb / ray_len2) : 0.f;
            } else {
                rsigmat = 0.f;
            }
            *ray_plane = {plane[0] * factor_normal / focal_x, plane[1] * factor_normal / focal_y, tc, rsigmat};
        } else {
            *ray_plane = {plane[0] * factor_normal / focal_x, plane[1] * factor_normal / focal_y, tc, 0.f};
        }

        glm::vec3 ray_normal_vector = {-plane[0] * factor_normal, -plane[1] * factor_normal, -1};
        glm::mat3 nJ                = glm::mat3(
            1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
            0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
            t.x / l, t.y / l, t.z / l);
        glm::vec3 cam_normal_vector = nJ * ray_normal_vector;
        glm::vec3 normal_vector     = glm::normalize(cam_normal_vector);

        *normal = {normal_vector.x, normal_vector.y, normal_vector.z, 0.f};
    }
    return well_conditioned;
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ __forceinline__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D) {
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0]     = mod * scale.x;
    S[1][1]     = mod * scale.y;
    S[2][2]     = mod * scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot; // / glm::length(rot);
    float r     = q.x;
    float x     = q.y;
    float y     = q.z;
    float z     = q.w;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C, bool INTE = false>
__global__ void preprocessCUDA(
    int P, int D, int M,
    const float* means3D,
    const float* colors_precomp,
    const float* opacities,
    const glm::vec3* scales,
    const float4* rotations,
    const float* cov3D_precomp,
    const float* shs,
    const float scale_modifier,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, const int H,
    const float focal_x, const float focal_y,
    const float tan_fovx, const float tan_fovy,
    const float kernel_size,
    int* radii,
    bool* clamped,
    float2* means2D,
    float* depths,
    float4* ray_planes,
    float4* normals,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Gaussian will not be processed further.
    radii[idx]         = 0;
    tiles_touched[idx] = 0;
    // Perform near culling, quit if outside.
    const float3 p_orig = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
    float3 p_view;
    if (!in_frustum(p_orig, viewmatrix, projmatrix, prefiltered, p_view))
        return;
    // Transform point by projecting
    float4 p_hom  = transformPoint4x4(p_orig, projmatrix);
    float p_w     = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};

    // If 3D covariance matrix is precomputed, use it, otherwise compute
    // from scaling and rotation parameters.
    const float* cov3D = nullptr;
    if (cov3D_precomp != nullptr) {
        cov3D = cov3D_precomp + idx * 6;
    }

    // Compute 2D screen-space covariance matrix
    float cov2D[3];
    float ceof;
    computeCov2D<INTE>(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, kernel_size, cov3D, viewmatrix, cov2D,
                       normals + idx, ray_planes + idx, ceof, scales + idx, rotations + idx, scale_modifier);
    const float3 cov = {cov2D[0], cov2D[1], cov2D[2]};

    // Invert covariance (EWA algorithm)
    float det = (cov.x * cov.z - cov.y * cov.y);
    if (det == 0.0f)
        return;
    float det_inv = 1.f / det;
    float3 conic  = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};

    // Compute extent in screen space (by finding eigenvalues of
    // 2D covariance matrix). Use extent to compute a bounding rectangle
    // of screen-space tiles that this Gaussian overlaps with. Quit if
    // rectangle covers 0 tiles.
    float mid          = 0.5f * (cov.x + cov.z);
    float lambda1      = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
    float lambda2      = mid - sqrtf(fmaxf(0.1f, mid * mid - det));
    float my_radius    = ceilf(3.f * sqrtf(fmaxf(lambda1, lambda2)));
    float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
    uint2 rect_min, rect_max;
    getRect(point_image, my_radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;

    // If colors have been precomputed, use them, otherwise convert
    // spherical harmonics coefficients to RGB color.
    if (colors_precomp == nullptr) {
        glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *cam_pos, shs, clamped);
        rgb[idx * C + 0] = result.x;
        rgb[idx * C + 1] = result.y;
        rgb[idx * C + 2] = result.z;
    }

    // Store some useful helper data for the next steps.
    // depths[idx]          = p_view.z;
    depths[idx]  = norm3df(p_view.x, p_view.y, p_view.z);
    radii[idx]   = my_radius;
    means2D[idx] = point_image;
    // Inverse 2D covariance and opacity neatly pack into one float4
    conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx] * ceof};
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS, bool GEOMETRY>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    renderCUDA(
        const uint2* __restrict__ ranges,
        const uint32_t* __restrict__ point_list,
        int W, int H,
        const float2* __restrict__ points_xy_image,
        const float4* __restrict__ conic_opacity,
        const float* __restrict__ features,
        const float4* __restrict__ ray_planes,
        const float4* __restrict__ normals,
        const float focal_x,
        const float focal_y,
        uint32_t* __restrict__ n_contrib,
        const float* __restrict__ bg_color,
        float* __restrict__ out_color,
        float* __restrict__ out_alpha,
        float* __restrict__ out_normal,
        float* __restrict__ out_depth,
        float* __restrict__ out_mdepth,
        float* __restrict__ accum_depth,
        float* __restrict__ normal_length) {
    // Identify current tile and associated min/max pixel range.
    auto block                 = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min              = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix_max              = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    uint2 pix                  = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id            = W * pix.y + pix.x;
    float2 pixf                = {static_cast<float>(pix.x), static_cast<float>(pix.y)};
    const float2 pixnf         = {(pixf.x - static_cast<float>(W - 1) / 2.f) / focal_x, (pixf.y - static_cast<float>(H - 1) / 2.f) / focal_y};
    const float rln            = rnorm3df(pixnf.x, pixnf.y, 1.f);

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range      = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo         = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float collected_feature[BLOCK_SIZE * CHANNELS];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
    [[maybe_unused]] __shared__ float3 collected_ray_planes[BLOCK_SIZE];
    [[maybe_unused]] __shared__ float3 collected_normals[BLOCK_SIZE];

    // Initialize helper variables
    float T                            = 1.0f;
    uint32_t contributor               = 0;
    uint32_t last_contributor          = 0;
    uint32_t max_contributor           = -1;
    float C[CHANNELS]                  = {0};
    [[maybe_unused]] float Depth       = 0;
    [[maybe_unused]] float mDepth      = 0;
    [[maybe_unused]] float Normal[3]   = {0};
    [[maybe_unused]] float last_depth  = 0;
    [[maybe_unused]] float last_weight = 0;

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id                                  = point_list[range.x + progress];
            collected_xy[block.thread_rank()]            = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
            for (int ch = 0; ch < CHANNELS; ch++)
                collected_feature[ch * BLOCK_SIZE + block.thread_rank()] = features[coll_id * CHANNELS + ch];
            if constexpr (GEOMETRY) {
                float4 ray_plane                          = ray_planes[coll_id];
                float4 normal                             = normals[coll_id];
                collected_ray_planes[block.thread_rank()] = {ray_plane.x, ray_plane.y, ray_plane.z};
                collected_normals[block.thread_rank()]    = {normal.x, normal.y, normal.z};
            }
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            // Keep track of current position in range
            contributor++;

            // Resample using conic matrix (cf. "Surface
            // Splatting" by Zwicker et al., 2001)
            float2 xy    = collected_xy[j];
            float2 d     = {xy.x - pixf.x, xy.y - pixf.y};
            float4 con_o = collected_conic_opacity[j];
            float power  = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
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
            float test_T = T * (1.f - alpha);
            if (test_T < 0.0001f) {
                done = true;
                continue;
            }

            const float aT = alpha * T;
            // Eq. (3) from 3D Gaussian splatting paper.
            for (int ch = 0; ch < CHANNELS; ch++)
                C[ch] += collected_feature[j + BLOCK_SIZE * ch] * aT;

            if constexpr (GEOMETRY) {
                float3 ray_plane = collected_ray_planes[j];
                float3 normal    = collected_normals[j];
                float t          = ray_plane.x * d.x + ray_plane.y * d.y + ray_plane.z;
                Depth += t * aT;
                bool before_median = T > 0.5;
                mDepth             = before_median ? t : mDepth;
                max_contributor    = before_median ? contributor : max_contributor;
                Normal[0] += normal.x * aT;
                Normal[1] += normal.y * aT;
                Normal[2] += normal.z * aT;
            }

            T = test_T;

            // Keep track of last range entry to update this
            // pixel.
            last_contributor = contributor;
        }
    }

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside) {
        n_contrib[pix_id]         = last_contributor;
        n_contrib[pix_id + H * W] = max_contributor;
#pragma unroll
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
        out_alpha[pix_id] = 1.f - T; // 1 - T;

        if constexpr (GEOMETRY) {
            float depth_ln        = Depth * rln;
            accum_depth[pix_id]   = depth_ln;
            out_depth[pix_id]     = last_contributor ? depth_ln / (1.f - T) : 0.f;
            out_mdepth[pix_id]    = mDepth * rln;
            float len_normal      = norm3df(Normal[0], Normal[1], Normal[2]);
            normal_length[pix_id] = len_normal;
            len_normal            = fmaxf(len_normal, NORMALIZE_EPS);
#pragma unroll
            for (int ch = 0; ch < 3; ch++)
                out_normal[ch * H * W + pix_id] = Normal[ch] / len_normal;
        }
    }
}

// the Bool inputs can be replaced by an enumeration variable for different functions.
void FORWARD::render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float4* conic_opacity,
    const float* colors,
    const float4* ray_planes,
    const float4* normals,
    const float focal_x,
    const float focal_y,
    uint32_t* n_contrib,
    const float* bg_color,
    float* out_color,
    float* out_alpha,
    float* out_normal,
    float* out_depth,
    float* out_mdepth,
    float* accum_depth,
    float* normal_length,
    bool require_depth) {
#define RENDER_CUDA_CALL(template_depth)                             \
    renderCUDA<NUM_CHANNELS, template_depth><<<grid, block>>>(       \
        ranges, point_list, W, H,                                    \
        means2D, conic_opacity, colors, ray_planes, normals,         \
        focal_x, focal_y, n_contrib, bg_color, out_color, out_alpha, \
        out_normal, out_depth, out_mdepth, accum_depth, normal_length)

    if (require_depth)
        RENDER_CUDA_CALL(true);
    else
        RENDER_CUDA_CALL(false);

#undef RENDER_CUDA_CALL
}

void FORWARD::preprocess(
    int P, int D, int M,
    const float* means3D,
    const float* colors_precomp,
    const float* opacities,
    const glm::vec3* scales,
    const float4* rotations,
    const float* cov3D_precomp,
    const float* shs,
    const float scale_modifier,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, const int H,
    const float focal_x, const float focal_y,
    const float tan_fovx, const float tan_fovy,
    const float kernel_size,
    int* radii,
    bool* clamped,
    float2* means2D,
    float* depths,
    float4* ray_planes,
    float4* normals,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered,
    bool integrate) {
    if (integrate)
        preprocessCUDA<NUM_CHANNELS, true><<<(P + 255) / 256, 256>>>(
            P, D, M,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            shs,
            scale_modifier,
            viewmatrix,
            projmatrix,
            cam_pos,
            W, H,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            kernel_size,
            radii,
            clamped,
            means2D,
            depths,
            ray_planes,
            normals,
            rgb,
            conic_opacity,
            grid,
            tiles_touched,
            prefiltered);
    else
        preprocessCUDA<NUM_CHANNELS, false><<<(P + 255) / 256, 256>>>(
            P, D, M,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            shs,
            scale_modifier,
            viewmatrix,
            projmatrix,
            cam_pos,
            W, H,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            kernel_size,
            radii,
            clamped,
            means2D,
            depths,
            ray_planes,
            normals,
            rgb,
            conic_opacity,
            grid,
            tiles_touched,
            prefiltered);
}
