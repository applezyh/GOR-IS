#ifndef CUDA_RASTERIZER_SAMPLE_H_INCLUDED
#define CUDA_RASTERIZER_SAMPLE_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
//  Perform initial steps for each Point prior to integration.
void preprocess_points(
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
    bool prefiltered);

void integrate(
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
    bool* inside);

void sampleDepth(
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
    bool* inside);
} // namespace FORWARD

#endif