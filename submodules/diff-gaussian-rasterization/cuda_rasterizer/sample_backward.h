#ifndef CUDA_SAMPLE_BACKWARD_H_INCLUDED
#define CUDA_SAMPLE_BACKWARD_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD {

void preprocess_points(
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
    float3* dL_dpoints3D);

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
    const uint32_t* n_contrib,
    const float* accum_depth,
    const float* final_T,
    const float3* dL_doutput,
    float3* dL_dgaussians2D,
    float4* dL_dconic2D,
    float4* dL_dray_planes,
    float2* dL_dpoints2D);
} // namespace BACKWARD

#endif