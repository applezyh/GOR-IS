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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

constexpr int NUM_CHANNELS      = 3;
constexpr int BLOCK_X           = 16;
constexpr int BLOCK_Y           = 16;
constexpr int BLOCK_SIZE        = BLOCK_X * BLOCK_Y;
constexpr int NUM_WARPS         = BLOCK_SIZE / 32;
constexpr int SAMPLE_BATCH_SIZE = 2;
constexpr float NEAR_PLANE      = 0.2f;
constexpr float FAR_PLANE       = 100.f;
constexpr float NORMALIZE_EPS   = 1.0E-12F;

#endif