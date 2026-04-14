# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# Modifications Copyright (C) 2026, [Yonghao Zhao / Nankai University]
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.


import nvdiffrast.torch as dr
import torch

from utils import renderutils as ru
from utils.general_utils import safe_normalize


def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

#----------------------------------------------------------------------------
# Image scaling
#----------------------------------------------------------------------------

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = safe_normalize(cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, resolution=256, init_value=0.8):
        super(EnvironmentLight, self).__init__()     
        self.base = torch.nn.Parameter(
            torch.full((6, resolution, resolution, 3), init_value, dtype=torch.float32),
            requires_grad=True,
        )

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99):
        self.base.data.clamp_min_(0.0)
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 

        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)
    
    def __call__(self, reflvec, roughness):
        prefix = reflvec.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            reflvec = reflvec.reshape(1, 1, -1, reflvec.shape[-1])
        roughness = roughness.reshape(1, 1, -1, 1)

        miplevel = self.get_mip(roughness)
        light = dr.texture(
            self.specular[0][None, ...], 
            reflvec,
            mip=list(m[None, ...] for m in self.specular[1:]), 
            mip_level_bias=miplevel[..., 0], 
            filter_mode='linear-mipmap-linear', 
            boundary_mode='cube'
        )
        light = light.view(*prefix, -1)

        return light
    
    def pure_env(self, reflvec):
        prefix = reflvec.shape[:-1]
        if len(prefix) != 3:  # Reshape to [B, H, W, -1] if necessary
            reflvec = reflvec.reshape(1, 1, -1, reflvec.shape[-1])
        light = dr.texture(
            self.base[None, ...], 
            reflvec,
            filter_mode='linear', 
            boundary_mode='cube'
        )
        light = light.view(*prefix, -1)

        return light
