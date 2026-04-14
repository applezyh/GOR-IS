#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modifications Copyright (C) 2026, [Yonghao Zhao / Nankai University]
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE file.
#
# For inquiries contact:
# - Original: george.drettakis@inria.fr
# - Modified version: applezyh@outlook.com
#

import math
from functools import partial

import torch
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)

from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal


def build_rasterizer(viewpoint_camera, pc, pipe, bg_color, kernel_size, scaling_modifier):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        require_depth=True,
        debug=pipe.debug,
    )

    return GaussianRasterizer(raster_settings=raster_settings)

def render_baking(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    rasterizer = build_rasterizer(viewpoint_camera, pc, pipe, bg_color, kernel_size, scaling_modifier)
    means2D = screenspace_points

    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    means3D = pc.get_xyz
    shs = pc.get_features
    rotations = pc.get_rotation

    rendered_image, radii, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    rendered_normal = (rendered_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    rendered_expected_depth = torch.nan_to_num(rendered_expected_depth, 0, 0, 0)
    rendered_median_depth = torch.nan_to_num(rendered_median_depth, 0, 0, 0)
    surf_depth = rendered_expected_depth * (1-pipe.depth_ratio) + (pipe.depth_ratio) * rendered_median_depth

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2,0,1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    outputs = {   
        "render": rendered_image,
        "rend_alpha": rendered_alpha,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
        "rend_normal": rendered_normal,
    }

    label = pc.get_label
    def rasterizer_fn(colors_precomp, fn, detach=False, m=means3D, s=scales, r=rotations, o=opacity):
        if detach:
            m, s, r, o = (m.detach(), s.detach(), r.detach(), o.detach())
        return fn(
            means3D = m,
            means2D = means2D.clone().detach(),
            shs = None,
            colors_precomp = colors_precomp,
            opacities = o,
            scales = s,
            rotations = r,
            cov3D_precomp = None,
        )[0]

    attrs = torch.cat((label,) * 3, dim=1)
    rendered_label = rasterizer_fn(attrs, fn=rasterizer, detach=True, o=opacity)

    outputs.update({
        "rend_label": rendered_label[0:1],
    })

    return outputs


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    rasterizer = build_rasterizer(viewpoint_camera, pc, pipe, bg_color, kernel_size, scaling_modifier)

    means2D = screenspace_points

    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    means3D = pc.get_xyz
    shs = pc.get_features
    rotations = pc.get_rotation

    rendered_image, radii, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal_world = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    rendered_normal = (rendered_normal_world.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)

    rendered_expected_depth = torch.nan_to_num(rendered_expected_depth, 0, 0, 0)
    rendered_median_depth = torch.nan_to_num(rendered_median_depth, 0, 0, 0)
    surf_depth = rendered_expected_depth * (1-pipe.depth_ratio) + (pipe.depth_ratio) * rendered_median_depth

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2,0,1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    outputs = {   
        "render": rendered_image,
        "rend_alpha": rendered_alpha,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
        "rend_normal": rendered_normal,
        "rend_normal_w": rendered_normal_world,
    }
    
    diffuse = pc.get_diffuse
    fresnel = pc.get_fresnel
    roughness = pc.get_roughness
    reflect = pc.get_reflect
    label = pc.get_label

    def rasterizer_fn(colors_precomp, fn, detach=False, m=means3D, s=scales, r=rotations, o=opacity):
        if detach:
            m, s, r, o = (m.detach(), s.detach(), r.detach(), o.detach())
        return fn(
            means3D = m,
            means2D = means2D.clone().detach(),
            shs = None,
            colors_precomp = colors_precomp,
            opacities = o,
            scales = s,
            rotations = r,
            cov3D_precomp = None,
        )[0]

    rendered_diffuse = rasterizer_fn(diffuse, fn=rasterizer)
    rendered_fresnel = rasterizer_fn(fresnel, fn=rasterizer)
    attrs = torch.cat((label, reflect, roughness), dim=1)
    rendered = rasterizer_fn(attrs, fn=rasterizer, detach=True)

    outputs.update({
        "rend_diffuse": rendered_diffuse,
        "rend_fresnel": rendered_fresnel,
        "rend_label": rendered[0:1],
        "rend_reflect": rendered[1:2],
        "rend_roughness": rendered[2:3],
    })

    return outputs
