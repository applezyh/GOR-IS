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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["rend_alpha"]
    elif output == 'normal':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
    elif output == 'depth':
        net_image = render_pkg["surf_depth"]
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image

def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]

    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image.clone()[:, ymin:ymax+1, xmin:xmax+1]

def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)

def th_dilate(data, kernel_size=10, iterations=1):
    """
    data: [1, H, W]
    return dilated_data: [1, H, W]
    
    """
    if kernel_size * iterations == 0:
        return data
    data_np = data[0].cpu().numpy()
    kernel = np.ones((kernel_size, kernel_size), data_np.dtype)
    data_np = cv2.dilate(data_np, kernel=kernel, iterations=iterations)
    dilated_data = torch.from_numpy(data_np)[None]

    return dilated_data.to(data.device)

import os

GAMMA = float(os.environ.get("GAMMA", 1.5))
    
def mapping(data):
    return data * GAMMA

def inverse_mapping(data):
    return data / GAMMA

from utils.point_utils import depths_to_points, world_points_to_screen


def cross_view_consistency(I1, I2, D1, D2, V1, V2, M=None, depth_thresh=0.5):
    """
    计算从 view1 重投影到 view2 的光度一致性误差
    """

    C1 = I1.shape[0]
    C2 = I2.shape[0]
    assert C1 == C2, "Channel must be the same for both images"

    if M is None:
        M = torch.ones_like(D1)

    # -------- Step1: project I1 to I2 --------
    world_pts1 = depths_to_points(V1, D1)  # [N, 3]
    uv, z_depth = world_points_to_screen(V2, world_pts1)  # [N, 2], [N,]

    # -------- Step2: sample I2 and D2 --------
    grid = uv.reshape(1, -1, 1, 2) * 2 - 1  # Normalize to [-1, 1]
    I2_warp = F.grid_sample(I2[None], grid, align_corners=False).reshape(C2, -1)  # [C2, N]
    D2_warp = F.grid_sample(D2[None], grid, align_corners=False).reshape(-1)  # [N]

    # -------- Step3: depth test --------
    depth_diff = torch.abs(D2_warp - z_depth)
    valid_mask = (depth_diff < depth_thresh).float() * M.reshape(-1)

    # -------- Step4: loss calc --------
    error = torch.abs(I1.reshape(C1, -1) - I2_warp)
    mean_error = (error * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return mean_error