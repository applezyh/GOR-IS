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

import numpy as np
import torch
import torch.nn.functional as F


def rays(view):
    """
        view: view camera
    """
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W).float().cuda() + 0.5, torch.arange(H).float().cuda() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T # (N, 3)
    rays_o = c2w[:3,3].expand(rays_d.shape) # (N, 3)
    return rays_o, rays_d

def ray_depths_to_z_depths(view, depthmap):
    _, rays_d = rays(view)
    z_depth = depthmap.reshape(-1, 1) / torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    return z_depth.reshape(depthmap.shape) # (1, H, W)

def z_depths_to_ray_depths(view, depthmap):
    _, rays_d = rays(view)
    ray_depth = depthmap.reshape(-1, 1) * torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    return ray_depth.reshape(depthmap.shape) # (1, H, W)

def depths_to_points(view, depthmap, z_depth=True):
    rays_o, rays_d = rays(view)
    if not z_depth:
        rays_d = F.normalize(rays_d, dim=-1)
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def world_points_to_screen(view, points_world):
    """
    把世界坐标点转换为归一化像素坐标 [0,1]^2。

    Args:
        view: 有属性 `world_view_transform` (4x4), `full_proj_transform` (4x4),
              `image_width`, `image_height`.
        points_world: tensor, shape (..., 3) 或 (N,3)

    Returns:
        tensor, shape (..., 2)，每个点的 (u, v) ∈ [0, 1]，u 对应 x（水平），v 对应 y（垂直）。
    """
    import torch

    orig_shape = points_world.shape
    assert orig_shape[-1] == 3, "points_world 最后一个维度必须为 3"
    pts = points_world.reshape(-1, 3)
    device = pts.device
    dtype = pts.dtype

    # 齐次坐标
    ones = torch.ones((pts.shape[0], 1), device=device, dtype=dtype)
    pts_h = torch.cat([pts, ones], dim=1)  # (N,4)

    # world -> camera
    # 假定 view.world_view_transform 是 4x4 world->camera
    world2cam = view.world_view_transform.T
    cam_pts_h = (world2cam @ pts_h.T).T  # (N,4)
    z_depth = cam_pts_h[..., 2]

    # camera -> clip space (投影)
    clip = (view.projection_matrix.T @ cam_pts_h.T).T  # (N,4)

    # 透视除法到 NDC
    w = clip[:, 3:4].clamp(min=1e-6)
    ndc = clip[:, :3] / w  # (N,3)

    # NDC [-1,1] -> 归一化像素 [0,1]
    uv = (ndc[:, :2] + 1.0) / 2.0  # (N,2)

    # 复原原始形状
    uv = uv.reshape(orig_shape[:-1] + (2,))
    return uv, z_depth.reshape(orig_shape[:-1])

def depth_to_normal(view, depth, z_depth=True):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth, z_depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def get_bbox(filtered_points, percentile=90, outlier_factor=1.0):
    """
    Estimate 3D bounding box size from point cloud using percentile clipping.

    Args:
        filtered_points (np.ndarray): (N, 3) array of 3D points
        percentile (float): central percentile to keep (e.g., 50 means keep 25%~75%)
        outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.

    Returns:
        bbox_min (np.ndarray): (3,) min corner
        bbox_max (np.ndarray): (3,) max corner
    """

    if len(filtered_points) == 0:
        return np.zeros(3), np.zeros(3), np.zeros(3)

    points = np.asarray(filtered_points)

    # 计算上下分位点
    lower_percentile = (100 - percentile) / 2
    upper_percentile = 100 - lower_percentile

    Q1 = np.percentile(points, lower_percentile, axis=0)
    Q3 = np.percentile(points, upper_percentile, axis=0)

    bbox_min = Q1
    bbox_max = Q3

    return bbox_min, bbox_max

def is_point_in_bbox(points, bbox_min, bbox_max, tol=0.4):
    """
    Check whether 3D point(s) lie inside an axis-aligned bounding box.

    Args:
        points (np.ndarray): (N, 3)
        bbox_min (np.ndarray): (3,) min corner
        bbox_max (np.ndarray): (3,) max corner
        tol (float): tolerance margin

    Returns:
        inside - (N,) boolean array
    """
    bbox_size = bbox_max - bbox_min

    inside = np.all(
        (points >= (bbox_min - tol * bbox_size)) & (points <= (bbox_max + tol * bbox_size)),
        axis=1
    )

    return inside

from scipy.spatial import Delaunay


def points_inside_point_convex_hull(points, ref_points, expand_ratio=1.0):
    center = np.mean(ref_points, axis=0)
    expanded_ref = center + (ref_points - center) * expand_ratio

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(expanded_ref)

    # Determine which points from the original point cloud are inside the convex hull
    mask = delaunay.find_simplex(points) >= 0
    return mask

def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = points_inside_point_convex_hull(point_cloud.cpu().numpy(), filtered_masked_points)

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask