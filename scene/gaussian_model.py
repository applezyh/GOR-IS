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


import os

import numpy as np
import torch
import trimesh
from gtracer import GaussianTracer
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from utils.general_utils import (build_rotation, build_scaling_rotation, dot,
                                 get_expon_lr_func, inverse_sigmoid,
                                 safe_normalize, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.light_utils import EnvironmentLight
from utils.neural_utils import SimpleNet
from utils.point_utils import (depths_to_points, points_inside_convex_hull,
                               rays, z_depths_to_ray_depths)
from utils.render_utils import screen_mips
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


def process_depth(depth, alpha):
    return torch.where(alpha > 0, depth, -1.0)

from enum import Enum

class Removal(Enum):
    off = 1
    remove_object = 2
    remove_background = 3

class GaussianModel:
    # use mip-splatting filters
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.gaussian_tracer = GaussianTracer(transmittance_min=0.0001)
        self.icosahedron = trimesh.creation.icosahedron()
        self.unit_icosahedron_vertices = torch.from_numpy(self.icosahedron.vertices).float().cuda() * 1.2584 
        self.unit_icosahedron_faces = torch.from_numpy(self.icosahedron.faces).long().cuda()
        self.env_light = EnvironmentLight(resolution=256, init_value=0.5).cuda()
        self.roughness_net = SimpleNet(in_dim=3, out_dim=1, latent_dim=8, layer_num=2).cuda()
        self.alpha_min = 1 / 255.

        self.use_screen_filter = True
        self.removal = Removal.off

    def get_boundings(self, alpha_min):
        mu = self.get_xyz
        scales, opacity = self.get_scaling_n_opacity_with_3D_filter
        L = build_scaling_rotation(scales, self._rotation)
        
        vertices_b = (2 * (opacity/alpha_min).log()).sqrt()[:, None] * (self.unit_icosahedron_vertices[None] @ L.transpose(-1, -2)) + mu[:, None]
        faces_b = self.unit_icosahedron_faces[None] + torch.arange(mu.shape[0], device="cuda")[:, None, None] * 12
        gs_id = torch.arange(mu.shape[0], device="cuda")[:, None].expand(-1, faces_b.shape[1])
        return vertices_b.reshape(-1, 3), faces_b.reshape(-1, 3), gs_id.reshape(-1)

    def build_bvh(self):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min)
        self.gaussian_tracer.build_bvh(vertices_b, faces_b, gs_id)
        
    def update_bvh(self):
        vertices_b, faces_b, gs_id = self.get_boundings(alpha_min=self.alpha_min)
        self.gaussian_tracer.update_bvh(vertices_b, faces_b, gs_id)
        
    def trace(self, rays_o, rays_d, color_override=None):
        means3D = self.get_xyz
        shs = self.get_features
        scales, opacity = self.get_scaling_n_opacity_with_3D_filter
        SinvR = build_scaling_rotation(1 / scales, self._rotation)

        deg = self.active_sh_degree
        if color_override is not None:
            shs = torch.zeros_like(shs)
            shs[:, 0, :3] = color_override
            deg = 0
    
        color, depth, alpha = self.gaussian_tracer.trace(rays_o, rays_d, means3D, opacity, SinvR, shs, alpha_min=self.alpha_min, deg=deg)
        
        alpha_ = alpha[..., None]
        color = torch.where(alpha_ < 1 - self.gaussian_tracer.transmittance_min, color, color / alpha_)
        depth = torch.where(alpha < 1 - self.gaussian_tracer.transmittance_min, depth, depth / alpha)
        alpha = torch.where(alpha < 1 - self.gaussian_tracer.transmittance_min, alpha, torch.ones_like(alpha))
        
        return {
            "color": color,
            "depth": depth,
            "alpha" : alpha,
        }

    def reflect_trace(self, points, reflect_d, alpha, color_override=None):
        self.build_bvh()
        valid_mask = alpha > 0
        outputs = self.trace((points + 0.1 * reflect_d.detach())[valid_mask], reflect_d[valid_mask], color_override)
        
        N = points.shape[0]
        traced_color = torch.zeros((N, 3), device=alpha.device)
        traced_alpha = torch.zeros(N, device=alpha.device)
        traced_visibility = torch.zeros(N, device=alpha.device)
        traced_depth = torch.zeros(N, device=alpha.device)

        traced_color[valid_mask] = outputs["color"]
        traced_alpha[valid_mask] = outputs["alpha"]
        traced_visibility[valid_mask] = 1 - outputs["alpha"]
        traced_depth[valid_mask] = outputs["depth"]

        return {
            "color": traced_color,
            "alpha": traced_alpha,
            "visibility": traced_visibility,
            "depth": traced_depth,
        }

    def screen_filtering(self, data, normal, roughness, ray_depth, view_depth, return_roughness=False):
        neural_input = torch.cat((roughness, ray_depth, view_depth), dim=0)
        screen_roughness = self.roughness_net(neural_input[None])[0]
        blured = screen_mips(data, normal, screen_roughness, kernel_size=5, mip_level=4)

        return (blured, screen_roughness) if return_roughness else blured

    def prepare_reflection(self, viewpoint_cam, normal):
        normal_flat = normal.permute(1, 2, 0).reshape(-1, 3)
        view_d_flat = safe_normalize(rays(viewpoint_cam)[1])
        reflect_d_flat = safe_normalize(view_d_flat - 2 * dot(normal_flat, view_d_flat) * normal_flat)

        return reflect_d_flat

    def pbr(
      self,
      viewpoint_cam,
      alpha: torch.Tensor,
      normal: torch.Tensor,
      z_depth: torch.Tensor,
      diffuse: torch.Tensor,
      fresnel: torch.Tensor,
      roughness: torch.Tensor,
      background: torch.Tensor = None,
    ):
        if background is None:
            background = torch.zeros(3, device=alpha.device)
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width

        normal = safe_normalize(normal, dim=0)
        points_flat = depths_to_points(viewpoint_cam, z_depth)
        reflect_d_flat = self.prepare_reflection(viewpoint_cam, normal)
        outputs = self.reflect_trace(points_flat, reflect_d_flat, alpha.flatten())

        indirect_spec = outputs["color"].reshape(H, W, 3).permute(2, 0, 1)
        ray_alpha = outputs["alpha"].reshape(1, H, W)
        visibility = outputs["visibility"].reshape(1, H, W)

        screen_roughness = torch.zeros_like(roughness)
        if self.use_screen_filter:
            # screen-space filtering to simulate glossy reflection
            # Depth is used as a conditional input to adjust the filtering strength.
            ray_depth = outputs["depth"].reshape(1, H, W)
            view_depth = z_depths_to_ray_depths(viewpoint_cam, z_depth)
            ray_depth = process_depth(ray_depth, ray_alpha).detach()
            view_depth = process_depth(view_depth, alpha).detach()

            blured, screen_roughness = self.screen_filtering(
                torch.cat((indirect_spec, visibility), dim=0),
                normal.detach(),
                roughness,
                ray_depth,
                1 / view_depth,
                return_roughness=True,
            )
            indirect_spec, visibility = blured.split([3, 1], dim=0)

        self.env_light.build_mips()
        direct_spec = self.env_light(reflect_d_flat, roughness.reshape(-1, 1)).reshape(H, W, 3).permute(2, 0, 1)

        NoL = dot(normal, reflect_d_flat.reshape(H, W, 3).permute(2, 0, 1), dim=0).detach()
        reflectance = (fresnel + (1 - fresnel) * torch.pow(1 - NoL, 5.0))

        specular = (indirect_spec + direct_spec * visibility) * reflectance
        render_color = (diffuse + specular) * alpha + (1 - alpha) * background[:, None, None]

        return {
            "render_color": render_color,
            "diffuse": diffuse * alpha,
            "specular": specular * alpha,
            "visibility": visibility * alpha,
            "screen_roughness": screen_roughness * alpha,
        }

    @torch.no_grad()
    def object_effect(
      self,
      viewpoint_cam,
      alpha: torch.Tensor,
      normal: torch.Tensor,
      z_depth: torch.Tensor,
      roughness: torch.Tensor,
      fresnel: torch.Tensor,
      reflect: torch.Tensor,
    ):
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        normal = safe_normalize(normal, dim=0)
        points_flat = depths_to_points(viewpoint_cam, z_depth)
        reflect_d_flat = self.prepare_reflection(viewpoint_cam, normal)
        color_override = RGB2SH(self.get_label).repeat(1, 3)
        outputs = self.reflect_trace(points_flat, reflect_d_flat, alpha.flatten(), color_override)

        indirect_from_object = outputs["color"].reshape(H, W, 3).permute(2, 0, 1)
        ray_alpha = outputs["alpha"].reshape(1, H, W)

        if self.use_screen_filter:
            ray_depth = outputs["depth"].reshape(1, H, W)
            view_depth = z_depths_to_ray_depths(viewpoint_cam, z_depth)
            ray_depth = process_depth(ray_depth, ray_alpha).detach()
            view_depth = process_depth(view_depth, alpha).detach()

            indirect_from_object = self.screen_filtering(
                indirect_from_object,
                normal,
                roughness,
                ray_depth,
                1 / view_depth,
            )

        NoL = dot(normal, reflect_d_flat.reshape(H, W, 3).permute(2, 0, 1), dim=0)
        reflectance = fresnel + (1 - fresnel) * torch.pow(1 - NoL, 5.0)

        return indirect_from_object * reflectance.mean(dim=0, keepdim=True) * (1 - reflect)


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._diffuse = torch.empty(0)
        self._fresnel = torch.empty(0)
        self._roughness = torch.empty(0)
        self._reflect = torch.empty(0)
        self._label = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._diffuse,
            self._fresnel,
            self._roughness,
            self._reflect,
            self._label,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self._diffuse,
        self._fresnel,
        self._roughness,
        self._reflect,
        self._label,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
   
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    def removal_off(self):
        self.removal = Removal.off
        self.mask3d = None

    def removal_object(self):
        self.removal = Removal.remove_object
        self.mask3d = self.get_mask3d().float().unsqueeze(1)

    def removal_background(self):
        self.removal = Removal.remove_background
        self.mask3d = self.get_mask3d().float().unsqueeze(1)

    @property
    def get_opacity(self):
        mask3d = 1.0
        if self.removal != Removal.off:
            if self.removal == Removal.remove_object:
                mask3d = 1 - self.mask3d
            elif self.removal == Removal.remove_background:
                mask3d = self.mask3d

        return self.opacity_activation(self._opacity) * mask3d

    def get_mask3d(self):
        points = self.get_xyz.clone().detach()
        mask3d = (self.get_label > 0.3).flatten()
        if mask3d.any():
            mask3d = ~torch.logical_or(~mask3d, points.isnan().any(dim=1))
            mask3d_convex = points_inside_convex_hull(points, mask3d, remove_outliers=True, outlier_factor=1.0)
            mask3d = torch.logical_or(mask3d, mask3d_convex)

        mask3d = ~torch.logical_or(~mask3d, points.isnan().any(dim=1))
        return mask3d

    @property
    def get_diffuse(self):
        return torch.sigmoid(self._diffuse)
    
    @property
    def get_fresnel(self):
        return torch.sigmoid(self._fresnel)
    
    @property
    def get_roughness(self):
        return torch.sigmoid(self._roughness)

    @property
    def get_reflect(self):
        return torch.sigmoid(self._reflect)
    
    @property
    def get_label(self):
        return torch.sigmoid(self._label)
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.get_opacity
        # apply 3D filter
        scales = self.get_scaling

        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)

        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    @property
    def get_scaling_n_opacity_with_3D_filter(self):
        opacity = self.get_opacity
        scales = self.get_scaling
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        scales = torch.sqrt(scales_after_square)
        return scales, opacity * coef[..., None]

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @torch.no_grad()
    def reset_3D_filter(self):
        xyz = self.get_xyz
        self.filter_3D = torch.zeros([xyz.shape[0], 1], device=xyz.device)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        xyz = self.get_xyz
        distance = torch.full((xyz.shape[0],), torch.inf, device=xyz.device)
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)

        # we should use the focal length of the highest resolution camera
        focal_length = 0.0
        for camera in cameras:
            W, H = camera.image_width, camera.image_height

            # transform points to camera space
            R = torch.from_numpy(camera.R).float().to(xyz.device)
            T = torch.from_numpy(camera.T).float().to(xyz.device)
            # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = torch.addmm(T[None, :], xyz, R)
            z = xyz_cam[:, 2]

            # project to screen space
            valid_depth = z > 0.2

            uv = xyz_cam[:, :2] / z.unsqueeze(-1)
            uv_abs = torch.abs(uv)

            boundry_x = camera.image_width / camera.Fx * 0.575
            boundry_y = camera.image_height / camera.Fy * 0.575
            in_screen = torch.logical_and(uv_abs[:, 0] <= boundry_x, uv_abs[:, 1] <= boundry_y)

            valid = torch.logical_and(valid_depth, in_screen)

            distance = torch.where(valid, torch.minimum(distance, z), distance)
            valid_points = torch.logical_or(valid_points, valid)
            focal_length = max(focal_length, camera.Fx, camera.Fy)

        distance[~valid_points] = distance[valid_points].max()

        filter_3D = distance / focal_length * (0.2**0.5)
        self.filter_3D = filter_3D[..., None]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        if type(pcd) is BasicPointCloud:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd._xyz)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd._rgb)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.detach().clone().float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        diffuse = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        fresnel = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda"))
        roughness = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        reflect = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        label = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._diffuse = nn.Parameter(diffuse.requires_grad_(True))
        self._fresnel = nn.Parameter(fresnel.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._reflect = nn.Parameter(reflect.requires_grad_(True))
        self._label = nn.Parameter(label.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._diffuse], 'lr': training_args.material_lr, "name": "diffuse"},
            {'params': [self._fresnel], 'lr': training_args.material_lr, "name": "fresnel"},
            {'params': [self._roughness], 'lr': training_args.material_lr, "name": "roughness"},
            {'params': [self._reflect], 'lr': training_args.reflect_lr, "name": "reflect"},
            {'params': [self._label], 'lr': training_args.label_lr, "name": "label"},
            {'params': self.env_light.parameters(), 'lr': training_args.env_lr_init, "name": "env_net"},
            {'params': self.roughness_net.parameters(), 'lr': training_args.neural_lr_init, "name": "roughness_net"},
        ]

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps
        )

        self.neural_scheduler_args = get_expon_lr_func(
            lr_init=training_args.neural_lr_init,
            lr_final=training_args.neural_lr_final,
            lr_delay_steps=training_args.neural_lr_delay_steps,
            lr_delay_mult=training_args.neural_lr_delay_mult,
            start_steps=training_args.neural_lr_start_steps,
            max_steps=training_args.neural_lr_max_steps
        )
        
        self.env_scheduler_args = get_expon_lr_func(
            lr_init=training_args.env_lr_init,
            lr_final=training_args.env_lr_final,
            max_steps=training_args.env_lr_max_steps
        )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if hasattr(self, "xyz_scheduler_args"):
                if param_group["name"] == "xyz":
                    lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = lr
            
            if hasattr(self, "neural_scheduler_args"):
                if param_group["name"] == "roughness_net":
                    lr = self.neural_scheduler_args(iteration)
                    param_group['lr'] = lr
            
            if hasattr(self, "env_scheduler_args"):
                if param_group["name"] == "env_net":
                    lr = self.env_scheduler_args(iteration)
                    param_group['lr'] = lr

    def construct_list_of_attributes(self, exclude_filter=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._diffuse.shape[1]):
            l.append('diffuse_{}'.format(i))
        for i in range(self._fresnel.shape[1]):
            l.append('fresnel_{}'.format(i))
        l.append('roughness')
        l.append('reflect')
        l.append('label')
        if not exclude_filter:
            l.append('filter_3D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        diffuse = self._diffuse.detach().cpu().numpy()
        fresnel = self._fresnel.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        reflect = self._reflect.detach().cpu().numpy()
        label = self._label.detach().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, diffuse, fresnel, roughness, reflect, label, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        save_path = path.replace('.ply', '.pth')
        torch.save({
            "env_net": self.env_light.state_dict(),
            "roughness_net": self.roughness_net.state_dict(),
            }, save_path)

    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)

        # apply 3D filter
        scales = self.get_scaling

        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)

        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        diffuse = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        diffuse[:, 0] = np.asarray(plydata.elements[0]["diffuse_0"])
        diffuse[:, 1] = np.asarray(plydata.elements[0]["diffuse_1"])
        diffuse[:, 2] = np.asarray(plydata.elements[0]["diffuse_2"])
        fresnel = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        fresnel[:, 0] = np.asarray(plydata.elements[0]["fresnel_0"])
        fresnel[:, 1] = np.asarray(plydata.elements[0]["fresnel_1"])
        fresnel[:, 2] = np.asarray(plydata.elements[0]["fresnel_2"])
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        reflect = np.asarray(plydata.elements[0]["reflect"])[..., np.newaxis]
        label = np.asarray(plydata.elements[0]["label"])[..., np.newaxis]

        filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

        self._diffuse = nn.Parameter(torch.tensor(diffuse, dtype=torch.float, device="cuda").requires_grad_(True))
        self._fresnel = nn.Parameter(torch.tensor(fresnel, dtype=torch.float, device="cuda").requires_grad_(True))
        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._reflect = nn.Parameter(torch.tensor(reflect, dtype=torch.float, device="cuda").requires_grad_(True))
        self._label = nn.Parameter(torch.tensor(label, dtype=torch.float, device="cuda").requires_grad_(True))

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

        net_path = path.replace('.ply', '.pth')
        if os.path.exists(net_path):
            net_ckpt = torch.load(net_path)
            env_ckpt = net_ckpt["env_net"]
            self.env_light = EnvironmentLight(env_ckpt['base'].shape[1]).cuda()
            self.env_light.load_state_dict(env_ckpt)
            self.roughness_net.load_state_dict(net_ckpt["roughness_net"])
            print("Restoring from ", net_path)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "net" in group["name"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "net" in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._diffuse = optimizable_tensors["diffuse"]
        self._fresnel = optimizable_tensors["fresnel"]
        self._roughness = optimizable_tensors["roughness"]
        self._reflect = optimizable_tensors["reflect"]
        self._label = optimizable_tensors["label"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "net" in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_diffuse, new_fresnel, new_roughness, new_reflect, new_label):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "diffuse": new_diffuse,
        "fresnel": new_fresnel,
        "roughness": new_roughness,
        "reflect": new_reflect,
        "label": new_label}
        extension_num = new_xyz.shape[0]
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._diffuse = optimizable_tensors["diffuse"]
        self._fresnel = optimizable_tensors["fresnel"]
        self._roughness = optimizable_tensors["roughness"]
        self._reflect = optimizable_tensors["reflect"]
        self._label = optimizable_tensors["label"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # self.max_radii2D = torch.cat([self.max_radii2D,torch.zeros(extension_num, device="cuda")])

    def densify_and_split(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_diffuse = self._diffuse[selected_pts_mask].repeat(N,1)
        new_fresnel = self._fresnel[selected_pts_mask].repeat(N,1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        new_reflect = self._reflect[selected_pts_mask].repeat(N,1)
        new_label = self._label[selected_pts_mask].repeat(N,1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_diffuse, new_fresnel, new_roughness, new_reflect, new_label)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        # new_opacities = 1-torch.sqrt(1-self.get_opacity[selected_pts_mask]*0.5)
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_diffuse = self._diffuse[selected_pts_mask]
        new_fresnel = self._fresnel[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_reflect = self._reflect[selected_pts_mask]
        new_label = self._label[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_diffuse, new_fresnel, new_roughness, new_reflect, new_label)

    # use the same densification strategy as GOF https://github.com/autonomousvision/gaussian-opacity-fields
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]

        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]
        torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune
    
    def densify(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]

        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        torch.cuda.empty_cache()
        return clone - before, split - clone, split

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def inpaint_setup(self, training_args, mask3d, _init_points=None):
        init_points = None
        if _init_points is not None:
            init_points = _init_points.clone().detach()

        def initialize_new_features(features, num_new_points, mask_xyz_values, init_points, k=5):
            """Initialize new points for multiple features based on neighbouring points in the remaining area."""
            new_features = {}
            
            if num_new_points == 0:
                for key in features:
                    new_features[key] = torch.empty((0, *features[key].shape[1:]), device=features[key].device)
                return new_features

            # Get remaining points from features
            remaining_xyz_values = features["xyz"]
            _remaining_xyz_values = torch.nan_to_num(remaining_xyz_values, nan=10000.0)
            remaining_xyz_values_np = _remaining_xyz_values.cpu().numpy()
            
            from scipy.spatial import KDTree

            # Build a KD-Tree for fast nearest-neighbor lookup
            kdtree = KDTree(remaining_xyz_values_np)
            
            # Sample random points from mask_xyz_values as query points
            _mask_xyz_values = torch.nan_to_num(mask_xyz_values, nan=10000.0)
            mask_xyz_values_np = _mask_xyz_values.cpu().numpy()
            query_points = mask_xyz_values_np

            if init_points is not None:
                query_points = init_points.cpu().numpy()

            # Find the k nearest neighbors in the remaining points for each query point
            distances, indices = kdtree.query(query_points, k=k)
            selected_indices = indices

            # Initialize new points for each feature
            for key, feature in features.items():
                # Convert feature to numpy array
                feature_np = feature.cpu().numpy()
                
                # If we have valid neighbors, calculate the mean of neighbor points
                if feature_np.ndim == 2:
                    neighbor_points = feature_np[selected_indices]
                elif feature_np.ndim == 3:
                    neighbor_points = feature_np[selected_indices, :, :]
                else:
                    raise ValueError(f"Unsupported feature dimension: {feature_np.ndim}")
                new_points_np = np.mean(neighbor_points, axis=1)
                
                # Convert back to tensor
                new_features[key] = torch.tensor(new_points_np, device=feature.device, dtype=feature.dtype)

            new_features['xyz'] = torch.tensor(query_points, device=new_features['xyz'].device, dtype=new_features['xyz'].dtype)
            
            return new_features
        
        mask3d = ~mask3d.bool().squeeze()
        mask_xyz_values = self._xyz[~mask3d]

        # Extracting subsets using the mask
        xyz_sub = self._xyz[mask3d].detach()
        features_dc_sub = self._features_dc[mask3d].detach()
        features_rest_sub = self._features_rest[mask3d].detach()
        opacity_sub = self._opacity[mask3d].detach()
        scaling_sub = self._scaling[mask3d].detach()
        rotation_sub = self._rotation[mask3d].detach()
        diffuse_sub = self._diffuse[mask3d].detach()
        fresnel_sub = self._fresnel[mask3d].detach()
        roughness_sub = self._roughness[mask3d].detach()
        reflect_sub = self._reflect[mask3d].detach()

        # Add new points with random initialization
        sub_features = {
            'xyz': xyz_sub,
            'features_dc': features_dc_sub,
            'scaling': scaling_sub,
            'features_rest': features_rest_sub,
            'opacity': opacity_sub,
            'rotation': rotation_sub,
            'diffuse': diffuse_sub,
            'fresnel': fresnel_sub,
            'roughness': roughness_sub,
            'reflect': reflect_sub,
        }

        num_new_points = len(mask_xyz_values)
        with torch.no_grad():
            new_features = initialize_new_features(sub_features, num_new_points, mask_xyz_values, init_points)
            new_xyz = new_features['xyz']
            new_scaling = new_features['scaling']
            new_rotation = new_features['rotation']

            new_opacity = inverse_sigmoid(torch.full_like(new_features["opacity"], fill_value=0.01))
            new_features_dc = torch.zeros_like(new_features['features_dc'])
            new_features_rest = torch.zeros_like(new_features['features_rest'])
            new_diffuse = torch.zeros_like(new_features['diffuse'])
            new_fresnel = torch.zeros_like(new_features['fresnel'])
            new_roughness = torch.zeros_like(new_features['roughness'])
            new_reflect = torch.zeros_like(new_features['reflect'])

        self._xyz = nn.Parameter(torch.cat([xyz_sub, new_xyz]))
        self._features_dc = nn.Parameter(torch.cat([features_dc_sub, new_features_dc]))
        self._features_rest = nn.Parameter(torch.cat([features_rest_sub, new_features_rest]))
        self._opacity = nn.Parameter(torch.cat([opacity_sub, new_opacity]))
        self._scaling = nn.Parameter(torch.cat([scaling_sub, new_scaling]))
        self._rotation = nn.Parameter(torch.cat([rotation_sub, new_rotation]))
        self._diffuse = nn.Parameter(torch.cat([diffuse_sub, new_diffuse]))
        self._fresnel = nn.Parameter(torch.cat([fresnel_sub, new_fresnel]))
        self._roughness = nn.Parameter(torch.cat([roughness_sub, new_roughness]))
        self._reflect = nn.Parameter(torch.cat([reflect_sub, new_reflect]))
        self._label = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 1), device="cuda"))

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        print("Spatial lr scale: ", self.spatial_lr_scale)
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_final * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._diffuse], 'lr': training_args.material_lr, "name": "diffuse"},
            {'params': [self._fresnel], 'lr': training_args.material_lr, "name": "fresnel"},
            {'params': [self._roughness], 'lr': training_args.material_lr, "name": "roughness"},
            {'params': [self._reflect], 'lr': training_args.reflect_lr, "name": "reflect"},
            {'params': [self._label], 'lr': training_args.label_lr, "name": "label"},
        ]

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    max_steps=training_args.position_lr_max_steps)

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

