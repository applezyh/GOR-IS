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
from functools import partial

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from scene import GaussianModel
from utils.image_utils import mapping, th_dilate
from utils.render_utils import save_img_f32, save_img_u8


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

class GaussianExtractor(object):
    def __init__(self, gaussians: GaussianModel, render, pipe, bg_color=None, kernel_size=0.0):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background, kernel_size=kernel_size)

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.fresnelmaps = []
        self.roughnessmaps = []
        self.diffusemaps = []
        self.specularmaps = []
        self.normals = []
        self.depth_normals = []
        self.labelmaps = []
        self.reflectmaps = []
        self.object_effects = []

        self.removed_rgbmaps = []
        self.removed_fresnelmaps = []
        self.removed_roughnessmaps = []
        self.removed_diffusemaps = []
        self.removed_reflectmaps = []
        self.removed_normals = []
        self.removed_depthmaps = []

        self.viewpoint_stack = []

    @torch.no_grad()
    def remove(self, viewpoint_stack, use_pbr):
        self.viewpoint_stack = viewpoint_stack
        self.gaussians.removal_object()
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            self.removed_rgbmaps.append(render_pkg['render'].cpu())
            self.removed_depthmaps.append(render_pkg['surf_depth'].cpu())
            if use_pbr:
                alpha = render_pkg['rend_alpha']
                normal = render_pkg['rend_normal']
                world_normal = render_pkg['rend_normal_w']
                depth = render_pkg['surf_depth']
                diffuse = render_pkg['rend_diffuse']
                fresnel = render_pkg['rend_fresnel']
                roughness = render_pkg['rend_roughness']
                reflect = render_pkg['rend_reflect']
                outputs = self.gaussians.pbr(
                    viewpoint_cam, alpha, normal, depth, diffuse, fresnel, roughness
                )
                render_color = outputs["render_color"] * (1 - reflect) + reflect * render_pkg['render']
                render_color = torch.clamp(mapping(render_color), 0.0, 1.0)

                self.removed_rgbmaps[-1] = render_color.cpu()
                self.removed_fresnelmaps.append(fresnel.cpu())
                self.removed_roughnessmaps.append(roughness.cpu())
                self.removed_diffusemaps.append(diffuse.cpu())
                self.removed_reflectmaps.append(reflect.cpu())
                self.removed_normals.append(world_normal.cpu())
        
        self.gaussians.removal_off()

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, use_pbr=True):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = render_pkg['rend_normal']
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            label = render_pkg['rend_label']
            rgb = torch.clamp(mapping(rgb), 0.0, 1.0)
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
            self.labelmaps.append(label.cpu())

            if use_pbr:
                diffuse = render_pkg['rend_diffuse']
                fresnel = render_pkg['rend_fresnel']
                roughness = render_pkg['rend_roughness']
                reflect = render_pkg['rend_reflect']
                outputs = self.gaussians.pbr(
                    viewpoint_cam, alpha, normal, depth, diffuse, fresnel, roughness
                )
                object_effect = self.gaussians.object_effect(
                    viewpoint_cam, alpha, normal, depth, roughness, fresnel, reflect
                )
                render_color = outputs["render_color"] * (1 - reflect) + reflect * render_pkg['render']

                render_color = torch.clamp(mapping(render_color), 0.0, 1.0)
                diffuse = torch.clamp(mapping(diffuse * (1 - reflect) + reflect * render_pkg['render']), 0.0, 1.0)
                specular = torch.clamp(mapping(outputs["specular"]) * (1 - reflect), 0.0, 1.0)

                self.rgbmaps[-1] = render_color.cpu()
                self.fresnelmaps.append(fresnel.cpu())
                self.roughnessmaps.append(roughness.cpu())
                self.diffusemaps.append(diffuse.cpu())
                self.specularmaps.append(specular.cpu())
                self.reflectmaps.append(reflect.cpu())
                self.object_effects.append(object_effect.cpu())

        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            label = self.labelmaps[i]

            rgb = rgb * (1 - label)
            depth = depth * (1 - label)
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path, is_removal=False):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        gt_masks_path = os.path.join(path, "gt_mask")
        vis_path = os.path.join(path, "vis")
        pbr_path = os.path.join(path, "pbr")
        removal_path = os.path.join(path, "removal")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            gt_mask = viewpoint_cam.obj_mask
            if is_removal:
                try:
                    gt = viewpoint_cam.obj_removal[0:3, :, :]
                except:
                    pass
            if gt_mask is not None:
                os.makedirs(gt_masks_path, exist_ok=True)
                save_img_u8(gt_mask[0].cpu().numpy(), os.path.join(gt_masks_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))

            if len(self.roughnessmaps) > 0:
                os.makedirs(pbr_path, exist_ok=True)
                save_img_u8(self.roughnessmaps[idx][0].cpu().numpy(), os.path.join(pbr_path, 'roughness_{0:05d}'.format(idx) + ".png")) 

            if len(self.fresnelmaps) > 0:
                os.makedirs(pbr_path, exist_ok=True)
                save_img_u8(self.fresnelmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(pbr_path, 'fresnel_{0:05d}'.format(idx) + ".png"))

            if len(self.diffusemaps) > 0:
                os.makedirs(pbr_path, exist_ok=True)
                save_img_u8(self.diffusemaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(pbr_path, 'diffuse_{0:05d}'.format(idx) + ".png"))

            if len(self.reflectmaps) > 0:
                os.makedirs(pbr_path, exist_ok=True)
                save_img_u8(self.reflectmaps[idx][0].cpu().numpy(), os.path.join(pbr_path, 'reflect_{0:05d}'.format(idx) + ".png"))

            if len(self.specularmaps) > 0:
                os.makedirs(pbr_path, exist_ok=True)
                save_img_u8(self.specularmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(pbr_path, 'specular_{0:05d}'.format(idx) + ".png"))

            if len(self.labelmaps) > 0:
                os.makedirs(removal_path, exist_ok=True)
                save_img_u8(self.labelmaps[idx][0].cpu().numpy(), os.path.join(removal_path, 'label_{0:05d}'.format(idx) + ".png"))

            if len(self.object_effects) > 0:
                os.makedirs(removal_path, exist_ok=True)
                save_img_u8(self.object_effects[idx].permute(1,2,0).cpu().numpy(), os.path.join(removal_path, 'object_effect_{0:05d}'.format(idx) + ".png"))

            if len(self.removed_rgbmaps) > 0:
                os.makedirs(removal_path, exist_ok=True)
                save_img_u8(self.removed_rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(removal_path, 'image_{0:05d}'.format(idx) + ".png"))

            if len(self.removed_reflectmaps) > 0:
                os.makedirs(removal_path, exist_ok=True)
                save_img_u8(self.removed_reflectmaps[idx][0].cpu().numpy(), os.path.join(removal_path, 'reflect_{0:05d}'.format(idx) + ".png"))
            
            if len(self.removed_diffusemaps) > 0:
                os.makedirs(removal_path, exist_ok=True)
                save_img_u8(self.removed_diffusemaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(removal_path, 'diffuse_{0:05d}'.format(idx) + ".png"))

            if len(self.removed_roughnessmaps) > 0:
                os.makedirs(removal_path, exist_ok=True)
                save_img_u8(self.removed_roughnessmaps[idx][0].cpu().numpy(), os.path.join(removal_path, 'roughness_{0:05d}'.format(idx) + ".png"))

            if len(self.removed_fresnelmaps) > 0:
                os.makedirs(removal_path, exist_ok=True)
                save_img_u8(self.removed_fresnelmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(removal_path, 'fresnel_{0:05d}'.format(idx) + ".png"))

            if len(self.removed_normals) > 0:
                os.makedirs(removal_path, exist_ok=True)
                save_img_u8(self.removed_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(removal_path, 'normal_{0:05d}'.format(idx) + ".png"))

            if len(self.removed_depthmaps) > 0:
                os.makedirs(removal_path, exist_ok=True)
                save_img_f32(self.removed_depthmaps[idx][0].cpu().numpy(), os.path.join(removal_path, 'depth_{0:05d}'.format(idx) + ".tiff"))

                os.makedirs(removal_path, exist_ok=True)
                inpainting_mask = th_dilate(gt_mask.float(), iterations=2)
                save_img_u8(inpainting_mask[0].cpu().numpy(), os.path.join(removal_path, 'mask_{0:05d}'.format(idx) + ".png"))

