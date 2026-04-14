#
# Copyright (C) 2026, Yonghao Zhao
# Nankai University
#
# This software is developed and maintained by the author.
#
# It is intended for non-commercial use, including research and
# evaluation purposes, under the terms specified in the LICENSE file.
#
# For inquiries, please contact:
# applezyh@outlook.com
#

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import choice, randint

import numpy as np
import torch
import torchvision.transforms as TT
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import (crop_using_bbox, inverse_mapping, mapping,
                               mask_to_bbox, psnr, render_net_image, th_dilate)
from utils.log_utils import Logger
from utils.loss_utils import l1_loss, ssim
from utils.point_utils import depths_to_points, world_points_to_screen


def tensor_inpaint(img, mask, inpaint_model):
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img_in = Image.fromarray((img * 255).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
    mask_in = Image.fromarray((mask * 255).detach().cpu().numpy().astype(np.uint8))

    img_inpaint = inpaint_model(img_in.convert("RGB"), mask_in)
    if img.shape[0] == 3:
        return TT.functional.to_tensor(img_inpaint.resize(img_in.size)).to(img.device)
    else:
        return TT.functional.to_tensor(img_inpaint.resize(img_in.size)).to(img.device)[0:1]

def pad_image(x, size=32):
    _, H, W = x.shape
    pad_H = max(0, size - H)
    pad_W = max(0, size - W)

    pad_top = pad_H // 2
    pad_bottom = pad_H - pad_top
    pad_left = pad_W // 2
    pad_right = pad_W - pad_left
    
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    return torch.nn.functional.pad(x, padding, mode='constant', value=0)

def filter(x, kernel_size=7):
    return TT.GaussianBlur(kernel_size=kernel_size, sigma=5.0)(x.clone())

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, load_iteration, logger):
    first_iter = 0

    from simple_lama_inpainting import SimpleLama
    simple_lama = SimpleLama(device="cuda")
    import lpips
    LPIPS = lpips.LPIPS(net='vgg').cuda()
    LPIPS.eval()

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration)
    first_iter = load_iteration if load_iteration > 0 else first_iter

    ref_view_dir = os.path.join(dataset.model_path, "reference_view")
    ref_view_name = [fn.split(".")[0] for fn in os.listdir(ref_view_dir)]
    print("Reference view for inpainting:", ref_view_name)
    ref_view_camera = [camera for camera in scene.getTrainCameras().copy() if camera.image_name in ref_view_name]

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    base_background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_normal_for_log = 0.0

    print("Updating inpainting points ...")
    with torch.no_grad():
        xyzs = []
        rgbs = []
        materials = []
        for view_cam in scene.getTrainCameras():
            if view_cam.image_name in ref_view_name:
                _depth = view_cam.inpainted_depth.cuda()
                _points = depths_to_points(view_cam, _depth)

                _rgbs = inverse_mapping(view_cam.inpainted_image.cuda().permute(1, 2, 0).reshape(-1, 3))
                _diffuses = view_cam.inpainted_diffuse.cuda().permute(1, 2, 0).reshape(-1, 3)
                _fresnels = view_cam.inpainted_fresnel.cuda().permute(1, 2, 0).reshape(-1, 3)
                _roughnesses = view_cam.inpainted_roughness.cuda().permute(1, 2, 0).reshape(-1, 1)
                _normals = view_cam.inpainted_normal.cuda().permute(1, 2, 0).reshape(-1, 3)
                _reflect = view_cam.inpainted_reflect.cuda().permute(1, 2, 0).reshape(-1, 1)
                _materials = torch.cat([_diffuses, _fresnels, _roughnesses, _normals, _reflect], 1)

                _valid_mask = view_cam.inpainting_mask.cuda().permute(1, 2, 0).reshape(-1) > 0

                xyzs.append(_points[_valid_mask])
                rgbs.append(_rgbs[_valid_mask])
                materials.append(_materials[_valid_mask])
        
        xyzs = torch.cat(xyzs, 0)
        rgbs = torch.cat(rgbs, 0)
        materials = torch.cat(materials, 0)
        points = torch.cat([xyzs, rgbs, materials], 1)

    mask3d = gaussians.get_mask3d()
    gaussians.spatial_lr_scale = scene.cameras_extent
    gaussians.inpaint_setup(opt, mask3d, points[..., :3])

    trainCameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):    

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        background = torch.rand_like(base_background) if dataset.random_background else base_background
        kernel_size = dataset.kernel_size

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        render_radiance, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = inverse_mapping(viewpoint_cam.original_image.cuda())
        inpainted_reflect = viewpoint_cam.inpainted_reflect.cuda()

        if args.use_reflection_mask:
            maintain_mask = 1 - th_dilate(((viewpoint_cam.object_effect + viewpoint_cam.obj_mask) > 0.1).float()).cuda()
        else:
            maintain_mask = 1 - th_dilate(viewpoint_cam.obj_mask.float()).cuda()

        gt_normal = viewpoint_cam.normal.cuda() if viewpoint_cam.normal is not None else None
        gt_mask = viewpoint_cam.gt_alpha_mask.cuda() if viewpoint_cam.gt_alpha_mask is not None else torch.ones_like(gt_image[0:1]).cuda()
        gt_spec_mask = viewpoint_cam.spec_mask.cuda() if viewpoint_cam.spec_mask is not None else torch.zeros_like(gt_image[0:1]).cuda()

        gt_image = gt_image + (1 - gt_mask) * background[:, None, None]
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > opt.geo_reg_from_iter else 0.0
        lambda_ref_normal = opt.lambda_ref_normal_end if iteration > opt.blend_from_iter else 0.0
        lambda_reflect = opt.lambda_reflect if iteration > opt.blend_from_iter else 0.0

        outputs = gaussians.pbr(
            viewpoint_cam, 
            render_pkg["rend_alpha"], 
            render_pkg["rend_normal"], 
            render_pkg["surf_depth"], 
            render_pkg["rend_diffuse"], 
            render_pkg["rend_fresnel"], 
            render_pkg["rend_roughness"], 
            background
        )
        render_pkg.update(outputs)

        rend_fresnel = render_pkg["rend_fresnel"]
        rend_diffuse = render_pkg["rend_diffuse"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        rend_reflect = render_pkg["rend_reflect"]
        rend_roughness = render_pkg["rend_roughness"]
        render_color = render_pkg["render_color"] * (1 - rend_reflect) + rend_reflect * render_radiance

        Ll1 = l1_loss(render_color * maintain_mask, gt_image * maintain_mask)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(render_color * maintain_mask, gt_image * maintain_mask))

        ref_normal_loss = torch.tensor(0.0, device="cuda")
        if gt_normal is not None:
            ref_normal_error = (rend_normal - gt_normal).abs()
            ref_normal_loss = lambda_ref_normal * (ref_normal_error * maintain_mask * gt_spec_mask).mean()

        reflect_loss = torch.tensor(0.0, device="cuda")
        if gt_spec_mask is not None:
            reflect_error = gt_spec_mask * rend_reflect
            reflect_loss = lambda_reflect * (reflect_error * maintain_mask).mean()

        normal_loss = (1 - (surf_normal * rend_normal).sum(0, keepdim=True)).mean() * lambda_normal

        total_loss = loss + ref_normal_loss + reflect_loss + normal_loss

        rgb_loss_inpaint = torch.tensor(0.0, device="cuda")
        normal_loss_inpaint = torch.tensor(0.0, device="cuda")
        fresnel_loss_inpaint = torch.tensor(0.0, device="cuda")
        roughness_loss_inpaint = torch.tensor(0.0, device="cuda")
        diffuse_loss_inpaint = torch.tensor(0.0, device="cuda")
        reflect_loss_inpaint = torch.tensor(0.0, device="cuda")

        inpainting_mask = viewpoint_cam.inpainting_mask.cuda()
        glossy_mask = inpainting_mask.clone()
        rough_mask = inpainting_mask.clone()
        if args.use_material_inpainting:
            glossy_mask *= (1 - inpainted_reflect)
            rough_mask *= inpainted_reflect

        if viewpoint_cam.image_name not in ref_view_name:
            uv_coords, z_depths = world_points_to_screen(viewpoint_cam, points[..., :3])
            uv_coords[..., 0] *= viewpoint_cam.image_width - 1
            uv_coords[..., 1] *= viewpoint_cam.image_height - 1

            valid_mask = (uv_coords[..., 0] >= 0) & (uv_coords[..., 0] < viewpoint_cam.image_width) & \
                         (uv_coords[..., 1] >= 0) & (uv_coords[..., 1] < viewpoint_cam.image_height)

            uv_coords[..., 0] = uv_coords[..., 0].clamp(0, viewpoint_cam.image_width - 1)
            uv_coords[..., 1] = uv_coords[..., 1].clamp(0, viewpoint_cam.image_height - 1)

            sampled_rend_depth = render_pkg["surf_depth"][0, uv_coords[..., 1].long(), uv_coords[..., 0].long()]
            valid_mask = torch.logical_and(valid_mask, (z_depths - sampled_rend_depth) < 0.5)[..., None]

            sampled_rgb = render_color
            sampled_radiance = render_radiance
            sampled_normal = rend_normal
            sampled_reflect = rend_reflect
            sampled_diffuse = rend_diffuse
            sampled_fresnel = rend_fresnel
            sampled_roughness = rend_roughness

            warped_rgb = torch.zeros_like(sampled_rgb)
            warped_diffuse = torch.zeros_like(sampled_diffuse)
            warped_fresnel = torch.zeros_like(sampled_fresnel)
            warped_roughness = torch.zeros_like(sampled_roughness)
            warped_normals = torch.zeros_like(sampled_normal)
            warped_reflect = torch.zeros_like(sampled_reflect)
            warped_valid_mask = torch.zeros_like(sampled_rgb[0:1])
            warped_valid_mask[:, uv_coords[..., 1].long(), uv_coords[..., 0].long()] = valid_mask.permute(1, 0).float()

            if warped_valid_mask.any():
                warped_rgb[:, uv_coords[..., 1].long(), uv_coords[..., 0].long()] = points[..., 3:6].permute(1, 0)
                warped_diffuse[:, uv_coords[..., 1].long(), uv_coords[..., 0].long()] = points[..., 6:9].permute(1, 0)
                warped_fresnel[:, uv_coords[..., 1].long(), uv_coords[..., 0].long()] = points[..., 9:12].permute(1, 0)
                warped_roughness[:, uv_coords[..., 1].long(), uv_coords[..., 0].long()] = points[..., 12:13].permute(1, 0)
                warped_normals[:, uv_coords[..., 1].long(), uv_coords[..., 0].long()] = points[..., 13:16].permute(1, 0)
                warped_reflect[:, uv_coords[..., 1].long(), uv_coords[..., 0].long()] = points[..., 16:17].permute(1, 0)

                bbox = mask_to_bbox(warped_valid_mask[0])

                glossy_mask = crop_using_bbox(glossy_mask.clone(), bbox)
                rough_mask = crop_using_bbox(rough_mask.clone(), bbox)

                sampled_rgb = crop_using_bbox(sampled_rgb.clone(), bbox)
                sampled_radiance = crop_using_bbox(sampled_radiance.clone(), bbox)
                sampled_normal = crop_using_bbox(sampled_normal.clone(), bbox)
                sampled_reflect = crop_using_bbox(sampled_reflect.clone(), bbox)
                sampled_diffuse = crop_using_bbox(sampled_diffuse.clone(), bbox)
                sampled_fresnel = crop_using_bbox(sampled_fresnel.clone(), bbox)
                sampled_roughness = crop_using_bbox(sampled_roughness.clone(), bbox)

                warped_rgb = crop_using_bbox(warped_rgb.clone(), bbox)
                warped_normals = crop_using_bbox(warped_normals.clone(), bbox)
                warped_reflect = crop_using_bbox(warped_reflect.clone(), bbox)
                warped_diffuse = crop_using_bbox(warped_diffuse.clone(), bbox)
                warped_fresnel = crop_using_bbox(warped_fresnel.clone(), bbox)
                warped_roughness = crop_using_bbox(warped_roughness.clone(), bbox)
                warped_valid_mask = crop_using_bbox(warped_valid_mask.clone(), bbox)

                process = pad_image
                sampled_rgb = process(sampled_rgb)
                sampled_radiance = process(sampled_radiance)
                sampled_normal = process(sampled_normal)
                sampled_diffuse = process(sampled_diffuse)
                sampled_reflect = process(sampled_reflect)
                sampled_fresnel = process(sampled_fresnel)
                sampled_roughness = process(sampled_roughness)

                warped_rgb = process(warped_rgb)
                warped_diffuse = process(warped_diffuse)
                warped_reflect = process(warped_reflect)
                warped_fresnel = process(warped_fresnel)
                warped_roughness = process(warped_roughness)
                warped_normals = process(warped_normals)
                glossy_mask = process(glossy_mask)
                rough_mask = process(rough_mask)
                warped_valid_mask = process(warped_valid_mask)

                inpainting_mask = 1 - warped_valid_mask
                # Due to the presence of holes in the mapped views, which degrades optimization quality, the 2D inpainting model is employed to complete the missing regions.
                warped_rgb = tensor_inpaint(warped_rgb, inpainting_mask[0], simple_lama)
                warped_diffuse = tensor_inpaint(warped_diffuse, inpainting_mask[0], simple_lama)
                warped_reflect = tensor_inpaint(warped_reflect, inpainting_mask[0], simple_lama)
                warped_fresnel = tensor_inpaint(warped_fresnel, inpainting_mask[0], simple_lama)
                warped_roughness = tensor_inpaint(warped_roughness, inpainting_mask[0], simple_lama)
                warped_normals = tensor_inpaint(warped_normals * 0.5 + 0.5, inpainting_mask[0], simple_lama)
                warped_normals = 2 * warped_normals - 1

                # Ensure the original RGB images while suppressing areas that do not require optimization
                sampled_rgb = sampled_rgb * rough_mask + sampled_rgb.detach() * (1 - rough_mask)
                sampled_radiance = sampled_radiance * rough_mask + sampled_radiance.detach() * (1 - rough_mask)

                # Gaussian filtering to eliminates high-frequency noise.
                sampled_rgb = filter(sampled_rgb)
                sampled_radiance = filter(sampled_radiance)
                warped_rgb = filter(warped_rgb)

                sampled_normal = filter(sampled_normal * glossy_mask)
                sampled_diffuse = filter(sampled_diffuse * glossy_mask)
                sampled_fresnel = filter(sampled_fresnel * glossy_mask)
                sampled_roughness = filter(sampled_roughness * glossy_mask)
                sampled_reflect = filter(sampled_reflect * glossy_mask)

                warped_normals = filter(warped_normals * glossy_mask)
                warped_diffuse = filter(warped_diffuse * glossy_mask)
                warped_fresnel = filter(warped_fresnel * glossy_mask)
                warped_roughness = filter(warped_roughness * glossy_mask)
                warped_reflect = filter(warped_reflect * glossy_mask)

                rgb_loss_inpaint += LPIPS(sampled_rgb[None], warped_rgb[None], normalize=True).mean() * 0.2
                rgb_loss_inpaint += LPIPS(sampled_radiance[None], warped_rgb[None], normalize=True).mean() * 0.2

                if args.use_material_inpainting:
                    normal_loss_inpaint += ((sampled_normal - warped_normals).abs()).mean()
                    diffuse_loss_inpaint += ((sampled_diffuse - warped_diffuse).abs()).mean()
                    fresnel_loss_inpaint += ((sampled_fresnel - warped_fresnel).abs()).mean()
                    roughness_loss_inpaint += ((sampled_roughness - warped_roughness).abs()).mean()
                    reflect_loss_inpaint += ((sampled_reflect - warped_reflect).abs()).mean()
        
        random_camera = choice(ref_view_camera)
        inpainting_mask = random_camera.inpainting_mask.cuda()
        glossy_mask = inpainting_mask.clone()
        rough_mask = inpainting_mask.clone()
        if args.use_material_inpainting:
            glossy_mask *= (1 - inpainted_reflect)
            rough_mask *= inpainted_reflect

        if inpainting_mask.any():
            render_pkg = render(random_camera, gaussians, pipe, background, kernel_size)
            render_radance = render_pkg["render"]
            ref_outputs = gaussians.pbr(
                random_camera, 
                render_pkg["rend_alpha"], 
                render_pkg["rend_normal"], 
                render_pkg["surf_depth"], 
                render_pkg["rend_diffuse"], 
                render_pkg["rend_fresnel"], 
                render_pkg["rend_roughness"], 
                background
            )
            render_pkg.update(ref_outputs)

            rend_fresnel = render_pkg["rend_fresnel"]
            rend_diffuse = render_pkg["rend_diffuse"]
            rend_normal = render_pkg["rend_normal"]
            rend_reflect = render_pkg["rend_reflect"]
            rend_roughness = render_pkg["rend_roughness"]
            render_color = render_pkg["render_color"] * (1 - rend_reflect) + rend_reflect * render_radance

            sampled_rgb = render_color
            sampled_radiance = render_radiance
            sampled_normal = rend_normal
            sampled_reflect = rend_reflect
            sampled_diffuse = rend_diffuse
            sampled_fresnel = rend_fresnel
            sampled_roughness = rend_roughness

            bbox = mask_to_bbox(inpainting_mask[0])
            glossy_mask = crop_using_bbox(glossy_mask.clone(), bbox)
            rough_mask = crop_using_bbox(rough_mask.clone(), bbox)

            sampled_rgb = crop_using_bbox(sampled_rgb.clone(), bbox)
            sampled_radiance = crop_using_bbox(sampled_radiance.clone(), bbox)
            sampled_normal = crop_using_bbox(sampled_normal.clone(), bbox)
            sampled_reflect = crop_using_bbox(sampled_reflect.clone(), bbox)
            sampled_diffuse = crop_using_bbox(sampled_diffuse.clone(), bbox)
            sampled_fresnel = crop_using_bbox(sampled_fresnel.clone(), bbox)
            sampled_roughness = crop_using_bbox(sampled_roughness.clone(), bbox)

            warped_rgb = crop_using_bbox(inverse_mapping(random_camera.inpainted_image).cuda(), bbox)
            warped_normals = crop_using_bbox(random_camera.inpainted_normal.clone().cuda(), bbox)
            warped_reflect = crop_using_bbox(random_camera.inpainted_reflect.clone().cuda(), bbox)
            warped_diffuse = crop_using_bbox(random_camera.inpainted_diffuse.clone().cuda(), bbox)
            warped_fresnel = crop_using_bbox(random_camera.inpainted_fresnel.clone().cuda(), bbox)
            warped_roughness = crop_using_bbox(random_camera.inpainted_roughness.clone().cuda(), bbox)

            process = pad_image
            sampled_rgb = sampled_rgb * rough_mask + sampled_rgb.detach() * (1 - rough_mask)
            sampled_radiance = sampled_radiance * rough_mask + sampled_radiance.detach() * (1 - rough_mask)
            sampled_rgb = process(sampled_rgb)
            sampled_radiance = process(sampled_radiance)
            sampled_normal = process(sampled_normal * glossy_mask)
            sampled_diffuse = process(sampled_diffuse * glossy_mask)
            sampled_reflect = process(sampled_reflect * glossy_mask)
            sampled_fresnel = process(sampled_fresnel * glossy_mask)
            sampled_roughness = process(sampled_roughness * glossy_mask)

            warped_rgb = process(warped_rgb)
            warped_diffuse = process(warped_diffuse * glossy_mask)
            warped_reflect = process(warped_reflect * glossy_mask)
            warped_fresnel = process(warped_fresnel * glossy_mask)
            warped_roughness = process(warped_roughness * glossy_mask)
            warped_normals = process(warped_normals * glossy_mask)

            rgb_loss_inpaint += LPIPS(sampled_rgb[None], warped_rgb[None], normalize=True).mean() * 0.2
            rgb_loss_inpaint += LPIPS(sampled_radiance[None], warped_rgb[None], normalize=True).mean() * 0.2

            if args.use_material_inpainting:
                normal_loss_inpaint += ((sampled_normal - warped_normals).abs()).mean()
                diffuse_loss_inpaint += ((sampled_diffuse - warped_diffuse).abs()).mean()
                fresnel_loss_inpaint += ((sampled_fresnel - warped_fresnel).abs()).mean()
                roughness_loss_inpaint += ((sampled_roughness - warped_roughness).abs()).mean()
                reflect_loss_inpaint += ((sampled_reflect - warped_reflect).abs()).mean()

        total_loss = total_loss + (rgb_loss_inpaint + normal_loss_inpaint + diffuse_loss_inpaint + fresnel_loss_inpaint + roughness_loss_inpaint + reflect_loss_inpaint) * 0.01

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log 

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if logger is not None:
                logger.log('train_loss_patches/main_loss', loss.item(), iteration)
                logger.log('iter_time', iter_start.elapsed_time(iter_end), iteration)
                logger.log('total_points', scene.gaussians.get_xyz.shape[0], iteration)

                logger.log('train_loss_patches/normal_loss', normal_loss, iteration)
                logger.log('train_loss_patches/ref_normal_loss', ref_normal_loss, iteration)
                logger.log('train_loss_patches/reflect_loss', reflect_loss, iteration)

                logger.log('train_loss_patches/rgb_loss_inpaint', rgb_loss_inpaint, iteration)
                logger.log('train_loss_patches/normal_loss_inpaint', normal_loss_inpaint, iteration)
                logger.log('train_loss_patches/diffuse_loss_inpaint', diffuse_loss_inpaint, iteration)
                logger.log('train_loss_patches/fresnel_loss_inpaint', fresnel_loss_inpaint, iteration)
                logger.log('train_loss_patches/roughness_loss_inpaint', roughness_loss_inpaint, iteration)
                logger.log('train_loss_patches/reflect_loss_inpaint', reflect_loss_inpaint, iteration)

            training_report(logger, iteration, testing_iterations, scene, render, (pipe, base_background, kernel_size), iteration == opt.iterations)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration == opt.iterations and logger is not None:
                logger.close()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify(opt.densify_grad_threshold, scene.cameras_extent)
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)
                
            if iteration % 100 == 0 and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args) -> Logger:    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    logger = None
    try:
        logger = Logger(args)
    except Exception as e:
        print(f"Failed to create logger, no logging will be done. Error: {e}")
    return logger

@torch.no_grad()
def training_report(logger: Logger, iteration: int, testing_iterations: int, scene : Scene, renderFunc, renderArgs, last_iteration):
    # Report test and samples of training set
    if iteration % testing_iterations == 0 or last_iteration:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = render_pkg["render"]
                    reflect = render_pkg["rend_reflect"]
                    alpha = render_pkg["rend_alpha"]
                    normal = render_pkg["rend_normal"]
                    depth = render_pkg["surf_depth"]
                    diffuse = render_pkg["rend_diffuse"]
                    fresnel = render_pkg["rend_fresnel"]
                    roughness = render_pkg["rend_roughness"]
                    outputs = scene.gaussians.pbr(
                        viewpoint, alpha, normal, depth, diffuse, fresnel, roughness, renderArgs[1]
                    )
                    render_pkg.update(outputs)
                    render_color = render_pkg["render_color"]
                    image = image * reflect + (1 - reflect) * render_color
                    image = torch.clamp(mapping(image), 0.0, 1.0)
                    try:
                        gt_image = torch.clamp(viewpoint.obj_removal.cuda(), 0.0, 1.0)
                    except:
                        gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    if logger and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        image_name = viewpoint.image_name
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        logger.log_image(config['name'] + f"_view_{image_name}/depth", depth, step=iteration)
                        logger.log_image(config['name'] + f"_view_{image_name}/render", image, step=iteration)
                        try:
                            diffuse = render_pkg["diffuse"]
                            fresnel = render_pkg["rend_fresnel"]
                            roughness = render_pkg["rend_roughness"]
                            s_roughness = render_pkg["screen_roughness"]
                            visibility = render_pkg["visibility"]
                            specular = torch.clamp(mapping(render_pkg["specular"]), 0.0, 1.0)
                            label = render_pkg["rend_label"]
                            logger.log_image(config['name'] + f"_view_{image_name}/diffuse", diffuse, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/specular", specular, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/label", label, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/fresnel", fresnel, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/roughness", roughness, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/s_roughness", s_roughness, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/reflect", reflect, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/visibility", visibility, step=iteration)
                        except Exception as e:
                            pass

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            logger.log_image(config['name'] + f"_view_{image_name}/rend_normal", rend_normal, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/surf_normal", surf_normal, step=iteration)
                            logger.log_image(config['name'] + f"_view_{image_name}/rend_alpha", rend_alpha, step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations:
                            logger.log_image(config['name'] + f"_view_{image_name}/ground_truth", gt_image, step=iteration)

                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: PSNR {}, SSIM {}".format(iteration, config['name'], psnr_test, ssim_test))
                if logger:
                    logger.log(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    logger.log(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--test_iterations", type=int, default=2000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--use_material_inpainting', action='store_true', default=False)
    parser.add_argument('--use_reflection_mask', action='store_true', default=False)
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "both", "none"])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    logger = prepare_output_and_logger(args)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.load_iteration, logger)

    # All done
    print("\nTraining complete.")
