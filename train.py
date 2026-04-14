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
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state, linear_decay
from utils.image_utils import inverse_mapping, mapping, psnr, render_net_image
from utils.log_utils import Logger
from utils.loss_utils import (bilateral_smooth_loss, binary_cross_entropy, l1_loss, loss_cls_3d, ssim)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, logger):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    base_background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_normal_for_log = 0.0

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
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = inverse_mapping(viewpoint_cam.original_image.cuda())
        gt_normal = viewpoint_cam.normal.cuda() if viewpoint_cam.normal is not None else None
        gt_mask = viewpoint_cam.gt_alpha_mask.cuda() if viewpoint_cam.gt_alpha_mask is not None else torch.ones_like(gt_image[0:1])
        gt_obj_mask = viewpoint_cam.obj_mask.cuda() if viewpoint_cam.obj_mask is not None else None
        gt_spec_mask = viewpoint_cam.spec_mask.cuda() if viewpoint_cam.spec_mask is not None else torch.zeros_like(gt_image[0:1])

        gt_image = gt_image + (1 - gt_mask) * background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > opt.geo_reg_from_iter else 0.0

        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']

        normal_loss = lambda_normal * (1 - (rend_normal * surf_normal).sum(dim=0)).mean()

        label_loss = torch.tensor(0.0, device="cuda")
        if gt_obj_mask is not None:
            lambda_label = opt.lambda_label if iteration > opt.label_from_iter else 0.0
            rend_label = render_pkg['rend_label']
            label_loss = lambda_label * binary_cross_entropy(rend_label, gt_obj_mask)

            if iteration % 2 == 0 and lambda_label > 0:
                # Forcing neighboring Gaussian labels to converge and suppressing edge noise.
                # regularize at certain intervals
                labels3d = gaussians.get_label # [N, 1]
                points3d = gaussians.get_xyz.clone().detach() # [N, 3]
                label_error_3d = loss_cls_3d(points3d, labels3d, sample_size=1000)
                label_loss = label_loss + lambda_label * label_error_3d

        ref_normal_loss = torch.tensor(0.0, device="cuda")
        smooth_loss = torch.tensor(0.0, device="cuda")
        reflect_loss = torch.tensor(0.0, device="cuda")

        if iteration > opt.blend_from_iter:
            # ref normal loss
            lambda_ref_normal = linear_decay(
                iteration, 
                opt.blend_from_iter, 
                opt.ref_normal_decay_end, 
                opt.lambda_ref_normal_init, 
                opt.lambda_ref_normal_end
            )
            if gt_normal is not None:
                ref_normal_error = (rend_normal - gt_normal).abs()
                ref_normal_loss = lambda_ref_normal * (ref_normal_error * gt_spec_mask).mean()

            alpha = render_pkg["rend_alpha"]
            depth = render_pkg["surf_depth"]
            diffuse = render_pkg["rend_diffuse"]
            fresnel = render_pkg["rend_fresnel"]
            roughness = render_pkg["rend_roughness"]
            reflect = render_pkg["rend_reflect"]

            # smooth loss
            smooth_loss = opt.lambda_smooth * (
                bilateral_smooth_loss(fresnel, gt_image, gt_spec_mask) +
                bilateral_smooth_loss(roughness, gt_image, gt_spec_mask) +
                bilateral_smooth_loss(rend_normal, gt_image, gt_spec_mask) +
                bilateral_smooth_loss(surf_normal, gt_image, gt_spec_mask)
            )

            # reflect loss
            if gt_spec_mask is not None:
                reflect_error = gt_spec_mask * reflect
                reflect_loss = opt.lambda_reflect * reflect_error.mean()

            # pbr shading
            outputs = gaussians.pbr(
                viewpoint_cam, alpha, rend_normal, depth, diffuse, fresnel, roughness, background
            )
            render_pkg.update(outputs)
            render_color = render_pkg["render_color"] * (1 - reflect) + reflect * image
            Ll1 = l1_loss(render_color, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(render_color, gt_image))

        total_loss = loss + normal_loss + label_loss + ref_normal_loss + smooth_loss + reflect_loss
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
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

                logger.log('train_loss_patches/label_loss', label_loss, iteration)
                logger.log('train_loss_patches/smooth_loss', smooth_loss, iteration)
                logger.log('train_loss_patches/reflect_loss', reflect_loss, iteration)

            training_report(logger, iteration, testing_iterations, scene, render, (pipe, base_background, kernel_size))
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
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
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
def training_report(logger: Logger, iteration: int, testing_iterations: int, scene : Scene, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration % testing_iterations == 0:
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
                    rend_normal = render_pkg["rend_normal"]
                    depth = render_pkg["surf_depth"]
                    diffuse = render_pkg["rend_diffuse"]
                    fresnel = render_pkg["rend_fresnel"]
                    roughness = render_pkg["rend_roughness"]
                    outputs = scene.gaussians.pbr(
                        viewpoint, alpha, rend_normal, depth, diffuse, fresnel, roughness, renderArgs[1]
                    )
                    render_pkg.update(outputs)
                    image = render_pkg["render_color"] * (1 - reflect) + reflect * image
                    image = torch.clamp(mapping(image), 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
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
                        except:
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
    parser.add_argument("--test_iterations", type=int, default=2000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
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
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, logger)

    # All done
    print("\nTraining complete.")