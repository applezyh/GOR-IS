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

import sys
from argparse import ArgumentParser
from random import randint, random

import torch
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render, render_baking
from scene import GaussianModel, Scene
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim


def camera_interpolation(cam1, cam2, alpha=0.5):
    rots = Rot.from_quat([
        Rot.from_matrix(cam1.R).as_quat(),
        Rot.from_matrix(cam2.R).as_quat()
    ])

    R = Slerp([0, 1], rots)([alpha])[0].as_matrix()
    T = (1 - alpha) * cam1.T + alpha * cam2.T

    return Camera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=cam1.FoVx,
        FoVy=cam1.FoVy,
        image=torch.ones_like(cam1.original_image),
        gt_alpha_mask=torch.ones_like(cam1.gt_alpha_mask),
        image_name="",
        uid=0,
    )

def training(dataset, opt, pipe, saving_iterations, checkpoint, load_iteration, src_iteration):
    first_iter = 0
    src_gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, src_gaussians, load_iteration=src_iteration)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        src_gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    base_background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        src_gaussians.reset_3D_filter()
    else:
        src_gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration)
    gaussians.training_setup(opt)
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

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
            viewpoint_stack = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
        viewpoint_cam_1 = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam_2 = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam = camera_interpolation(viewpoint_cam_1, viewpoint_cam_2, alpha=random())

        with torch.no_grad():
            ref_render_pkg = render(viewpoint_cam, src_gaussians, pipe, background, kernel_size)

            rend_reflect = ref_render_pkg["rend_reflect"]
            alpha = ref_render_pkg["rend_alpha"]
            normal = ref_render_pkg["rend_normal"]
            depth = ref_render_pkg["surf_depth"]
            diffuse = ref_render_pkg["rend_diffuse"]
            fresnel = ref_render_pkg["rend_fresnel"]
            roughness = ref_render_pkg["rend_roughness"]
            outputs = src_gaussians.pbr(
                viewpoint_cam, alpha, normal, depth, diffuse, fresnel, roughness, background
            )
            ref_render_pkg.update(outputs)
            render_color = ref_render_pkg["render_color"] * (1 - rend_reflect) + rend_reflect * ref_render_pkg["render"]
        
        render_pkg = render_baking(viewpoint_cam, gaussians, pipe, background, kernel_size)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        Ll1 = l1_loss(image, render_color)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, render_color))
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

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
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--src_iteration", type=int, default=34000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.start_checkpoint, args.load_iteration, args.src_iteration)

    # All done
    print("\nTraining complete.")