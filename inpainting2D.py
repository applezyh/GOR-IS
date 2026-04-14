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
from itertools import combinations

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos
from utils.image_utils import cross_view_consistency
from utils.point_utils import depths_to_points, points_inside_point_convex_hull

device = "cuda"

def median_filter(input, kernel_size=5):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    # Ensure 4D tensor [B, C, H, W]
    if input.dim() == 3:
        x = input.unsqueeze(0)  # [1, C, H, W]
    B, C, H, W = x.shape

    padding = kernel_size // 2
    x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode='reflect')  # [B, C, H+2p, W+2p]

    # Unfold to extract sliding windows
    patches = x_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # [B, C, H, W, k, k]
    patches = patches.contiguous().view(B, C, H, W, -1)  # [B, C, H, W, k*k]

    median = patches.median(dim=-1)[0]  # [B, C, H, W]

    # Remove batch dimension if input didn't have it
    if input.dim() == 3:
        return median[0]
    else:
        return median

def load_scene(args, source_path, images, eval):
    if os.path.exists(os.path.join(source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args, source_path, images, eval)
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args, source_path, False, eval)
    else:
        assert False, "Could not recognize scene type!"

    return scene_info

def init_inpainting(inpainting_model, device):
    assert inpainting_model in ["lama", "sd"], f"Invalid inpainting model {inpainting_model}. Supported options are 'lama' or 'sd'."

    if inpainting_model == "lama":
        from simple_lama_inpainting import SimpleLama
        simple_lama = SimpleLama(device=device)

        def inpainting_func(image, mask, ret_original=False):
            output = simple_lama(image, mask)
            if ret_original:
                return np.array(output) / 255.0
            return output

    elif inpainting_model == "sd":
        from diffusers import StableDiffusionInpaintPipeline
        from diffusers.utils import load_image
        pipe = StableDiffusionInpaintPipeline.from_pretrained("sd-legacy/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16").to(device)

        def inpainting_func(image, mask, ret_original=False):
            image = load_image(image)
            mask_image = load_image(mask)

            output = pipe(
                prompt="big blurry hole",
                image=image,
                mask_image=mask_image,
                guidance_scale=10.0,
                num_inference_steps=20,  # steps between 15 and 30 work well for us
                strength=0.99,  # make sure to use `strength` below 1.0
                generator=torch.Generator(device=device).manual_seed(0),
            ).images[0].resize(image.size)

            return output[1] if ret_original else output

    return inpainting_func

if __name__ == "__main__":
    from argparse import ArgumentParser

    from arguments import ModelParams
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--inpainting_model", type=str, default="lama", help="Inpainting model to use")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration to load")
    parser.add_argument("--store_data", action='store_true')
    model = ModelParams(parser)
    args = parser.parse_args(sys.argv[1:])
    dataset = model.extract(args)

    removal_path = os.path.join(args.model_path, f"train/ours_{args.iteration}/removal")
    inpainting_path = os.path.join(args.model_path, f"train/ours_{args.iteration}/inpainting")
    os.makedirs(inpainting_path, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(
        os.path.join(
            args.model_path, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply"
        )
    )
    print("extracting object points ...")
    mask3d = gaussians.get_mask3d()
    object_points = gaussians.get_xyz[mask3d].detach().cpu().numpy()

    scene_info = load_scene(dataset, args.source_path, images=args.images, eval=args.eval)
    train_cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)

    inpainting_func = init_inpainting(args.inpainting_model, device=device)

    for i, camera in enumerate(tqdm(train_cameras, desc="Inpainting")):
        # Load files
        inpainting_mask = Image.open(os.path.join(removal_path, f"mask_{i:05d}.png"))
        depth = cv2.imread(os.path.join(removal_path, f"depth_{i:05d}.tiff"), cv2.IMREAD_UNCHANGED)

        d_min, d_max = depth.min(), depth.max()
        range_val = d_max - d_min + 1e-8
        depth_norm = ((depth - d_min) / range_val * 255).astype(np.uint8)
        depth_rgb = Image.fromarray(depth_norm).convert("RGB")

        depth_inpaint = inpainting_func(depth_rgb, inpainting_mask, ret_original=True)

        depth_rescaled = (depth_inpaint * range_val + d_min).mean(axis=-1)
        depth_t = torch.from_numpy(depth_rescaled).to(device).unsqueeze(0)
        mask_t = TF.to_tensor(inpainting_mask).to(device)

        h, w = camera.image_height, camera.image_width
        depth_resized = TF.resize(depth_t, (h, w))[0]
        mask_resized = TF.resize(mask_t, (h, w))[0]

        points = depths_to_points(camera, depth_resized)

        inside_mask = points_inside_point_convex_hull(points.cpu().numpy(), object_points, expand_ratio=1.5)
        inside_mask_t = torch.from_numpy(inside_mask).to(device).float().view_as(mask_resized).float()
        inpainting_mask_t = median_filter(inside_mask_t[None])
        inpainting_mask = Image.fromarray((inpainting_mask_t[0].cpu().numpy() * 255).astype(np.uint8))
        inpainting_mask.save(os.path.join(inpainting_path, f"mask_{camera.image_name}.png"))

        vis_depth = (depth_inpaint * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(inpainting_path, f"depth_vis_{camera.image_name}.png"), vis_depth)

        depth_inpaint = (depth_inpaint * range_val + d_min).mean(axis=-1)
        cv2.imwrite(os.path.join(inpainting_path, f"depth_{camera.image_name}.tiff"), depth_inpaint)

        for inpainting_item in ["image", "diffuse", "fresnel", "roughness", "reflect", "normal"]:
            image = Image.open(os.path.join(removal_path, f"{inpainting_item}_{i:05d}.png")).convert("RGB")
            inpainting_func(image, inpainting_mask).save(os.path.join(inpainting_path, f"{inpainting_item}_{camera.image_name}.png"))

        object_effect_path = os.path.join(removal_path, f"object_effect_{i:05d}.png")
        os.system(f"cp {object_effect_path} {os.path.join(inpainting_path, f'object_effect_{camera.image_name}.png')}")

    candidate_views = []
    top_k_masks = []
    for i, camera in enumerate(tqdm(train_cameras, desc="Top k inpainting masks")):
        inpainting_mask = TF.to_tensor(Image.open(os.path.join(inpainting_path, f"mask_{camera.image_name}.png")).convert("L")).to(device)
        top_k_masks.append(inpainting_mask.sum().item())

    top_k = 10
    top_k_indices = torch.topk(torch.tensor(top_k_masks), k=top_k).indices
    candidate_views = [train_cameras[i] for i in top_k_indices]

    # ---- Preload all data once ----
    view_data = []

    for i, camera in enumerate(candidate_views):
        image_width, image_height = camera.image_width, camera.image_height

        mask = TF.to_tensor(
            Image.open(os.path.join(inpainting_path, f"mask_{camera.image_name}.png")).convert("L")
        ).to(device)

        image = TF.to_tensor(
            Image.open(os.path.join(inpainting_path, f"image_{camera.image_name}.png")).convert("RGB")
        ).to(device)

        depth = torch.tensor(
            cv2.imread(
                os.path.join(inpainting_path, f"depth_{camera.image_name}.tiff"),
                cv2.IMREAD_UNCHANGED
            ),
            device=device
        ).float()[None]

        # Resize once
        mask = TF.resize(mask, (image_height, image_width))
        image = TF.resize(image, (image_height, image_width))
        depth = TF.resize(depth, (image_height, image_width))

        view_data.append((mask, image, depth, camera))

    score_matrix = torch.zeros((top_k, top_k), device=device)
    num_views = len(candidate_views)
    # ---- Compute pairwise consistency ----
    for i, j in combinations(range(num_views), 2):
        mask_i, img_i, depth_i, cam_i = view_data[i]
        mask_j, img_j, depth_j, cam_j = view_data[j]

        consistency_i2j = cross_view_consistency(
            img_i, img_j, depth_i, depth_j,
            cam_i, cam_j,
            M=mask_i
        )

        consistency_j2i = cross_view_consistency(
            img_j, img_i, depth_j, depth_i,
            cam_j, cam_i,
            M=mask_j
        )

        c2w_i = (cam_i.world_view_transform.T).inverse()
        c2w_j = (cam_j.world_view_transform.T).inverse()

        # View direction
        dir_i = torch.nn.functional.normalize(c2w_i[:3, 2], dim=0)
        dir_j = torch.nn.functional.normalize(c2w_j[:3, 2], dim=0)

        # Calculate cosine similarity
        weight = (torch.dot(dir_i, dir_j) + 1) / 2

        score = (consistency_i2j + consistency_j2i) * (1 - weight)
        score_matrix[i, j] = score
        score_matrix[j, i] = score  

    best_score = -float("inf")
    REFERENCE_VIEW_NUM = 3
    best_group = (0,) * REFERENCE_VIEW_NUM

    # All N view combinations
    if REFERENCE_VIEW_NUM > 1:
        for view_group in combinations(range(num_views), REFERENCE_VIEW_NUM):
            total_score = sum(score_matrix[i, j] for i, j in combinations(view_group, 2))
            if total_score > best_score:
                best_score = total_score
                best_group = view_group

    if os.path.exists(os.path.join(args.model_path, "reference_view")):
        os.system(f"rm -r {os.path.join(args.model_path, 'reference_view')}")
    os.makedirs(os.path.join(args.model_path, "reference_view"), exist_ok=True)
    for idx in best_group:
        camera = candidate_views[idx]
        image_name = camera.image_name + ".png"
        # Save in model_path/reference_view
        os.system(f"cp {os.path.join(inpainting_path, f'image_{image_name}')} {os.path.join(args.model_path, 'reference_view', image_name)}")