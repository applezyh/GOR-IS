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

import json
import os
import tempfile
from argparse import ArgumentParser

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr as psnr_func
from utils.loss_utils import ssim as ssim_func

lpips_func = LPIPS("alex").cuda()

def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]

    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

def evaluate(args):
    model_paths = args.model_paths
    print("Scenes: ", model_paths)

    metrics = {"ssim": 0.0, "psnr": 0.0, "lpips": 0.0, "masked_psnr": 0.0, "masked_ssim": 0.0, "masked_lpips": 0.0}

    collected_gt_images = []
    collected_pred_images = []

    collected_masked_gt_images = []
    collected_masked_pred_images = []

    for scene_path in tqdm(model_paths):
        scene_metrics = {key: 0.0 for key in metrics.keys()}

        test_dir = os.path.join(scene_path, "test")
        results = sorted(os.listdir(test_dir))[-1]

        gt_dir = os.path.join(test_dir, results, "gt")
        gt_mask_dir = os.path.join(test_dir, results, "gt_mask")
        pred_dir = os.path.join(test_dir, results, "renders")

        gt_images = sorted([name for name in os.listdir(gt_dir) if name.endswith(".png") or name.endswith(".jpg")])
        gt_masks = sorted([name for name in os.listdir(gt_mask_dir) if name.endswith(".png") or name.endswith(".jpg")])
        pred_images = sorted([name for name in os.listdir(pred_dir) if name.endswith(".png") or name.endswith(".jpg")])

        count = 0
        masked_count = 0

        for gt_img_name, gt_masks, pred_img_name in tqdm(zip(gt_images, gt_masks, pred_images)):
            gt_img_path = os.path.join(gt_dir, gt_img_name)
            pred_img_path = os.path.join(pred_dir, pred_img_name)

            gt_img = Image.open(gt_img_path).convert("RGB")
            gt_mask = Image.open(os.path.join(gt_mask_dir, gt_masks)).convert("L")
            pred_img = Image.open(pred_img_path).convert("RGB")

            if args.fid:
                collected_gt_images.append(gt_img)
                collected_pred_images.append(pred_img)

            gt_tensor = tf.to_tensor(gt_img).unsqueeze(0).cuda()
            pred_tensor = tf.to_tensor(pred_img).unsqueeze(0).cuda()
            gt_mask = tf.to_tensor(gt_mask)[0].cuda()

            if args.cropping:
                padding = 100
                gt_tensor = gt_tensor[:, :, padding:-padding, padding:-padding]
                pred_tensor = pred_tensor[:, :, padding:-padding, padding:-padding]
                gt_mask = gt_mask[padding:-padding, padding:-padding]

            psnr = psnr_func(pred_tensor, gt_tensor).item()
            ssim = ssim_func(pred_tensor, gt_tensor).item()
            lpips = lpips_func(pred_tensor, gt_tensor).item()

            scene_metrics["psnr"] += psnr
            scene_metrics["ssim"] += ssim
            scene_metrics["lpips"] += lpips
            count += 1

            if gt_mask.sum() == 0:
                continue

            bbox = mask_to_bbox(gt_mask)
            masked_gt = crop_using_bbox(gt_tensor[0], bbox).unsqueeze(0)
            masked_pred = crop_using_bbox(pred_tensor[0], bbox).unsqueeze(0)

            if masked_gt.shape[2] < 32 or masked_gt.shape[3] < 32:
                continue

            if args.fid:
                collected_masked_gt_images.append(tf.to_pil_image(masked_gt[0]))
                collected_masked_pred_images.append(tf.to_pil_image(masked_pred[0]))

            masked_psnr = psnr_func(masked_pred, masked_gt).item()
            masked_ssim = ssim_func(masked_pred, masked_gt).item()
            masked_lpips = lpips_func(masked_pred, masked_gt).item()

            scene_metrics["masked_psnr"] += masked_psnr
            scene_metrics["masked_ssim"] += masked_ssim
            scene_metrics["masked_lpips"] += masked_lpips
            masked_count += 1

        scene_metrics["psnr"] /= count
        scene_metrics["ssim"] /= count
        scene_metrics["lpips"] /= count
        scene_metrics["masked_psnr"] /= masked_count
        scene_metrics["masked_ssim"] /= masked_count
        scene_metrics["masked_lpips"] /= masked_count

        with open(os.path.join(scene_path, "per_scene_metrics.json"), "w") as f:
            json.dump(scene_metrics, f, indent=4)

        metrics["psnr"] += scene_metrics["psnr"]
        metrics["ssim"] += scene_metrics["ssim"]
        metrics["lpips"] += scene_metrics["lpips"]
        metrics["masked_psnr"] += scene_metrics["masked_psnr"]
        metrics["masked_ssim"] += scene_metrics["masked_ssim"]
        metrics["masked_lpips"] += scene_metrics["masked_lpips"]

    total_count = len(model_paths)
    metrics['psnr'] /= total_count
    metrics['ssim'] /= total_count
    metrics['lpips'] /= total_count
    metrics['masked_psnr'] /= total_count
    metrics['masked_ssim'] /= total_count
    metrics['masked_lpips'] /= total_count

    print("*" * 100)
    print(f"  PSNR: {metrics['psnr']}")
    print(f"  SSIM: {metrics['ssim']}")
    print(f"  LPIPS: {metrics['lpips']}")
    print(f"  Masked PSNR: {metrics['masked_psnr']}")
    print(f"  Masked SSIM: {metrics['masked_ssim']}")
    print(f"  Masked LPIPS: {metrics['masked_lpips']}")

    if args.fid:
        metrics.update({"fid": 0.0, "masked_fid": 0.0})
        with tempfile.TemporaryDirectory() as tmpdir:
            from pytorch_fid.fid_score import calculate_fid_given_paths
            tmp_gt_images = os.path.join(tmpdir, "tmp_gt_images")
            tmp_pred_images = os.path.join(tmpdir, "tmp_pred_images")
            os.makedirs(tmp_gt_images)
            os.makedirs(tmp_pred_images)

            for i, (gt_img, pred_img) in enumerate(zip(collected_gt_images, collected_pred_images)):
                gt_img.save(os.path.join(tmp_gt_images, f"image_{i}.png"))
                pred_img.save(os.path.join(tmp_pred_images, f"image_{i}.png"))

            tmp_masked_gt_images = os.path.join(tmpdir, "tmp_masked_gt_images")
            tmp_masked_pred_images = os.path.join(tmpdir, "tmp_masked_pred_images")
            os.makedirs(tmp_masked_gt_images)
            os.makedirs(tmp_masked_pred_images)

            for i, (gt_img, pred_img) in enumerate(zip(collected_masked_gt_images, collected_masked_pred_images)):
                gt_img.save(os.path.join(tmp_masked_gt_images, f"image_{i}.png"))
                pred_img.save(os.path.join(tmp_masked_pred_images, f"image_{i}.png"))

            fid_value = calculate_fid_given_paths([tmp_gt_images, tmp_pred_images],
                                          1,
                                          "cuda",
                                          2048,
                                          8)

            masked_fid_value = calculate_fid_given_paths([tmp_masked_gt_images, tmp_masked_pred_images],
                                          1,
                                          "cuda",
                                          2048,
                                          8)

            metrics["fid"] = fid_value
            metrics["masked_fid"] = masked_fid_value

            print(f"  FID: {metrics['fid']}")
            print(f"  Masked FID: {metrics['masked_fid']}")
    print("*" * 100)

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--fid', '-f', action='store_true', help='Enable FID computation')
    parser.add_argument('--cropping', '-c', action='store_true', help='Enable cropping')
    args = parser.parse_args()
    evaluate(args)