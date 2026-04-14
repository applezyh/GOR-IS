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
import numpy as np
import torch

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    gt_normal = None
    if cam_info.normal is not None:
        resized_normal = torch.from_numpy(cv2.resize(cam_info.normal, resolution, interpolation=cv2.INTER_LINEAR)).float()
        resized_normal = torch.nn.functional.normalize(resized_normal, dim=-1)
        resized_normal[..., 1:] *= -1  # flip y and z axis
        W2C = torch.from_numpy(cam_info.R.transpose()).float()
        C2W = torch.linalg.inv(W2C)
        gt_normal = (resized_normal @ C2W.T).permute(2, 0, 1)

    gt_obj_removal = None
    if cam_info.obj_removal is not None:
        if len(cam_info.obj_removal.split()) > 3:
            resized_obj_removal = torch.cat([PILtoTorch(im, resolution) for im in cam_info.obj_removal.split()[:3]], dim=0)
            obj_removal_mask = PILtoTorch(cam_info.obj_removal.split()[3], resolution)
            gt_obj_removal = resized_obj_removal * obj_removal_mask
        else:
            gt_obj_removal = PILtoTorch(cam_info.obj_removal, resolution)

    gt_obj_mask = None
    if cam_info.obj_mask is not None:
        gt_obj_mask = PILtoTorch(cam_info.obj_mask, resolution)[:1]

    gt_spec_mask = None
    if cam_info.spec_mask is not None:
        gt_spec_mask = PILtoTorch(cam_info.spec_mask, resolution)[:1]

    # Inpainting related
    inpainting_mask = None
    if cam_info.inpainting_mask is not None:
        inpainting_mask = PILtoTorch(cam_info.inpainting_mask, resolution)[:1]
    
    inpainted_image = None
    if cam_info.inpainted_image is not None:
        inpainted_image = PILtoTorch(cam_info.inpainted_image, resolution)

    inpainted_reflect = None
    if cam_info.inpainted_reflect is not None:
        inpainted_reflect = PILtoTorch(cam_info.inpainted_reflect, resolution)[:1]

    inpainted_fresnel = None
    if cam_info.inpainted_fresnel is not None:
        inpainted_fresnel = PILtoTorch(cam_info.inpainted_fresnel, resolution)

    inpainted_diffuse = None
    if cam_info.inpainted_diffuse is not None:
        inpainted_diffuse = PILtoTorch(cam_info.inpainted_diffuse, resolution)

    inpainted_roughness = None
    if cam_info.inpainted_roughness is not None:
        inpainted_roughness = PILtoTorch(cam_info.inpainted_roughness, resolution)[:1]

    object_effect = None
    if cam_info.object_effect is not None:
        object_effect = PILtoTorch(cam_info.object_effect, resolution)[:1]

    inpainted_normal = None
    if cam_info.inpainted_normal is not None:
        inpainted_normal = PILtoTorch(cam_info.inpainted_normal, resolution)
        inpainted_normal = 2 * inpainted_normal - 1.0  # to [-1, 1]
        inpainted_normal = torch.nn.functional.normalize(inpainted_normal, dim=0)
        W2C = torch.from_numpy(cam_info.R.transpose()).float()
        C2W = torch.linalg.inv(W2C)
        inpainted_normal = (inpainted_normal.permute(1, 2, 0) @ C2W.T).permute(2, 0, 1)

    inpainted_depth = None
    if cam_info.inpainted_depth is not None:
        inpainted_depth = cv2.resize(cam_info.inpainted_depth, resolution, interpolation=cv2.INTER_LINEAR)
        inpainted_depth = torch.from_numpy(inpainted_depth).float()
        if len(inpainted_depth.shape) == 2:
            inpainted_depth = inpainted_depth.unsqueeze(0)

    return Camera(
        # --- Basic camera info ---
        colmap_id=cam_info.uid,
        uid=id,
        image_name=cam_info.image_name,
        data_device=args.data_device,

        # --- Camera pose & intrinsics ---
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,

        # --- Ground truth ---
        image=gt_image,
        normal=gt_normal,

        # --- Masks ---
        gt_alpha_mask=loaded_mask,
        obj_mask=gt_obj_mask,
        spec_mask=gt_spec_mask,
        obj_removal=gt_obj_removal,
        inpainting_mask=inpainting_mask,

        # --- Inpainting results ---
        inpainted_image=inpainted_image,
        inpainted_depth=inpainted_depth,
        inpainted_normal=inpainted_normal,
        inpainted_reflect=inpainted_reflect,
        inpainted_fresnel=inpainted_fresnel,
        inpainted_diffuse=inpainted_diffuse,
        inpainted_roughness=inpainted_roughness,

        # --- Object-related ---
        object_effect=object_effect,
    )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry