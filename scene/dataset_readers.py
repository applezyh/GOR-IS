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
import sys
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    image_name: str
    image_path: str
    width: int
    height: int

    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array

    image: np.array
    normal: np.array
    obj_removal: np.array

    obj_mask: np.array
    spec_mask: np.array
    inpainting_mask: np.array

    inpainted_image: np.array
    inpainted_fresnel: np.array
    inpainted_diffuse: np.array
    inpainted_roughness: np.array
    inpainted_reflect: np.array
    inpainted_normal: np.array
    inpainted_depth: np.array

    object_effect: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(args, cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        extr_name = extr.name
        image_path = os.path.join(images_folder, extr_name)
        assert os.path.exists(image_path), f"Image file {extr_name} not found in {images_folder}."

        image = Image.open(image_path)
        path = os.path.dirname(images_folder)
        image_name = os.path.basename(extr_name).split(".")[0]

        obj_mask = None
        for ext in [".png", ".jpg", ".JPG", ".jpeg"]:
            obj_mask_path = os.path.join(path, args.object_mask, image_name + ext)
            if os.path.exists(obj_mask_path):
                obj_mask = Image.open(obj_mask_path)
                break

        obj_removal = None
        for ext in [".png", ".jpg", ".JPG", ".jpeg"]:
            obj_removal_path = os.path.join(path, args.object_removal, image_name + ext)
            if os.path.exists(obj_removal_path):
                obj_removal = Image.open(obj_removal_path)
                break

        spec_mask = None
        for ext in [".png", ".jpg", ".JPG", ".jpeg"]:
            spec_mask_path = os.path.join(path, args.specular_mask, image_name + ext)
            if os.path.exists(spec_mask_path):
                spec_mask = Image.open(spec_mask_path)
                break

        normal_path = os.path.join(path, args.normal, image_name + ".npy")
        normal = np.load(normal_path) if os.path.exists(normal_path) else None

        inpainted_list = [
            "image",
            "reflect", 
            "normal", 
            "depth", 
            "fresnel", 
            "diffuse", 
            "roughness",
        ]
        inpainting_mask = None
        inpainted_data = {f"inpainted_{attr}": None for attr in inpainted_list}
        object_effect = None
        if args.load_inpainted:
            # Inpainting data
            loaded_iter = args.inpainting_loaded_iter
            inpainting_path = os.path.join(args.model_path, f"train/ours_{loaded_iter}/inpainting")

            inpainting_mask_path = os.path.join(inpainting_path, f"mask_{image_name}.png")
            inpainting_mask = Image.open(inpainting_mask_path) if os.path.exists(inpainting_mask_path) else None
            for attr in inpainted_list:
                suffix = ".png" if attr != "depth" else ".tiff"
                data_path = os.path.join(inpainting_path, f"{attr}_{image_name}{suffix}")
                if attr != "depth":
                    inpainted_data[f"inpainted_{attr}"] = Image.open(data_path) if os.path.exists(data_path) else None
                else:
                    inpainted_data[f"inpainted_{attr}"] = cv2.imread(data_path, cv2.IMREAD_UNCHANGED) if os.path.exists(data_path) else None

            object_effect_path = os.path.join(inpainting_path, f"object_effect_{image_name}.png")
            object_effect = Image.open(object_effect_path) if os.path.exists(object_effect_path) else None

        cam_infos.append(
            CameraInfo(
                # --- Basic info ---
                uid=idx, image_name=image_name, image_path=image_path,
                # --- Camera parameters ---
                R=R, T=T, FovX=FovX, FovY=FovY, width=image.size[0], height=image.size[1],
                # --- Input / GT ---
                image=image, normal=normal, obj_removal=obj_removal,
                # --- Masks ---
                obj_mask=obj_mask, spec_mask=spec_mask, inpainting_mask=inpainting_mask,
                # --- Inpainting results ---
                **inpainted_data,
                # --- Object-related ---
                object_effect=object_effect,
            )
        )
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(args, path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(args, cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if "spin" in path: # For the SPIN-NeRF dataset, the first 40 images for testing and the rest for training
            train_cam_infos = cam_infos[40:]
            test_cam_infos = cam_infos[:40]
        elif os.path.exists(os.path.join(path, "test_list.txt")):
            with open(os.path.join(path, "test_list.txt")) as f:
                readed_test_list = f.readlines()
            test_list = [x.split(".")[0] for x in readed_test_list]

            with open(os.path.join(path, "train_list.txt")) as f:
                readed_train_list = f.readlines()
            train_list = [x.split(".")[0] for x in readed_train_list]

            train_cam_infos = [c for c in cam_infos if c.image_name in train_list]
            test_cam_infos = [c for c in cam_infos if c.image_name in test_list]
        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(args, path, transformsfile, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            base_path = os.path.dirname(image_path)
            image_name = os.path.basename(base_path)
            image = Image.open(image_path)

            obj_mask = None
            for ext in [".png", ".jpg", ".JPG", ".jpeg"]:
                obj_mask_path = os.path.join(base_path, "attributes", args.object_mask, image_name + ext)
                if os.path.exists(obj_mask_path):
                    obj_mask = Image.open(obj_mask_path)
                    break

            obj_removal = None
            for ext in [".png", ".jpg", ".JPG", ".jpeg"]:
                obj_removal_path = os.path.join(base_path, "attributes", args.object_removal, image_name + ext)
                if os.path.exists(obj_removal_path):
                    obj_removal = Image.open(obj_removal_path)
                    break

            spec_mask = None
            for ext in [".png", ".jpg", ".JPG", ".jpeg"]:
                spec_mask_path = os.path.join(base_path, "attributes", args.specular_mask, image_name + ext)
                if os.path.exists(spec_mask_path):
                    spec_mask = Image.open(spec_mask_path)
                    break

            normal_path = os.path.join(base_path, "attributes", args.normal, image_name + ".npy")
            normal = np.load(normal_path) if os.path.exists(normal_path) else None

            inpainted_list = [
                "image",
                "reflect", 
                "normal", 
                "depth", 
                "fresnel", 
                "diffuse", 
                "roughness",
            ]
            inpainting_mask = None
            inpainted_data = {f"inpainted_{attr}": None for attr in inpainted_list}
            object_effect = None
            if args.load_inpainted:
                # Inpainting data
                loaded_iter = args.inpainting_loaded_iter
                inpainting_path = os.path.join(args.model_path, f"train/ours_{loaded_iter}/inpainting")

                inpainting_mask_path = os.path.join(inpainting_path, f"mask_{image_name}.png")
                inpainting_mask = Image.open(inpainting_mask_path) if os.path.exists(inpainting_mask_path) else None
                for attr in inpainted_list:
                    suffix = ".png" if attr != "depth" else ".tiff"
                    data_path = os.path.join(inpainting_path, f"{attr}_{image_name}{suffix}")
                    if attr != "depth":
                        inpainted_data[f"inpainted_{attr}"] = Image.open(data_path) if os.path.exists(data_path) else None
                    else:
                        inpainted_data[f"inpainted_{attr}"] = cv2.imread(data_path, cv2.IMREAD_UNCHANGED) if os.path.exists(data_path) else None

                object_effect_path = os.path.join(inpainting_path, f"object_effect_{image_name}.png")
                object_effect = Image.open(object_effect_path) if os.path.exists(object_effect_path) else None

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    # --- Basic info ---
                    uid=idx, image_name=image_name, image_path=image_path,
                    # --- Camera parameters ---
                    R=R, T=T, FovX=FovX, FovY=FovY, width=image.size[0], height=image.size[1],
                    # --- Input / GT ---
                    image=image, normal=normal, obj_removal=obj_removal,
                    # --- Masks ---
                    obj_mask=obj_mask, spec_mask=spec_mask, inpainting_mask=inpainting_mask,
                    # --- Inpainting results ---
                    **inpainted_data,
                    # --- Object-related ---
                    object_effect=object_effect,
                )
            )
            
    return cam_infos

def readNerfSyntheticInfo(args, path, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(args, path, "transforms_train.json", extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(args, path, "transforms_test.json", extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
