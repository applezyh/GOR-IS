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

import multiprocessing
import os
import subprocess
import time
from argparse import ArgumentParser


def run_command(command, gpu_id, work_name, stage, logs_dir):
    log_dir = os.path.join(logs_dir, work_name)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{stage}_gpu{gpu_id}_{timestamp}.log")

    print(f"[GPU {gpu_id}] Running: {command}")
    print(f"[GPU {gpu_id}] Log: {log_file}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TORCH_CUDA_ARCH_LIST"] = "8.6"
    env["GAMMA"] = "1.5"

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            command,
            shell=True,
            env=env,
            stdout=f,
            stderr=f
        )
        ret = process.wait()

    if ret != 0:
        raise RuntimeError(f"[GPU {gpu_id}] Failed: {command}")

    print(f"[GPU {gpu_id}] Done.")


def worker(args, gpu_id, data_paths):
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for data_path in data_paths:
        work_name = os.path.basename(data_path)
        output_dir = os.path.join(args.output_dir, work_name)

        RES = 512 # Synthetic scenes
        if any(x in work_name for x in ["scene_9", "scene_10", "garden", "spheres"]):
            RES = 4 # Real-world scenes

        common_args = f"-s {data_path} -i images --random_background --resolution {RES} --depth_ratio 0.0"
        common_train_args = f"{common_args} --eval --port {gpu_id + 7001} --logger {args.logger}"

        try:
            if args.recon:
                cmd = f"python train.py {common_train_args} -m {output_dir} --iterations 30000"
                run_command(cmd, gpu_id, work_name, "recon", logs_dir)

            if args.remove_object:
                cmd = f"python render.py {common_args} --iteration 30000 -m {output_dir} --use_pbr --skip_mesh --skip_test --skip_train --remove_object"
                run_command(cmd, gpu_id, work_name, "remove_object", logs_dir)

            if args.eval_recon:
                cmd = f"python metrics.py -m {output_dir}"
                run_command(cmd, gpu_id, work_name, "eval_recon", logs_dir)

            if args.inpainting2D:
                cmd = f"python inpainting2D.py --eval --iteration 30000 -s {data_path} -m {output_dir} -i images --resolution {RES}"
                run_command(cmd, gpu_id, work_name, "inpainting2D", logs_dir)

            if args.inpainting3D:
                cmd = f"python inpainting3D.py {common_train_args} -m {output_dir} --iterations 34000 --load_inpainted --use_material_inpainting --use_reflection_mask --densify_until_iter 32000 --densify_from_iter 30500"
                run_command(cmd, gpu_id, work_name, "inpainting3D", logs_dir)

            if args.render_inpainting3D:
                cmd = f"python render.py {common_args} --iteration 34000 -m {output_dir} --use_pbr --skip_mesh --skip_train --is_removal"
                run_command(cmd, gpu_id, work_name, "render_inpainting3D", logs_dir)

            if args.eval_inpainting3D:
                cmd = f"python metrics.py -m {output_dir}"
                run_command(cmd, gpu_id, work_name, "eval_inpainting3D", logs_dir)

            if args.baking:
                baking_dir = output_dir + "_baking"
                os.makedirs(baking_dir, exist_ok=True)

                subprocess.run(
                    f"cp -r {os.path.join(output_dir, 'point_cloud')} {baking_dir}",
                    shell=True,
                    check=True
                )

                cmd = f"python baking.py {common_train_args} -m {baking_dir} --iterations 30000 --densification_interval 100 --densify_until_iter 15000"
                run_command(cmd, gpu_id, work_name, "baking", logs_dir)

            if args.eval_baking:
                cmd = f"python render.py {common_args} --iteration 30000 -m {output_dir}_baking --skip_mesh --skip_train"
                run_command(cmd, gpu_id, work_name, "eval_baking", logs_dir)

        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR on {work_name}: {e}")
            raise


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--recon", action='store_true')
    parser.add_argument("--remove_object", action='store_true')
    parser.add_argument("--eval_recon", action='store_true')
    parser.add_argument("--mesh", action='store_true')

    parser.add_argument("--inpainting2D", action='store_true')
    parser.add_argument("--inpainting3D", action='store_true')
    parser.add_argument("--render_inpainting3D", action='store_true')
    parser.add_argument("--eval_inpainting3D", action='store_true')

    parser.add_argument("--baking", action='store_true')
    parser.add_argument("--eval_baking", action='store_true')

    parser.add_argument("--data_list", nargs="+", type=str, default=[])
    parser.add_argument("--device_list", nargs="+", type=int, default=[])
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb", "both", "none"])
    args = parser.parse_args()

    root = args.root_dir
    data_list = args.data_list

    # GPU list
    devices = args.device_list
    if len(devices) == 0:
        import torch
        devices = list(range(torch.cuda.device_count()))

    num_devices = len(devices)

    data_path_pool = {gpu: [] for gpu in devices}
    for i, data in enumerate(data_list):
        gpu = devices[i % num_devices]
        data_path_pool[gpu].append(os.path.join(root, data))

    num_workers = min(num_devices, len(data_list))

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = []
        for gpu in devices:
            if len(data_path_pool[gpu]) == 0:
                continue
            results.append(
                pool.apply_async(worker, (args, gpu, data_path_pool[gpu]))
            )

        for r in results:
            r.get()

    print("All processes are done.")