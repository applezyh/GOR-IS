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

import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x, eps=1e-6):
    if isinstance(x, torch.Tensor):
        _x = x.clamp(eps, 1 - eps)
        return torch.log(_x/(1-_x))
    elif isinstance(x, np.ndarray):
        x = np.clip(x, eps, 1 - eps)
        return np.log(x/(1-x))
    elif isinstance(x, (float, int)):
        x = max(min(x, 1 - eps), eps)
        return np.log(x/(1-x))
    else:
        raise ValueError("Unsupported type for inverse_sigmoid: {}".format(type(x)))

def inverse_softplus(x):
    if isinstance(x, torch.Tensor):
        return torch.log(torch.exp(x)-1)
    elif isinstance(x, (float, int, np.ndarray)):
        return np.log(np.exp(x)-1)
    else:
        raise ValueError("Unsupported type for inverse_softplus: {}".format(type(x)))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, start_steps=0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip((step - start_steps) / (lr_delay_steps - start_steps), 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip((step - start_steps) / (max_steps - start_steps), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def sample_points_from_gaussian(mu, scale, rotation, N, sigma=3.0):
    """
    mu: (M, 3,)
    q: quaternion, (M, 4)
    s: (M, 3)
    N: int
    """
    M = mu.shape[0]
    z = (sigma * (2 * torch.rand(M, N, 3, device=mu.device) - 1)).reshape(M*N, 3) # (MN, 3)

    L = build_scaling_rotation(scale, rotation)[:, None].repeat(1, N, 1, 1).reshape(M*N, 3, 3) # (MN, 3, 3)
    x_local = torch.matmul(z.reshape(M*N, 1, 3), L.transpose(1, 2)).reshape(M, N, 3) # (M, N, 3)

    x_world = mu.unsqueeze(1) + x_local  # (M, N, 3)

    return x_world

def dot(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sum(x*y, dim=dim, keepdim=True)

def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-20) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return x / norm

def exp_decay(step, start_step, end_step, value_start, value_end):
    if value_start == value_end:
        return value_start
    elif step < start_step:
        return value_start
    elif step > end_step:
        return value_end
    else:
        ratio = (step - start_step) / (end_step - start_step)
        lr = value_start * (value_end / value_start) ** ratio
        return lr

def cos_decay(step, start_step, end_step, value_start, value_end):
    if value_start == value_end:
        return value_start
    elif step < start_step:
        return value_start
    elif step > end_step:
        return value_end
    else:
        ratio = (step - start_step) / (end_step - start_step)
        lr = value_end + 0.5 * (value_start - value_end) * (1 + np.cos(np.pi * ratio))
        return lr
    
def linear_decay(step, start_step, end_step, value_start, value_end):
    if value_start == value_end:
        return value_start
    elif step < start_step:
        return value_start
    elif step > end_step:
        return value_end
    else:
        ratio = (step - start_step) / (end_step - start_step)
        lr = value_start + ratio * (value_end - value_start)
        return lr

def safe_state(silent, seed):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))

def colormap(img, cmap='jet'):
    import matplotlib.pyplot as plt
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    return img
