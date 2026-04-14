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

import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim, layer_num):
        super(SimpleNet, self).__init__()
        # Simple Convolutional Network
        self.input_layer = nn.Conv2d(in_dim, latent_dim, kernel_size=1)
        self.output_layer = nn.Conv2d(latent_dim, out_dim, kernel_size=1)

        self.middle_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for _ in range(layer_num):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                )
            )

        for _ in range(layer_num):
            self.middle_blocks.append(
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=5, padding=2),
                    nn.ReLU(),
                )
            )

        for _ in range(layer_num):
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                )
            )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        if H % 8 != 0 or W % 8 != 0:
            # resieze to the nearest multiple of 8
            new_H = (H // 8 + 1) * 8
            new_W = (W // 8 + 1) * 8
            x = nn.functional.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)

        out = self.input_layer(x)
        feats = []
        for down in self.down_blocks:
            out = down(out)
            feats.append(out)
        for middle in self.middle_blocks:
            out = middle(out)
        for up in self.up_blocks:
            out = up(out + feats.pop())

        out = torch.sigmoid(self.output_layer(out))
        out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
