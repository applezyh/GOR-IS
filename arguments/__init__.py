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
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.object_mask = "object_mask"
        self.object_removal = "object_removal"

        self.load_inpainted = False
        self.inpainting_loaded_iter = 30_000

        self.normal = "normal"
        self.specular_mask = "specular_mask"
        self._resolution = -1
        self._white_background = False
        self.random_background = False
        self.data_device = "cuda"
        self.eval = False
        self.disable_filter3D = False
        self.kernel_size = 0.0 # Size of 2D filter in mip-splatting
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_max_steps = 30_000

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.material_lr = 0.0025
        self.reflect_lr = 0.005
        self.label_lr = 0.01

        self.env_lr_init = 0.01
        self.env_lr_final = 0.001
        self.env_lr_max_steps = 30_000

        self.neural_lr_init = 0.00001
        self.neural_lr_final = 0.00001
        self.neural_lr_delay_mult = 0.1
        self.neural_lr_delay_steps = 10_000
        self.neural_lr_start_steps = 4000
        self.neural_lr_max_steps = 30_000

        self.lambda_dssim = 0.2
        self.lambda_normal = 0.05
        self.lambda_ref_normal_init = 0.5
        self.lambda_ref_normal_end = 0.001
        self.lambda_smooth = 0.05
        self.lambda_label = 1.0
        self.lambda_reflect = 1.0

        self.geo_reg_from_iter = 7_000
        self.label_from_iter = 0
        
        self.blend_from_iter = 4000
        self.ref_normal_decay_end = 10_000

        self.percent_dense = 0.01
        self.opacity_cull = 0.05

        self.densification_interval = 500
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 12_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
