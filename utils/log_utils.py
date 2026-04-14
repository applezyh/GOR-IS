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

try:
    import wandb
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass

class Logger:
    def __init__(self, args):
        self.use_loggder = True
        if args.logger == "none":
            self.use_loggder = False
            return
        wandb_args = {'project': 'glossy-gaussian-splatting', 'entity': 'apple-lab'}
        tensorboard_args = {'log_dir': args.model_path}
        self.use_tensorboard = args.logger in ["tensorboard", "both"]
        self.use_wandb = args.logger in ["wandb", "both"]

        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(**tensorboard_args)

        if self.use_wandb:
            import os
            os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
            self.wandb_writer = wandb.init(**wandb_args, config=args)

    def log(self, tag, value, step):
        if not self.use_loggder:
            return
        if self.use_tensorboard:
            self.tb_writer.add_scalar(tag, value, step)
        if self.use_wandb:
            self.wandb_writer.log({tag: value}, step=step)

    def log_image(self, tag, img, step):
        if not self.use_loggder:
            return
        if self.use_tensorboard:
            self.tb_writer.add_image(tag, img.cpu(), step)
        if self.use_wandb:
            _img = img.permute(1, 2, 0).detach().cpu().numpy()
            self.wandb_writer.log({tag: wandb.Image(_img)}, step=step)

    def log_histogram(self, tag, values, step):
        if not self.use_loggder:
            return
        if self.use_tensorboard:
            self.tb_writer.add_histogram(tag, values, step)
        if self.use_wandb:
            _values = values.detach().cpu().numpy()
            self.wandb_writer.log({tag: wandb.Histogram(_values)}, step=step)

    def close(self):
        if not self.use_loggder:
            return
        if self.use_tensorboard:
            self.tb_writer.close()
        if self.use_wandb:
            self.wandb_writer.finish()