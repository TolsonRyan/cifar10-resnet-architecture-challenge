"""Existing crop/flip/RandAugment followed by per-image Cutout.

Cutout concept: DeVries & Taylor, https://arxiv.org/abs/1708.04552.
Applied after normalization: the square is filled with the training mean.
"""
import torch
from round2_augment import TrainTransform as PreviousTransform


class TrainTransform(PreviousTransform):
    def __init__(self, mean, std, config):
        super().__init__(mean, std, config)
        self.cutout_size = config['cutout_size']
        self.cutout_probability = config['cutout_probability']

    def __call__(self, images):
        z = super().__call__(images)
        if not self.cutout_size or not self.cutout_probability:
            return z
        n, _, height, width = z.shape
        center = torch.randint(0, 32, (n, 2), device=z.device)
        lower = center - self.cutout_size // 2
        upper = lower + self.cutout_size
        rows = torch.arange(height, device=z.device)[None, :, None]
        cols = torch.arange(width, device=z.device)[None, None, :]
        mask = ((rows >= lower[:, 0, None, None]) & (rows < upper[:, 0, None, None]) &
                (cols >= lower[:, 1, None, None]) & (cols < upper[:, 1, None, None]))
        mask &= torch.rand(n, 1, 1, device=z.device) < self.cutout_probability
        return z.masked_fill(mask[:, None], 0).contiguous(memory_format=torch.channels_last)
