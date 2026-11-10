"""CIFAR augmentations for round two; torchvision RandAugment policy.

Operations and magnitude definitions come from torchvision 0.21 (BSD-3-Clause).
We use its tensor operations and vectorize the equivalent equalization LUT to
avoid one GPU synchronization per image/channel. Each shuffled group shares
two policy choices; crops, flips, and CutMix/Mixup pairing remain randomized.
References: Cubuk et al., arXiv:1909.13719; Yun et al., arXiv:1905.04899.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import RandAugment, InterpolationMode
from torchvision.transforms.autoaugment import _apply_op


def equalize_batch(x):
    """Exact uint8 histogram equalization, including constant-image handling."""
    flat = x.reshape(-1, x.shape[-2] * x.shape[-1]).long()
    hist = torch.zeros((len(flat), 256), dtype=torch.long, device=x.device)
    hist.scatter_add_(1, flat, torch.ones_like(flat))
    bins = torch.arange(256, device=x.device)[None]
    last = torch.where(hist > 0, bins, 0).amax(1, keepdim=True)
    step = (hist.sum(1, keepdim=True) - hist.gather(1, last)) // 255
    lut = ((hist.cumsum(1) + step // 2) // step.clamp_min(1))
    lut = F.pad(lut[:, :-1], (1, 0)).clamp(0, 255)
    lut = torch.where(step > 0, lut, bins)
    return lut.gather(1, flat).reshape_as(x).to(torch.uint8)


class GroupRandAugment(RandAugment):
    def __init__(self, magnitude, group_size, fill):
        super().__init__(2, magnitude, interpolation=InterpolationMode.BILINEAR, fill=fill)
        self.group_size = group_size
        self.space = self._augmentation_space(31, (32, 32))

    def forward(self, images):
        chunks = []
        for chunk in images.split(self.group_size):
            for _ in range(self.num_ops):
                name = list(self.space)[int(torch.randint(len(self.space), ()).item())]
                magnitudes, signed = self.space[name]
                magnitude = float(magnitudes[self.magnitude]) if magnitudes.ndim else 0.0
                if signed and bool(torch.randint(2, ()).item()):
                    magnitude *= -1
                chunk = equalize_batch(chunk) if name == 'Equalize' else _apply_op(
                    chunk, name, magnitude, interpolation=self.interpolation, fill=self.fill)
            chunks.append(chunk)
        return torch.cat(chunks)


class TrainTransform:
    def __init__(self, mean, std, config):
        self.mean = torch.tensor(mean, device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor(std, device='cuda').view(1, 3, 1, 1)
        self.grid = torch.arange(32, device='cuda')
        self.ra = GroupRandAugment(config['ra_magnitude'], config['ra_group_size'],
                                   [round(v * 255) for v in mean]) if config['ra_magnitude'] else None

    def __call__(self, x):
        n = len(x)
        # Cast for reflection padding; convert back before uint8 RandAugment.
        x = F.pad(x.float(), (4, 4, 4, 4), mode='reflect').to(torch.uint8)
        offsets = torch.randint(0, 9, (n, 2), device=x.device)
        rows = self.grid[None, :, None] + offsets[:, 0, None, None]
        cols = self.grid[None, None, :] + offsets[:, 1, None, None]
        x = x.permute(0, 2, 3, 1)[torch.arange(n, device=x.device)[:, None, None], rows, cols].permute(0, 3, 1, 2)
        x = torch.where(torch.rand(n, 1, 1, 1, device=x.device) < .5, x.flip(-1), x)
        if self.ra is not None:
            x = self.ra(x)
        return ((x.float() / 255 - self.mean) / self.std).contiguous(memory_format=torch.channels_last)


def mix_batch(x, y, config):
    """Mix labels by actual clipped CutMix area, or the sampled Mixup weight."""
    order = torch.randperm(len(x), device=x.device)
    if np.random.random() < config['cutmix_probability']:
        lam = float(np.random.beta(config['cutmix_alpha'], config['cutmix_alpha']))
        side = int(32 * math.sqrt(1 - lam))
        cx, cy = np.random.randint(0, 32, 2)
        x1, x2 = max(0, cx - side // 2), min(32, cx + side // 2)
        y1, y2 = max(0, cy - side // 2), min(32, cy + side // 2)
        x[:, :, y1:y2, x1:x2] = x[order, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1)) / 1024
        kind = 'cutmix'
    else:
        alpha = config['mixup_alpha']
        lam = float(np.random.beta(alpha, alpha)) if alpha else 1.0
        x = lam * x + (1 - lam) * x[order]
        kind = 'mixup'
    return x, y, y[order], lam, kind
