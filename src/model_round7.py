"""A local CIFAR PyramidNet implementation with image-level ShakeDrop.

Architecture concepts: Han et al., https://arxiv.org/abs/1610.02915
Block order cross-checked against the authors' PyramidNet-PyTorch/PyramidNet.py.
ShakeDrop equations (5)-(6): https://arxiv.org/abs/1802.02375
No downloaded implementation, pretrained tensors or external data are loaded.
"""
import torch
from torch import nn
from torch.nn import functional as F



class ShakeMultiply(torch.autograd.Function):
    """Use different, explicitly sampled forward and backward multipliers."""
    @staticmethod
    def forward(ctx, residual, forward_scale, backward_scale):
        ctx.save_for_backward(backward_scale)
        return residual * forward_scale

    @staticmethod
    def backward(ctx, gradient):
        (backward_scale,) = ctx.saved_tensors
        return gradient * backward_scale, None, None


class ShakeDrop(nn.Module):
    def __init__(self, death_probability):
        super().__init__()
        assert 0 <= death_probability <= 1
        self.death_probability = death_probability

    def forward(self, residual):
        keep = 1 - self.death_probability
        if not self.training:
            # E[alpha] = 0, hence E[b + alpha - b*alpha] = keep.
            return residual * keep
        gate = (torch.rand((), device=residual.device) < keep).to(residual.dtype)
        shape = (len(residual), 1, 1, 1)
        alpha = torch.rand(shape, device=residual.device, dtype=residual.dtype) * 2 - 1
        beta = torch.rand(shape, device=residual.device, dtype=residual.dtype)
        return ShakeMultiply.apply(residual, gate + (1 - gate) * alpha,
                                   gate + (1 - gate) * beta)


class PyramidBlock(nn.Module):
    def __init__(self, cin, cout, stride, death_probability):
        super().__init__()
        assert cout >= cin
        self.stride, self.extra_channels = stride, cout - cin
        self.input_bn = nn.BatchNorm2d(cin)
        self.first = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.middle_bn = nn.BatchNorm2d(cout)
        self.second = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.output_bn = nn.BatchNorm2d(cout)
        self.shake = ShakeDrop(death_probability)

    def forward(self, x):
        residual = self.first(self.input_bn(x))
        residual = self.second(F.relu(self.middle_bn(residual)))
        residual = self.shake(self.output_bn(residual))
        skip = F.avg_pool2d(x, 2, 2, ceil_mode=True) if self.stride == 2 else x
        # Zero-channel padding preserves every existing identity channel.
        if self.extra_channels:
            skip = F.pad(skip, (0, 0, 0, 0, 0, self.extra_channels))
        return skip + residual


class PyramidNet(nn.Module):
    def __init__(self, depth, alpha, shake_death):
        super().__init__()
        assert (depth - 2) % 6 == 0 and depth >= 8
        per_stage = (depth - 2) // 6
        count = 3 * per_stage
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16))
        blocks, cin = [], 16
        for i in range(count):
            cout = round(16 + alpha * (i + 1) / count)
            stride = 2 if i in (per_stage, 2 * per_stage) else 1
            blocks.append(PyramidBlock(cin, cout, stride, shake_death * (i + 1) / count))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.final_bn = nn.BatchNorm2d(cin)
        self.classifier = nn.Linear(cin, 10)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        features = F.relu(self.final_bn(self.blocks(self.stem(images))))
        return self.classifier(features.mean((2, 3)))


def build(config):
    if config.get('architecture') != 'pyramid_shakedrop':
        raise ValueError('Only the final PyramidNet architecture is supported')
    model = PyramidNet(config['depth'], config['alpha'], config['shake_death'])
    count = sum(p.numel() for p in model.parameters())
    assert count == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < count <= 5_000_000, count
    return model, count
