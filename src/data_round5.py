"""Explicit split-aware loading of allowed cached training images only."""
import hashlib
import numpy as np
import torch
from data import ROOT, Transform


def load_train(split_path='results/split_seed2026.npz'):
    x = torch.from_numpy(np.load(ROOT / 'data/cache/train.npy')).permute(0, 3, 1, 2).contiguous().cuda()
    y = torch.from_numpy(np.load(ROOT / 'data/cache/labels.npy')).cuda()
    with np.load(ROOT / split_path) as split:
        train, val = split['train'], split['val']
    assert len(train) == 45000 and len(val) == 5000
    combined = np.concatenate([train, val])
    assert np.array_equal(np.sort(combined), np.arange(50000))
    return x, y, torch.as_tensor(train, device='cuda'), torch.as_tensor(val, device='cuda')


def create_confirmation_split():
    output = ROOT / 'results/round5/split_seed52029.npz'
    if output.exists():
        return output
    labels = np.load(ROOT / 'data/cache/labels.npy')
    rng = np.random.default_rng(52029)
    training, validation = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label))
        assert len(indices) == 5000
        validation.extend(indices[:500])
        training.extend(indices[500:])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, train=np.sort(training).astype(np.int32), val=np.sort(validation).astype(np.int32))
    return output
