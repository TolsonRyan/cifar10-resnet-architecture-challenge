"""GPU-resident uint8 dataset and per-image randomized CIFAR augmentations."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]

def load_train():
    x=torch.from_numpy(np.load(ROOT/'data/cache/train.npy')).permute(0,3,1,2).contiguous().cuda()
    y=torch.from_numpy(np.load(ROOT/'data/cache/labels.npy')).cuda()
    split=np.load(ROOT/'results/split_seed2026.npz')
    return x,y,torch.tensor(split['train'],device='cuda'),torch.tensor(split['val'],device='cuda')

class Transform:
    def __init__(self,mean,std):
        self.mean=torch.tensor(mean,device='cuda').view(1,3,1,1)
        self.std=torch.tensor(std,device='cuda').view(1,3,1,1)
        self.grid=torch.arange(32,device='cuda')

    def __call__(self,x,train=False,cutout=0):
        x=x.float().div_(255)
        n=len(x)
        if train:
            # Reflection padding and independent uniform integer crop offsets.
            x=F.pad(x,(4,4,4,4),mode='reflect')
            offsets=torch.randint(0,9,(n,2),device=x.device)
            rows=self.grid[None,:,None]+offsets[:,0,None,None]
            cols=self.grid[None,None,:]+offsets[:,1,None,None]
            x=x.permute(0,2,3,1)[torch.arange(n,device=x.device)[:,None,None],rows,cols].permute(0,3,1,2)
            flip=torch.rand(n,1,1,1,device=x.device)<0.5
            x=torch.where(flip,x.flip(-1),x)
        x=(x-self.mean)/self.std
        if train and cutout:
            centers=torch.randint(0,32,(n,2),device=x.device)
            dy=self.grid[None,:,None]-centers[:,0,None,None]
            dx=self.grid[None,None,:]-centers[:,1,None,None]
            mask=(dy>=-cutout//2)&(dy<cutout//2)&(dx>=-cutout//2)&(dx<cutout//2)
            x=x.masked_fill(mask[:,None],0)
        return x.contiguous(memory_format=torch.channels_last)
