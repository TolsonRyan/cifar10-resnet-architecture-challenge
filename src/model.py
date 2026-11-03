"""CIFAR residual network, implemented locally; no pretrained model imports.

Architecture references: He et al. (2016), Zagoruyko & Komodakis (2016).
"""
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        identity = self.skip(x)
        y = self.act(self.bn1(self.conv1(x)))
        return self.act(self.bn2(self.conv2(y)) + identity)

class ResNet(nn.Module):
    def __init__(self,width=32,blocks=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,width,3,padding=1,bias=False), nn.BatchNorm2d(width),nn.ReLU(inplace=True))
        layers=[]
        cin=width
        for stage,cout in enumerate([width,2*width,4*width]):
            for b in range(blocks):
                layers.append(Block(cin,cout,2 if stage>0 and b==0 else 1))
                cin=cout
        self.blocks=nn.Sequential(*layers)
        self.head=nn.Linear(cin,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self,x):
        return self.head(self.blocks(self.stem(x)).mean((2,3)))

def build(config):
    model=ResNet(config['width'],config['blocks'])
    n=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0<n<=5_000_000, n
    return model,n
