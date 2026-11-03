"""Record actual hardware/software and verify forward/backward on CUDA."""
import json,platform,shutil,subprocess
import numpy as np
import PIL,psutil,torch
from data import ROOT,Transform
from model import build

def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'CUDA PyTorch is required'
    assert torch.cuda.is_bf16_supported()
    info={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'numpy':np.__version__,'pillow':PIL.__version__,'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'bf16':torch.cuda.is_bf16_supported(),'ram':dict(psutil.virtual_memory()._asdict()),'disk':dict(shutil.disk_usage(ROOT)._asdict())}
    info['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True,creationflags=0x08000000)
    info['candidates']={}
    for w in [32,48,64]:
        m,n=build({'width':w,'blocks':3}); m.cuda().to(memory_format=torch.channels_last)
        t=Transform([.5]*3,[.25]*3)
        x=t(torch.randint(0,256,(8,3,32,32),device='cuda',dtype=torch.uint8),train=True,cutout=16)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=m(x); loss=logits.float().square().mean()
        loss.backward()
        assert logits.shape==(8,10)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        info['candidates'][str(w)]={'parameters':n,'forward_backward':'passed'}
        del m,logits,loss
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/environment.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
