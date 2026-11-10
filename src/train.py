import os, subprocess
import torch
def save_atomic(obj,path):
    temp=path.with_suffix('.tmp')
    torch.save(obj,temp)
    os.replace(temp,path)

def gpu_status():
    try:
        value=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,creationflags=0x08000000).strip().split(',')
        return dict(zip(['temperature_c','gpu_total_used_mib','power_w'],map(float,value)))
    except Exception: return {}

