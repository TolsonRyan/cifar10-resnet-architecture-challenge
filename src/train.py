"""CUDA training with complete checkpoint/RNG recovery and real epoch metrics."""
import argparse, copy, csv, hashlib, json, math, os, platform, random, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from data import ROOT,load_train,Transform
from model import build

def save_atomic(obj,path):
    temp=path.with_suffix('.tmp')
    torch.save(obj,temp)
    os.replace(temp,path)

def gpu_status():
    try:
        value=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,creationflags=0x08000000).strip().split(',')
        return dict(zip(['temperature_c','gpu_total_used_mib','power_w'],map(float,value)))
    except Exception: return {}

@torch.inference_mode()
def evaluate(model,x,y,indices,transform,batch_size=256):
    model.eval()
    total_loss=torch.zeros((),device='cuda'); correct=torch.zeros((),device='cuda')
    for idx in indices.split(batch_size):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(transform(x[idx]))
            loss=nn.functional.cross_entropy(logits,y[idx],reduction='sum')
        total_loss+=loss.float(); correct+=(logits.argmax(1)==y[idx]).sum()
    return total_loss.item()/len(indices),correct.item()/len(indices)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int,help='Gracefully stop at an epoch boundary for checkpoint/recovery testing')
    args=parser.parse_args()
    c=json.loads(Path(args.config).read_text())
    assert json.loads((ROOT/'results/data_audit.json').read_text())['status']=='passed'
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    run=ROOT/'runs'/c['name']; run.mkdir(parents=True,exist_ok=True)
    if (run/'metrics.csv').exists() and not args.resume: raise RuntimeError('Run already exists; use --resume or a new name')
    if args.resume:
        assert json.loads((run/'config.json').read_text())==c, 'Resume config must be unchanged'
    (run/'config.json').write_text(json.dumps(c,indent=2))
    model,n=build(c); model=model.cuda().to(memory_format=torch.channels_last)
    optimizer=torch.optim.SGD(model.parameters(),lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'],nesterov=True)
    x,y,train_idx,val_idx=load_train()
    # Statistics use training indices only, never validation or hidden test pixels.
    stats_sum=torch.zeros(3,device='cuda',dtype=torch.float64)
    stats_sq=stats_sum.clone()
    for idx in train_idx.split(512):
        z=x[idx].double()/255
        stats_sum+=z.sum((0,2,3)); stats_sq+=z.square().sum((0,2,3))
    denom=len(train_idx)*32*32
    mean=(stats_sum/denom); std=(stats_sq/denom-mean.square()).sqrt()
    transform=Transform(mean.tolist(),std.tolist())
    norm={'mean':mean.tolist(),'std':std.tolist()}
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':torch.cuda.get_device_name(0),'trainable_parameters':n,'normalization':norm,'split_sha256':hashlib.sha256((ROOT/'results/split_seed2026.npz').read_bytes()).hexdigest(),'precision':'bfloat16','cudnn_benchmark':True,'determinism_note':'Seeds/RNG saved; exact bitwise CUDA reproducibility is not guaranteed.','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}}
    (run/'environment.json').write_text(json.dumps(env,indent=2))
    source=run/'source'; source.mkdir(exist_ok=True)
    for p in (ROOT/'src').glob('*.py'): (source/p.name).write_bytes(p.read_bytes())
    start=0; best_acc=-1.; best_loss=float('inf'); best_epoch=0; elapsed_prior=0.
    if args.resume:
        ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        start=ck['epoch']; best_acc=ck['best_acc']; best_loss=ck['best_loss']; best_epoch=ck['best_epoch']; elapsed_prior=ck['elapsed_seconds']
        random.setstate(ck['rng_python']); np.random.set_state(ck['rng_numpy']); torch.set_rng_state(ck['rng_torch']); torch.cuda.set_rng_state_all(ck['rng_cuda'])
    print(json.dumps({'run':c['name'],'parameters':n,'start_epoch':start,'normalization':norm}),flush=True)
    wall=time.perf_counter()
    fields=['epoch','lr','train_loss','train_accuracy','val_loss','val_accuracy','epoch_seconds','elapsed_seconds','peak_allocated_mib','temperature_c','gpu_total_used_mib','power_w']
    metric_path=run/'metrics.csv'
    if start:
        with metric_path.open(newline='') as f: prior=[r for r in csv.DictReader(f) if int(r['epoch'])<=start]
        with metric_path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(prior)
    with metric_path.open('a',newline='',buffering=1) as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        if not start: writer.writeheader()
        for epoch in range(start,c['epochs']):
            tick=time.perf_counter(); torch.cuda.reset_peak_memory_stats(); model.train()
            permutation=train_idx[torch.randperm(len(train_idx),device='cuda')]
            loss_sum=torch.zeros((),device='cuda'); correct=torch.zeros((),device='cuda')
            batches=permutation.split(c['batch_size'])
            for step,idx in enumerate(batches):
                progress=epoch+step/len(batches)
                warmup=c['warmup_epochs']
                factor=(progress+1/len(batches))/warmup if progress<warmup else 0.5*(1+math.cos(math.pi*(progress-warmup)/(c['epochs']-warmup)))
                lr=c['lr']*(c['min_lr_ratio']+(1-c['min_lr_ratio'])*min(1.,factor))
                for group in optimizer.param_groups: group['lr']=lr
                inputs=transform(x[idx],train=True,cutout=c['cutout'])
                labels=y[idx]
                mix=c.get('mixup_alpha',0)
                if mix:
                    lam=float(np.random.beta(mix,mix))
                    order=torch.randperm(len(idx),device='cuda')
                    inputs=lam*inputs+(1-lam)*inputs[order]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(inputs)
                    loss=nn.functional.cross_entropy(logits,labels,label_smoothing=c['label_smoothing'])
                    if mix: loss=lam*loss+(1-lam)*nn.functional.cross_entropy(logits,labels[order],label_smoothing=c['label_smoothing'])
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
                loss.backward(); optimizer.step()
                loss_sum+=loss.detach().float()*len(idx)
                if mix: correct+=lam*(logits.argmax(1)==labels).sum()+(1-lam)*(logits.argmax(1)==labels[order]).sum()
                else: correct+=(logits.argmax(1)==labels).sum()
            val_loss,val_acc=evaluate(model,x,y,val_idx,transform)
            torch.cuda.synchronize()
            improved=val_acc>best_acc or (val_acc==best_acc and val_loss<best_loss)
            if improved: best_acc=val_acc; best_loss=val_loss; best_epoch=epoch+1
            elapsed=elapsed_prior+time.perf_counter()-wall
            record=dict(epoch=epoch+1,lr=lr,train_loss=loss_sum.item()/len(train_idx),train_accuracy=correct.item()/len(train_idx),val_loss=val_loss,val_accuracy=val_acc,epoch_seconds=time.perf_counter()-tick,elapsed_seconds=elapsed,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,**gpu_status())
            checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'best_acc':best_acc,'best_loss':best_loss,'best_epoch':best_epoch,'elapsed_seconds':elapsed,'config':c,'parameters':n,'normalization':norm,'split_sha256':env['split_sha256'],'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'scheduler':{'type':'per_step_warmup_cosine','completed_epochs':epoch+1,'total_epochs':c['epochs']}}
            save_atomic(checkpoint,run/'last.pt')
            if improved: save_atomic({k:v for k,v in checkpoint.items() if k!='optimizer'},run/'best.pt')
            writer.writerow(record); file.flush()
            summary={'name':c['name'],'parameters':n,'best_val_accuracy':best_acc,'best_val_loss':best_loss,'best_epoch':best_epoch,'completed_epochs':epoch+1,'planned_epochs':c['epochs'],'elapsed_seconds':elapsed,'status':'completed' if epoch+1==c['epochs'] else 'running'}
            (run/'summary.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps(record),flush=True)
            if args.stop_after_epoch and epoch+1>=args.stop_after_epoch:
                print('STOPPED_AT_CHECKPOINT '+json.dumps(summary),flush=True)
                return
    print('COMPLETED '+json.dumps(summary),flush=True)

if __name__=='__main__': main()
