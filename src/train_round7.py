"""Scratch training for the PyramidNet/ShakeDrop architecture study.

No hidden test images are loaded. Validation uses FP32 throughout. Full-data
runs have no validation score and use a preselected final epoch/weight policy.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from data_round5 import ROOT, Transform, load_train
from model_round7 import build
from augment_round7 import TrainTransform
from train import save_atomic, gpu_status


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        dst, src = [], []
        live_state = model.state_dict()
        for key, value in self.model.state_dict().items():
            live = live_state[key]
            if value.is_floating_point():
                dst.append(value)
                src.append(live)
            else:
                value.copy_(live)
        torch._foreach_lerp_(dst, src, 1 - decay)


@torch.inference_mode()
def evaluate(model, x, y, indices, transform):
    model.eval()
    loss, correct = torch.zeros((), device='cuda'), torch.zeros((), device='cuda')
    for idx in indices.split(256):
        logits = model(transform(x[idx]))
        loss += F.cross_entropy(logits, y[idx], reduction='sum')
        correct += (logits.argmax(1) == y[idx]).sum()
    return loss.item() / len(indices), correct.item() / len(indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert c['name'].startswith('round7/')
    assert c['optimizer'] == 'sgd'
    assert c['cutmix_probability'] == 0 and c['mixup_alpha'] == 0
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    run = ROOT / 'runs' / c['name']
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'config.json').exists():
        assert args.resume, 'Choose a new run directory or explicitly resume'
        assert json.loads((run / 'config.json').read_text()) == c
    else:
        assert not args.resume
        (run / 'config.json').write_text(json.dumps(c, indent=2))
    model, n = build(c)
    model = model.cuda().to(memory_format=torch.channels_last)
    ema = EMA(model, c['ema_decay'])
    opt = torch.optim.SGD(model.parameters(), lr=c['lr'], momentum=c['momentum'], nesterov=True, weight_decay=c['weight_decay'])
    x, y, train_idx, val_idx = load_train(c['split_path'])
    if c['full_data']:
        train_idx = torch.arange(len(x), device='cuda')
        val_idx = val_idx[:0]
    total = torch.zeros(3, dtype=torch.float64, device='cuda')
    square = total.clone()
    for idx in train_idx.split(512):
        z = x[idx].double() / 255
        total += z.sum((0, 2, 3)); square += z.square().sum((0, 2, 3))
    mean = total / (len(train_idx) * 1024)
    std = (square / (len(train_idx) * 1024) - mean.square()).sqrt()
    norm = {'mean': mean.tolist(), 'std': std.tolist()}
    train_transform = TrainTransform(**norm, config=c)
    eval_transform = Transform(**norm)
    split_hash = hashlib.sha256((ROOT / c['split_path']).read_bytes()).hexdigest() if not c['full_data'] else None
    env = dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), device=torch.cuda.get_device_name(), parameters=n, normalization=norm,
               train_count=len(train_idx), validation_count=len(val_idx), split_sha256=split_hash,
               training_precision='bfloat16', evaluation_precision='float32', tf32=False, cudnn_benchmark=True,
               reproducibility='RNG and optimizer states saved; CUDA kernels are not guaranteed bitwise deterministic.')
    if not args.resume:
        source = run / 'source'; source.mkdir()
        for p in (ROOT / 'src').glob('*.py'):
            (source / p.name).write_bytes(p.read_bytes())
        env['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
        (run / 'environment.json').write_text(json.dumps(env, indent=2))
    best = {key: {'accuracy': -1., 'loss': float('inf'), 'epoch': 0} for key in ('raw', 'ema')}
    start, elapsed_prior = 0, 0.
    if args.resume:
        ck = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        assert ck['config'] == c and ck['normalization'] == norm
        model.load_state_dict(ck['model']); ema.model.load_state_dict(ck['ema']); ema.updates = ck['ema_updates']
        opt.load_state_dict(ck['optimizer']); start = ck['epoch']; best = ck['best']; elapsed_prior = ck['elapsed_seconds']
        random.setstate(ck['rng_python']); np.random.set_state(ck['rng_numpy']); torch.set_rng_state(ck['rng_torch']); torch.cuda.set_rng_state_all(ck['rng_cuda'])
    fields = ['epoch', 'lr', 'train_loss', 'train_accuracy', 'raw_val_loss', 'raw_val_accuracy', 'ema_val_loss', 'ema_val_accuracy', 'epoch_seconds', 'elapsed_seconds', 'peak_allocated_mib', 'temperature_c', 'gpu_total_used_mib', 'power_w']
    metric_path = run / 'metrics.csv'
    if args.resume:
        with metric_path.open(newline='') as file:
            prior = [row for row in csv.DictReader(file) if int(row['epoch']) <= start]
        with metric_path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fields); writer.writeheader(); writer.writerows(prior)
    print(json.dumps({'run': c['name'], 'parameters': n, 'train_count': len(train_idx), 'validation_count': len(val_idx), 'start_epoch': start}), flush=True)
    wall = time.perf_counter()
    with metric_path.open('a', newline='', buffering=1) as file:
        writer = csv.DictWriter(file, fields)
        if not start:
            writer.writeheader()
        for epoch in range(start, c['epochs']):
            tick = time.perf_counter(); torch.cuda.reset_peak_memory_stats(); model.train()
            batches = train_idx[torch.randperm(len(train_idx), device='cuda')].split(c['batch_size'])
            loss_sum, correct = torch.zeros((), device='cuda'), torch.zeros((), device='cuda')
            for step, idx in enumerate(batches):
                progress = epoch + step / len(batches)
                warmup = c['warmup_epochs']
                factor = (progress + 1 / len(batches)) / warmup if progress < warmup else .5 * (1 + math.cos(math.pi * (progress - warmup) / (c['epochs'] - warmup)))
                lr = c['lr'] * (c['min_lr_ratio'] + (1 - c['min_lr_ratio']) * min(1., factor))
                for group in opt.param_groups:
                    group['lr'] = lr
                inputs, ya, yb, lam = train_transform(x[idx]), y[idx], y[idx], 1.0
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, ya, label_smoothing=c['label_smoothing'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                opt.step(); ema.update(model)
                loss_sum += loss.detach().float() * len(idx)
                correct += lam * (logits.argmax(1) == ya).sum() + (1 - lam) * (logits.argmax(1) == yb).sum()
            scores, improved = {}, []
            if len(val_idx):
                for key, net in [('raw', model), ('ema', ema.model)]:
                    loss, acc = evaluate(net, x, y, val_idx, eval_transform)
                    scores[f'{key}_val_loss'], scores[f'{key}_val_accuracy'] = loss, acc
                    if acc > best[key]['accuracy'] or (acc == best[key]['accuracy'] and loss < best[key]['loss']):
                        best[key] = dict(accuracy=acc, loss=loss, epoch=epoch + 1); improved.append(key)
            torch.cuda.synchronize()
            elapsed = elapsed_prior + time.perf_counter() - wall
            checkpoint = dict(model=model.state_dict(), ema=ema.model.state_dict(), ema_updates=ema.updates,
                              optimizer=opt.state_dict(), epoch=epoch + 1, config=c, parameters=n, normalization=norm,
                              best=best, elapsed_seconds=elapsed, split_sha256=split_hash,
                              rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                              scheduler={'type': 'per_step_warmup_cosine', 'completed_epochs': epoch + 1, 'total_epochs': c['epochs']})
            save_atomic(checkpoint, run / 'last.pt')
            for key in improved:
                saved = dict(model=(model if key == 'raw' else ema.model).state_dict(), weight_type=key, config=c, parameters=n,
                             normalization=norm, split_sha256=split_hash, epoch=epoch + 1, validation=best[key])
                save_atomic(saved, run / f'best_{key}.pt')
            if epoch + 1 == c['epochs']:
                for key, net in [('raw', model), ('ema', ema.model)]:
                    validation = dict(accuracy=scores[f'{key}_val_accuracy'], loss=scores[f'{key}_val_loss'], epoch=epoch + 1) if len(val_idx) else None
                    saved = dict(model=net.state_dict(), weight_type=key, config=c, parameters=n,
                                 normalization=norm, split_sha256=split_hash, epoch=epoch + 1, validation=validation)
                    save_atomic(saved, run / f'final_{key}.pt')
            record = dict(epoch=epoch + 1, lr=lr, train_loss=loss_sum.item() / len(train_idx),
                          train_accuracy=correct.item() / len(train_idx),
                          **scores, epoch_seconds=time.perf_counter() - tick, elapsed_seconds=elapsed,
                          peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20, **gpu_status())
            writer.writerow(record); file.flush()
            summary = dict(name=c['name'], parameters=n, best=best if len(val_idx) else None, completed_epochs=epoch + 1,
                           planned_epochs=c['epochs'], elapsed_seconds=elapsed, train_count=len(train_idx), validation_count=len(val_idx),
                           status='completed' if epoch + 1 == c['epochs'] else 'running')
            (run / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(record), flush=True)
            if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch:
                print('STOPPED_AT_CHECKPOINT', flush=True); return
    print('COMPLETED ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
