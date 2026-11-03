"""Generate figures/tables only from saved real experiment logs."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ROOT

def main():
    results=ROOT/'results'; results.mkdir(exist_ok=True)
    summaries=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for run in sorted((ROOT/'runs').iterdir()):
        if not (run/'summary.json').exists(): continue
        s=json.loads((run/'summary.json').read_text()); c=json.loads((run/'config.json').read_text())
        with (run/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
        epochs=[int(r['epoch']) for r in rows]
        f,a=plt.subplots(1,2,figsize=(11,4))
        for key,axis in [('accuracy',a[0]),('loss',a[1])]:
            for split in ['train','val']: axis.plot(epochs,[float(r[split+'_'+key]) for r in rows],label=split)
            axis.set(xlabel='Epoch',ylabel=key,title=c['name']); axis.legend(); axis.grid(alpha=.2)
        f.tight_layout(); f.savefig(run/'curves.png',dpi=160); plt.close(f)
        if c['name']=='pilot': continue
        for ax,key in zip(axes,['val_accuracy','val_loss']): ax.plot(epochs,[float(r[key]) for r in rows],label=c['name']); ax.set(xlabel='Epoch',ylabel=key); ax.grid(alpha=.2)
        evaluation=json.loads((run/'evaluation.json').read_text()) if (run/'evaluation.json').exists() else {}
        summaries.append({**s,'reevaluated_val_accuracy':evaluation.get('validation_accuracy'),'reevaluated_val_loss':evaluation.get('validation_loss'),'width':c['width'],'blocks_per_stage':c['blocks'],'augmentation':f"reflect crop+flip; cutout={c['cutout']}; mixup={c['mixup_alpha']}",'optimizer':'SGD/Nesterov','lr':c['lr'],'weight_decay':c['weight_decay'],'label_smoothing':c['label_smoothing']})
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(results/'training_curves.png',dpi=180); plt.close(fig)
    if summaries:
        with (results/'experiment_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
        fig,ax=plt.subplots(figsize=(7,4)); ax.bar([s['name'] for s in summaries],[100*s['best_val_accuracy'] for s in summaries]); ax.set(ylabel='Best validation accuracy (%)'); fig.tight_layout(); fig.savefig(results/'validation_comparison.png',dpi=180); plt.close(fig)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
