"""Sequential orchestration; never submits to Kaggle or publishes anything.

Stops on errors, resumes completed epochs, and checks baseline inference before
spending GPU time on larger candidates. Call after the successful CUDA pilot.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def call(args,log):
    command=[sys.executable,'-u',*args]
    print('RUN '+subprocess.list2cmdline(command),flush=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\nCOMMAND '+subprocess.list2cmdline(command)+'\n'); f.flush()
        p=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in p.stdout:
            f.write(line);f.flush();print(line,end='',flush=True)
        code=p.wait()
        if code: raise RuntimeError(f'Command failed ({code}); see {log}')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--configs',nargs='+',default=['baseline','wide_cutout','wide_mixup']); a=parser.parse_args()
    results=ROOT/'results'; results.mkdir(exist_ok=True)
    status_path=results/'progress.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    try:
        for name in a.configs:
            c=json.loads((ROOT/'configs'/f'{name}.json').read_text())
            run=ROOT/'runs'/c['name'];run.mkdir(parents=True,exist_ok=True)
            summary_path=run/'summary.json'
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
            status.update(stage='training',active_run=c['name'],training_started=True,kaggle_submitted=False)
            status_path.write_text(json.dumps(status,indent=2))
            if summary.get('status')!='completed':
                args=['src/train.py','--config',f'configs/{name}.json']
                if (run/'last.pt').exists():args.append('--resume')
                call(args,run/'console.log')
            call(['src/evaluate.py','--checkpoint',f'runs/{c["name"]}/best.pt'],run/'console.log')
            call(['src/predict.py','--checkpoint',f'runs/{c["name"]}/best.pt','--output',f'submissions/{c["name"]}/submission.csv'],run/'console.log')
            call(['src/summarize.py'],results/'pipeline.log')
            complete=[json.loads(p.read_text()) for p in (ROOT/'runs').glob('*/summary.json') if p.parent.name!='pilot']
            complete=[s for s in complete if s['status']=='completed']
            winner=max(complete,key=lambda s:(s['best_val_accuracy'],-s['best_val_loss']))
            # At every completed milestone keep a checked, best-so-far submission.
            call(['src/predict.py','--checkpoint',f'runs/{winner["name"]}/best.pt'],results/'pipeline.log')
            status.update(best_completed_run=winner['name'],validation_accuracy=winner['best_val_accuracy'],parameters=winner['parameters'])
            status_path.write_text(json.dumps(status,indent=2))
        status.update(stage='experiments_completed',active_run=None)
        status_path.write_text(json.dumps(status,indent=2))
    except Exception as exc:
        status.update(stage='error',error=repr(exc));status_path.write_text(json.dumps(status,indent=2));raise

if __name__=='__main__':main()
