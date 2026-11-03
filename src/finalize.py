"""Audit completed artifacts and export report assets from real logs only."""
import csv, hashlib, json, shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ROOT
from model import build
from predict import validate

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2)
    results=ROOT/'results'; export=results/'experiments'; export.mkdir(exist_ok=True)
    figures=ROOT/'report/figures'; figures.mkdir(parents=True,exist_ok=True)
    names=['baseline','wide_cutout','wide_mixup']
    audits=[]
    for name in names:
        run=ROOT/'runs'/name
        c=json.loads((run/'config.json').read_text()); summary=json.loads((run/'summary.json').read_text())
        evaluation=json.loads((run/'evaluation.json').read_text())
        with (run/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
        assert summary['status']=='completed' and len(rows)==c['epochs']==summary['completed_epochs']
        assert [int(r['epoch']) for r in rows]==list(range(1,c['epochs']+1))
        assert all(np.isfinite(float(r[k])) for r in rows for k in ['train_loss','train_accuracy','val_loss','val_accuracy','epoch_seconds'])
        expected=max(rows,key=lambda r:(float(r['val_accuracy']),-float(r['val_loss'])))
        ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        model,n=build(c); model.load_state_dict(ck['model'],strict=True)
        assert n==ck['parameters']==summary['parameters'] and n<=5_000_000
        assert ck['epoch']==int(expected['epoch'])==summary['best_epoch']
        assert ck['best_acc']==float(expected['val_accuracy'])==summary['best_val_accuracy']
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        assert last['epoch']==c['epochs'] and 'optimizer' in last and 'rng_cuda' in last and 'scheduler' in last
        assert ck['split_sha256']==sha(results/'split_seed2026.npz')
        destination=export/name; destination.mkdir(exist_ok=True)
        for filename in ['config.json','summary.json','environment.json','metrics.csv','evaluation.json','curves.png']:
            shutil.copy2(run/filename,destination/filename)
        audits.append({'name':name,'status':'passed','epochs':len(rows),'parameters':n,'best_epoch':ck['epoch'],'training_best_accuracy':ck['best_acc'],'reevaluated_accuracy':evaluation['validation_accuracy'],'checkpoint_sha256':sha(run/'best.pt'),'seconds':summary['elapsed_seconds'],'maximum_temperature_c':max(float(r['temperature_c']) for r in rows),'peak_allocated_mib':max(float(r['peak_allocated_mib']) for r in rows)})
    winner=max(audits,key=lambda a:a['reevaluated_accuracy'])
    assert winner['name']=='wide_mixup'
    submission=validate(ROOT/'submissions/submission.csv')
    checks=json.loads((ROOT/'submissions/submission.checks.json').read_text())
    assert checks['checkpoint']==f'runs/{winner["name"]}/best.pt'
    assert checks['checkpoint_sha256']==winner['checkpoint_sha256']
    assert checks['sha256']==submission['sha256']
    assert (ROOT/'submissions/submission.csv').read_bytes()==(ROOT/'submissions/reproduction_check/submission.csv').read_bytes()
    audit=json.loads((results/'data_audit.json').read_text())
    for name,info in audit['files'].items(): assert sha(ROOT/'data'/name)==info['sha256']
    (results/'artifact_audit.json').write_text(json.dumps({'status':'passed','runs':audits,'submission':submission,'prediction_reproduction':'byte-identical','csv_source_hashes_unchanged':True},indent=2))
    evaluation=json.loads((ROOT/'runs'/winner['name']/'evaluation.json').read_text())
    final={'run':winner['name'],'checkpoint':checks['checkpoint'],'checkpoint_sha256':winner['checkpoint_sha256'],'parameters':winner['parameters'],'parameter_limit':5000000,'validation_accuracy':evaluation['validation_accuracy'],'validation_correct':int(round(5000*evaluation['validation_accuracy'])),'validation_total':5000,'validation_loss':evaluation['validation_loss'],'training_images':45000,'validation_images':5000,'best_epoch':winner['best_epoch'],'epochs':winner['epochs'],'training_seconds':winner['seconds'],'all_formal_experiment_training_seconds':sum(a['seconds'] for a in audits),'submission':'submissions/submission.csv','submission_checks':submission,'random_initialization':True,'external_training_data':False,'pretrained_weights':False,'ensemble':False,'test_time_augmentation':False,'full_50000_retraining':False,'kaggle_submitted':False,'kaggle_score':None,'uncertainty_note':evaluation['interval_note'],'wilson_95_interval':evaluation['wilson_95_interval']}
    (results/'final_model_metrics.json').write_text(json.dumps(final,indent=2))
    shutil.copy2(ROOT/'configs/wide_mixup.json',results/'final_config.json')
    # Paired error counts, without asserting statistical significance from best-of-epoch selection.
    preds={name:dict(np.load(ROOT/'runs'/name/'validation_predictions.npz')) for name in names}
    paired={}
    for name in names[:-1]:
        a,b=preds[name],preds[winner['name']]
        assert np.array_equal(a['indices'],b['indices']) and np.array_equal(a['truth'],b['truth'])
        ac=a['prediction']==a['truth'];bc=b['prediction']==b['truth']
        paired[name]={'both_correct':int((ac&bc).sum()),'only_final_correct':int((~ac&bc).sum()),'only_comparator_correct':int((ac&~bc).sum()),'both_wrong':int((~ac&~bc).sum())}
    (results/'paired_comparison.json').write_text(json.dumps(paired,indent=2))
    cm=np.array(evaluation['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax,fraction=.046)
    ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted class ID',ylabel='True class ID',title='Final model: validation confusion matrix (n=5,000)')
    for i in range(10):
        for j in range(10): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>250 else 'black')
    fig.tight_layout();fig.savefig(figures/'confusion_matrix.png',dpi=180);plt.close(fig)
    for filename in ['training_curves.png','validation_comparison.png']:shutil.copy2(results/filename,figures/filename)
    shutil.copy2(ROOT/'runs/wide_mixup/curves.png',figures/'final_training_curves.png')
    progress=json.loads((results/'progress.json').read_text())
    report_ready=(ROOT/'report/aaai_report.tex').exists() and (ROOT/'report/README.md').exists()
    progress.update(stage='local_deliverables_complete_external_actions_pending' if report_ready else 'training_and_submission_verified',candidate_plan_note='All three planned formal experiments completed; budgets were 60, 160, and 160 epochs.',prediction_reproduction='byte-identical',report_status='source_and_figures_prepared_pdf_not_compiled' if report_ready else 'in_preparation',local_git_initialized=(ROOT/'.git').exists())
    progress.setdefault('github_remote_created',False)
    team_file=ROOT/'configs/team.json'
    team=json.loads(team_file.read_text()) if team_file.exists() else {}
    if team:
        progress['team_members']=team['members'];progress['kaggle_team_name']=team['kaggle_team_name']
        progress['github_connected']=team.get('github_connected',False);progress['github_login']=team.get('github_login')
        progress['github_repository']=team.get('github_repository')
        progress['github_remote_created']=bool(team.get('github_repository'))
    progress['pending_user_actions']=['Kaggle submission decision','Course registration confirmation','Report template/format confirmation and PDF compilation']
    if not team.get('github_repository'): progress['pending_user_actions'].insert(1,'Private GitHub repository creation/access')
    (results/'progress.json').write_text(json.dumps(progress,indent=2))
    print(json.dumps({'final':final,'paired_comparison':paired,'artifact_audit':'passed'},indent=2))

if __name__=='__main__':main()
