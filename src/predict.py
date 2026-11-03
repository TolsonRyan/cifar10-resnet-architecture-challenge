"""Predict from one legal checkpoint, preserving the official submission ID order."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from audit_data import read_csv
from data import ROOT,Transform
from model import build

def validate(path):
    fields,rows=read_csv(path)
    _,sample=read_csv(ROOT/'data/sample_submission.csv')
    _,test=read_csv(ROOT/'data/test.csv')
    assert fields==['id','label']
    assert len(rows)==len(sample)==len(test)==2000
    ids=[r['id'] for r in rows]
    assert ids==[r['id'] for r in sample]
    assert len(set(ids))==2000 and set(ids)=={r['id'] for r in test}
    assert all(r['label'] in [str(i) for i in range(10)] for r in rows)
    return {'status':'passed','rows':len(rows),'columns':fields,'unique_ids':len(set(ids)),'matches_sample_order':True,'matches_test_ids':True,'integer_labels_0_to_9':True,'label_counts':np.bincount([int(r['label']) for r in rows],minlength=10).tolist(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint'); p.add_argument('--output',default='submissions/submission.csv'); p.add_argument('--validate-only',action='store_true'); a=p.parse_args()
    out=ROOT/a.output
    if a.validate_only:
        print(json.dumps(validate(out),indent=2)); return
    torch.set_num_threads(4)
    ck=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=False)
    model,n=build(ck['config']); assert n==ck['parameters']
    model.load_state_dict(ck['model']); model.cuda().to(memory_format=torch.channels_last).eval()
    transform=Transform(**ck['normalization'])
    array=np.load(ROOT/'data/cache/test.npy')
    test_ids=np.load(ROOT/'data/cache/test_ids.npy')
    preds=[]
    for start in range(0,len(array),256):
        x=torch.from_numpy(array[start:start+256]).permute(0,3,1,2).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(transform(x))
        preds.extend(logits.argmax(1).cpu().tolist())
    mapping=dict(zip(test_ids.tolist(),preds))
    _,sample=read_csv(ROOT/'data/sample_submission.csv')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','label']); w.writeheader()
        w.writerows({'id':r['id'],'label':mapping[int(r['id'])]} for r in sample)
    result=validate(out)
    result.update(checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256((ROOT/a.checkpoint).read_bytes()).hexdigest(),parameters=n,validation_accuracy=ck['best_acc'],best_epoch=ck['best_epoch'],inference='single model, no TTA, no ensemble',kaggle_submitted=False)
    out.with_suffix('.checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
