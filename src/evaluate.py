import argparse,json
from pathlib import Path
import numpy as np
import torch
from data import ROOT,load_train,Transform
from model import build
from train import evaluate

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); args=p.parse_args()
    torch.set_num_threads(4)
    ck=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=False)
    model,n=build(ck['config']); model.load_state_dict(ck['model']); model.cuda().to(memory_format=torch.channels_last).eval()
    x,y,_,idx=load_train(); t=Transform(**ck['normalization'])
    loss,accuracy=evaluate(model,x,y,idx,t)
    truth=[]; predictions=[]
    for batch in idx.split(256):
        with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(t(x[batch]))
        predictions.extend(logits.argmax(1).cpu().tolist()); truth.extend(y[batch].cpu().tolist())
    confusion=np.zeros((10,10),dtype=int)
    np.add.at(confusion,(truth,predictions),1)
    count=len(idx); z=1.959963984540054
    center=(accuracy+z*z/(2*count))/(1+z*z/count)
    half=z*((accuracy*(1-accuracy)/count+z*z/(4*count*count))**.5)/(1+z*z/count)
    np.savez((ROOT/args.checkpoint).parent/'validation_predictions.npz',indices=idx.cpu().numpy(),truth=np.array(truth),prediction=np.array(predictions))
    result={'checkpoint':args.checkpoint,'parameters':n,'validation_loss':loss,'validation_accuracy':accuracy,'validation_count':len(idx),'wilson_95_interval':[center-half,center+half],'interval_note':'Descriptive binomial interval; does not adjust for repeated model/epoch selection on this validation set.','confusion_matrix':confusion.tolist(),'per_class_accuracy':(confusion.diagonal()/confusion.sum(1)).tolist()}
    path=(ROOT/args.checkpoint).parent/'evaluation.json'; path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
