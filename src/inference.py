"""FP32 inference using the submitted model and exact ordered 50-view policy."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from model_round7 import build
ROOT=Path(__file__).resolve().parents[1]

def views(z):
 yield z;yield z.flip(-1)
 padded=F.pad(z,(2,2,2,2),mode='reflect')
 offsets=[(0,0),(0,4),(4,0),(4,4),(0,2),(2,0),(2,4),(4,2)]
 offsets += [(r,c) for r in range(5) for c in range(5) if r%2 or c%2]
 for r,c in offsets:
  v=padded[:,:,r:r+32,c:c+32];yield v;yield v.flip(-1)

def load(checkpoint,device):
 torch.set_num_threads(1)
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 model,n=build(ck['config']);model.load_state_dict(ck['model'],strict=True)
 assert n==ck['parameters']==4804512
 model.to(device=device,memory_format=torch.channels_last).eval()
 return model,ck

@torch.inference_mode()
def predict(model,ck,images,device='cuda',view_count=50,batch_size=64):
 mean=torch.tensor(ck['normalization']['mean'],device=device).view(1,3,1,1)
 std=torch.tensor(ck['normalization']['std'],device=device).view(1,3,1,1)
 out=[]
 for start in range(0,len(images),batch_size):
  z=torch.from_numpy(np.array(images[start:start+batch_size],copy=True)).permute(0,3,1,2).to(device).float()/255
  total=torch.zeros(len(z),10,device=device)
  for i,v in enumerate(views(z)):
   if i>=view_count:break
   total+=model(((v-mean)/std).contiguous(memory_format=torch.channels_last)).softmax(1)
  out.append((total/view_count).cpu().numpy())
 return np.concatenate(out)

def export(model,ck,data,output,device='cuda'):
 def rows(p):
  with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
 sample=rows(data/'sample_submission.csv');test=rows(data/'test.csv');lookup={r['id']:r['filename'] for r in test}
 assert len(sample)==len(test)==len(lookup)==2000
 images=[]
 for r in sample:
  with Image.open(data/'test_images'/lookup[r['id']]) as im:
   assert im.mode=='RGB' and im.size==(32,32);images.append(np.array(im))
 p=predict(model,ck,np.stack(images),device)
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['id','label']);w.writerows((r['id'],int(k)) for r,k in zip(sample,p.argmax(1)))
 actual=rows(output)
 assert len(actual)==2000 and [r['id'] for r in actual]==[r['id'] for r in sample]
 assert all(0<=int(r['label'])<=9 for r in actual)
 return {'rows':2000,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'parameters':4804512}

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/final.pt');a.add_argument('--data',type=Path,default=ROOT/'data');a.add_argument('--output',type=Path,default=ROOT/'outputs/submission.csv');a.add_argument('--device',default='cuda');a.add_argument('--validate',action='store_true');args=a.parse_args()
 model,ck=load(args.checkpoint,args.device)
 if args.validate:
  split=np.load(ROOT/'results/split_seed2026.npz')['val'];x=np.load(args.data/'cache/train.npy',mmap_mode='r')[split];y=np.load(args.data/'cache/labels.npy')[split]
  p=predict(model,ck,x,args.device);print(json.dumps({'accuracy':float((p.argmax(1)==y).mean()),'nll':float(-np.log(p[np.arange(len(y)),y]).mean()),'count':len(y)}))
 else:print(json.dumps(export(model,ck,args.data,args.output,args.device)))
