"""Create one inference model from the two fixed final Raw endpoints."""
import argparse,hashlib
from pathlib import Path
import torch
from model_round7 import build

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists()
 first,second=[torch.load(f,map_location='cpu',weights_only=True) for f in [a.first,a.second]]
 assert first['normalization']==second['normalization'] and first['split_sha256']==second['split_sha256']
 assert all(first['config'][k]==second['config'][k] for k in first['config'] if k not in ['name','seed'])
 for k,v in second['model'].items():
  if v.is_floating_point():second['model'][k]=(first['model'][k]+v)*.5
 second.update(weight_type='two_seed_fixed_weight_average',validation=None,processing={'source_sha256':[hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.first,a.second]],'floating_tensors':'equal average','integer_buffers':'second branch'})
 model,count=build(second['config']);model.load_state_dict(second['model'],strict=True);assert count==4804512
 a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(second,a.output)
