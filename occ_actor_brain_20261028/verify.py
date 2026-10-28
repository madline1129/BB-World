"""Checks label geometry, category coverage, masks, and target isolation."""
import argparse
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from brain import Brain, objective
from prepare import actor_class, pose


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifests',required=True);args=p.parse_args()
    root=Path(args.manifests)
    frame={'ego2global_rotation':[2**-.5,0,0,2**-.5],'ego2global_translation':[10,20,0]}
    global_point=np.array([10,22,1,1])
    assert np.allclose((np.linalg.inv(pose(frame))@global_point)[:3],[2,0,1])
    assert actor_class('vehicle.construction')==4
    assert actor_class('human.pedestrian.personal_mobility') is None
    assert list(inspect.signature(Brain.forward).parameters)==['self','occ','current_to_history','history_times']
    report={'synthetic_SE3_passed':True,'model_input_has_no_targets':True}
    for split in ['train','val']:
        rows=pickle.load((root/(split+'.pkl')).open('rb'))
        classes=set();valid=total=0
        for r in rows:
            n=len(r['classes']);classes.update(r['classes'].tolist())
            assert len(r['instance_ids'])==len(set(r['instance_ids']))==n
            assert len(set(r['history_tokens']+r['future_tokens']))==11
            assert not set(r['history_tokens'])&set(r['future_tokens'])
            assert np.allclose(r['current_to_history'][-1],np.eye(4),atol=1e-5)
            assert r['future'].shape==(n,6,3) and r['valid'].shape==(n,6)
            assert np.isfinite(r['future'][r['valid']]).all()
            valid+=r['valid'].sum();total+=r['valid'].size
        assert classes==set(range(8)),(split,classes)
        report[split]={'windows':len(rows),'all_8_classes_present':True,'valid_future_fraction':float(valid/total)}
    # Invalid future values must have zero effect on the objective and gradient.
    torch.manual_seed(1)
    out={'logits':torch.randn(1,4,9,requires_grad=True),'centers':torch.randn(1,4,2,requires_grad=True),
         'future':torch.randn(1,4,6,2,requires_grad=True)}
    g={'classes':torch.tensor([0,2]),'centers':torch.randn(2,3),'future':torch.randn(2,6,3),
       'valid':torch.tensor([[1,1,1,0,0,0],[1,1,1,1,1,1]],dtype=torch.bool)}
    loss=objective(out,[g])[0]
    grad=torch.autograd.grad(loss,out['future'],retain_graph=True)[0]
    other={k:v.clone() for k,v in g.items()};other['future'][~other['valid']]=1e9
    loss2=objective(out,[other])[0]
    grad2=torch.autograd.grad(loss2,out['future'])[0]
    assert torch.equal(loss,loss2) and torch.equal(grad,grad2)
    # Annotation iteration order cannot change set matching supervision.
    perm={k:v.flip(0) for k,v in g.items()}
    assert torch.allclose(loss,objective(out,[perm])[0])
    report.update(missing_label_mask_passed=True,actor_permutation_passed=True)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
