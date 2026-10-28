"""Freeze epoch-24 brain and cache predictions only; future ego poses are permitted conditions."""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from frozen_brain import Brain, OccActors
from brain_prepare_reference import pose, sha256


class History(Dataset):
    def __init__(self,rows,indices):self.rows,self.indices=rows,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        idx=self.indices[i];r=self.rows[idx]
        return idx,{'occ':torch.from_numpy(np.stack([OccActors.load_occ(p) for p in r['history_paths']])),
                    'current_to_history':torch.from_numpy(r['current_to_history']),
                    'history_times':torch.from_numpy(r['history_times'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--brain',required=True);p.add_argument('--manifests',required=True)
    p.add_argument('--predict-ego',action='store_true');p.add_argument('--metadata-cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',device_id=device)
    out=Path(a.output)
    assert not (out/'train.pt').exists() and not (out/'val.pt').exists(), 'Use a fresh cache directory'
    out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.brain,map_location='cpu',weights_only=False)
    brain_class = Brain
    if a.predict_ego:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'occ_actor_brain_20261028'))
        from brain import Brain as EgoBrain
        brain_class = EgoBrain
        assert any(k.startswith('ego_head.') for k in ck['model']), 'Checkpoint has no trained ego head'
    else:
        assert ck['epoch']==23 and ck['step']==31704
    model=(brain_class(predict_ego=True) if a.predict_ego else brain_class()).to(device);model.load_state_dict(ck['model']);model.eval().requires_grad_(False)
    report={'brain_sha256':sha256(a.brain),'brain_epoch':ck['epoch']+1,'brain_step':ck['step'],
            'future_ego_pose_condition':not a.predict_ego,'ego_source':'predicted' if a.predict_ego else 'gt','gt_actor_inputs':False,'split':{}}
    for split,meta in [('train','m0.bin'),('val','m1.bin')]:
        rows=pickle.load(open(Path(a.manifests)/(split+'.pkl'),'rb'))
        indices=list(range(rank,len(rows),world));loader=DataLoader(History(rows,indices),batch_size=16,num_workers=4,pin_memory=True)
        records=[]
        with torch.inference_mode():
            for batch,(ids,x) in enumerate(loader):
                x={k:v.to(device,non_blocking=True) for k,v in x.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(**x)
                centers=pred['centers'].float().cpu().numpy();future=pred['future'].float().cpu().numpy()
                conf=(1-pred['logits'].float().softmax(-1)[...,-1]).cpu().numpy()
                ego = pred['ego_future_to_current'].float().cpu().numpy() if a.predict_ego else None
                for j,idx in enumerate(ids.tolist()):
                    records.append((idx,centers[j],future[j],conf[j],ego[j] if ego is not None else None))
                if rank==0 and batch%100==0:print(json.dumps({'split':split,'batch':batch,'total_batches':len(loader)}),flush=True)
        torch.save(records,out/f'{split}.rank{rank}.pt');dist.barrier()
        if rank==0:
            full=[]
            for r in range(world):full.extend(torch.load(out/f'{split}.rank{r}.pt',weights_only=False))
            full.sort(key=lambda x:x[0]);assert [r[0] for r in full]==list(range(len(rows)))
            infos=pickle.load(open(Path(a.metadata_cache)/meta,'rb'))['infos']
            frames={f['token']:f for scene in infos.values() for f in scene}
            current_paths=[];future_paths=[];future_to_current=[];tokens=[]
            for row in rows:
                root=Path(row['history_paths'][-1]).parent.parent
                current_paths.append(row['history_paths'][-1]);tokens.append(row['token'])
                paths=[str(root/t/'labels.npz') for t in row['future_tokens']]
                assert all(Path(x).is_file() for x in paths)
                future_paths.append(paths)
                inv=np.linalg.inv(pose(frames[row['token']]))
                future_to_current.append(np.stack([inv@pose(frames[t]) for t in row['future_tokens']]).astype(np.float32))
            data={'current_paths':current_paths,'future_paths':future_paths,'tokens':tokens,
                  'future_to_current':np.stack(future_to_current),
                  'centers':np.stack([x[1] for x in full]),'future':np.stack([x[2] for x in full]),
                  'confidence':np.stack([x[3] for x in full]),'brain_sha256':report['brain_sha256']}
            if a.predict_ego:
                data['predicted_future_to_current'] = np.stack([x[4] for x in full])
                assert np.isfinite(data['predicted_future_to_current']).all()
            data['ego_source'] = report['ego_source']
            assert all(np.isfinite(data[k]).all() for k in ['centers','future','confidence','future_to_current'])
            torch.save(data,out/f'{split}.pt')
            report['split'][split]={'windows':len(rows),'cache_sha256':sha256(out/f'{split}.pt'),
                                     'source_manifest_sha256':sha256(Path(a.manifests)/(split+'.pkl'))}
            (out/'audit.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report['split'][split]),flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
