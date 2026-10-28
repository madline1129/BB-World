import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from brain import Brain, OccActors, collate, objective, ego_objective


def transfer(batch, device):
    x, ys = batch
    return ({k:v.to(device,non_blocking=True) for k,v in x.items()},
            [{k:v.to(device,non_blocking=True) for k,v in y.items()} for y in ys])


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()
    # GT, predictions, TP, trajectory distance/points, final distance/points,
    # static baseline distance/points, moving trajectory distance/points.
    stats=torch.zeros(16,device=device,dtype=torch.float64)
    for batch in loader:
        x,targets=transfer(batch,device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(**x)
        if 'ego_future_to_current' in out:
            pred = out['ego_future_to_current'].float()
            truth = torch.stack([g['ego_future_to_current'] for g in targets])
            error = (pred[..., :3, 3] - truth[..., :3, 3]).norm(dim=-1)
            relative = pred[..., :3, :3].transpose(-1, -2) @ truth[..., :3, :3]
            angle = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1))
            stats[11] += error.sum(); stats[12] += error.numel()
            stats[13] += error[:, -1].sum(); stats[14] += len(error)
            stats[15] += angle.sum() * (180 / math.pi)
        for i,g in enumerate(targets):
            prob=out['logits'][i].float().softmax(-1)
            conf=1-prob[:,-1];cls=prob[:,:8].argmax(-1)
            selected=torch.where(conf>=.5)[0]
            stats[0]+=len(g['classes']);stats[1]+=len(selected)
            if not len(selected) or not len(g['classes']):continue
            distance=torch.cdist(out['centers'][i,selected].float(),g['centers'][:,:2])
            legal=(distance<=2)&(cls[selected,None]==g['classes'][None])
            pi,gi=linear_sum_assignment((distance+1e5*(~legal)).cpu().numpy())
            good=legal[pi,gi].cpu().numpy();pi,gi=pi[good],gi[good]
            if not len(pi):continue
            stats[2]+=len(pi)
            pred=out['future'][i,selected[pi]].float();truth=g['future'][gi,:,:2];mask=g['valid'][gi]
            error=(pred-truth).norm(dim=-1)
            baseline=(g['centers'][gi,None,:2]-truth).norm(dim=-1)
            stats[3]+=error[mask].sum();stats[4]+=mask.sum()
            stats[5]+=error[:,-1][mask[:,-1]].sum();stats[6]+=mask[:,-1].sum()
            stats[7]+=baseline[mask].sum();stats[8]+=mask.sum()
            moving=(baseline[:,-1]>1)&mask[:,-1]
            mm=mask&moving[:,None]
            stats[9]+=error[mm].sum();stats[10]+=mm.sum()
    dist.all_reduce(stats)
    s=stats.tolist()
    ratio=lambda a,b: a/b if b else None
    result = {'gt_actors':s[0],'predicted_actors':s[1],'matched_actors':s[2],
            'recall_at_2m_conf05':ratio(s[2],s[0]),'precision_at_2m_conf05':ratio(s[2],s[1]),
            'matched_ADE_m':ratio(s[3],s[4]),'matched_FDE_3s_m':ratio(s[5],s[6]),
            'matched_oracle_current_center_static_ADE_m':ratio(s[7],s[8]),'matched_moving_ADE_m':ratio(s[9],s[10])}
    if s[12]:
        result.update(ego_ADE_xyz_m=ratio(s[11],s[12]), ego_FDE_xyz_3s_m=ratio(s[13],s[14]),
                      ego_rotation_error_deg=ratio(s[15],s[12]))
    return result


def atomic_save(payload,path):
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifests',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--max-steps',type=int,default=0)
    parser.add_argument('--resume')
    parser.add_argument('--ego-cache', help='Existing cerebellum cache directory; future poses are labels only')
    parser.add_argument('--init', help='Actor brain checkpoint to initialize, without optimizer state')
    parser.add_argument('--ego-only', action='store_true', help='Freeze actor network and train only new ego head')
    args=parser.parse_args()
    if args.ego_only and not (args.ego_cache and (args.init or args.resume)):
        parser.error('--ego-only requires --ego-cache and either --init or --resume')
    if args.resume and args.init:
        parser.error('--resume and --init are mutually exclusive')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',device_id=device)
    random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    def emit(kind,**kw):
        if rank==0:
            row={'time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'kind':kind,**kw}
            print(json.dumps(row),flush=True)
            with (outdir/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    train=OccActors(Path(args.manifests)/'train.pkl',limit=4 if args.smoke else 0, ego_cache=Path(args.ego_cache)/'train.pt' if args.ego_cache else None)
    sampler=DistributedSampler(train,shuffle=True,seed=args.seed)
    loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,
                      collate_fn=collate,pin_memory=True,persistent_workers=args.workers>0)
    val=OccActors(Path(args.manifests)/'val.pkl',limit=8 if args.smoke else 0, ego_cache=Path(args.ego_cache)/'val.pt' if args.ego_cache else None)
    # Strided unique validation subset, no DistributedSampler padding/duplication.
    val_loader=DataLoader(Subset(val,list(range(rank,len(val),world))),batch_size=args.batch_size,
                          num_workers=args.workers,collate_fn=collate,pin_memory=True,persistent_workers=args.workers>0)
    net=Brain(predict_ego=bool(args.ego_cache)).to(device)
    if args.init:
        initial = torch.load(args.init, map_location='cpu', weights_only=False)
        missing, unexpected = net.load_state_dict(initial['model'], strict=False)
        assert not unexpected and all(k.startswith('ego_head.') for k in missing), (missing, unexpected)
    if args.ego_only:
        net.requires_grad_(False); net.ego_head.requires_grad_(True)
    def loss_fn(prediction, targets):
        if args.ego_only:
            return ego_objective(prediction['ego_future_to_current'], torch.stack([g['ego_future_to_current'] for g in targets]))
        loss, parts = objective(prediction, targets)
        if args.ego_cache:
            ego_loss, ego_parts = ego_objective(prediction['ego_future_to_current'], torch.stack([g['ego_future_to_current'] for g in targets]))
            loss = loss + ego_loss; parts.update(ego_parts)
        return loss, parts
    model=DDP(net,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    step=0;start_epoch=0;start_batch=0;best_recall=-1.;best_ego=float('inf')
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False)
        net.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        step=ck['step'];start_epoch=ck['epoch'];start_batch=ck['batch']+1
        best_recall=ck.get('best_recall',-1.)
        best_ego=ck.get('best_ego',float('inf'))
    config={**vars(args),'world_size':world,'global_batch_size':args.batch_size*world,
            'parameters':sum(p.numel() for p in net.parameters()),'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'supervision':'extra original nuScenes actor boxes/instances',
            'input':'five GT Occ frames plus history-only ego transforms/timestamps',
            'p_definition':'current actor detection confidence; not future uncertainty',
            'manifest_audit':json.loads((Path(args.manifests)/'audit.json').read_text()),
            'future_GT_ego_input':False,'future_ego_supervision':bool(args.ego_cache),
            'ego_horizons_s':[.5,1.,1.5,2.,2.5,3.] if args.ego_cache else None,
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    if rank==0:(outdir/'config.json').write_text(json.dumps(config,indent=2))
    emit('start',train_windows=len(train),val_windows=len(val),parameters=config['parameters'],global_batch=config['global_batch_size'])
    fixed=transfer(next(iter(loader)),device) if args.smoke else None
    initial_loss=None
    if args.smoke:
        net.eval()
        with torch.no_grad():initial_loss=float(loss_fn(net(**fixed[0]),fixed[1])[0])
        aligned=net.align(fixed[0]['occ'][:,-1:],fixed[0]['current_to_history'][:,-1:])
        assert torch.equal(aligned,fixed[0]['occ'][:,-1].long()),'Identity warp corrupted voxel axes'
        # +0.4m current->history x transform samples the next source x voxel.
        transform=fixed[0]['current_to_history'][:,-1:].clone();transform[:,:,0,3]=.4
        shifted=net.align(fixed[0]['occ'][:,-1:],transform)
        assert torch.equal(shifted[:,:-1],fixed[0]['occ'][:,-1,1:].long()),'Translation warp axis/sign error'
        assert torch.all(shifted[:,-1]==18),'Unknown padding conflated with free space'
        emit('geometry_checks_passed',initial_loss=initial_loss)
    total_steps=args.max_steps or args.epochs*len(loader)
    tick=time.monotonic();stop=False
    for epoch in range(start_epoch,args.epochs):
        sampler.set_epoch(epoch);model.train()
        if args.ego_only:
            net.eval(); net.ego_head.train()
        for batch_id,batch in enumerate(loader):
            if epoch==start_epoch and batch_id<start_batch:continue
            step+=1
            warmup=1 if args.smoke else min(1.,step/200)
            cosine=1 if args.smoke else .1+.9*.5*(1+math.cos(math.pi*min(step,total_steps)/total_steps))
            for group in optimizer.param_groups:group['lr']=args.lr*warmup*cosine
            x,targets=transfer(batch,device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(**x);loss,parts=loss_fn(prediction,targets)
            assert torch.isfinite(loss).item(),f'Nonfinite loss rank={rank} step={step}'
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            if step==1:
                embed_grad=float((net.ego_head.net[-1].weight if args.ego_only else net.embedding.weight).grad.norm())
                assert embed_grad>0,'Occ encoder receives no gradient'
                emit('encoder_gradient_verified',norm=embed_grad)
            optimizer.step()
            if step==1 or step%10==0:
                values=torch.stack([loss.detach(),*parts.values()]).float();dist.all_reduce(values);values/=world
                emit('train',epoch=epoch+1,step=step,loss=float(values[0]),**{k:float(v) for k,v in zip(parts,values[1:])},
                     grad_norm=float(grad),lr=optimizer.param_groups[0]['lr'],elapsed_s=round(time.monotonic()-tick,2),
                     peak_memory_gb=round(torch.cuda.max_memory_allocated()/1e9,2))
            if rank==0 and (step==20 or step%500==0):
                atomic_save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,'batch':batch_id,'config':config},outdir/'latest.pt')
            if args.max_steps and step>=args.max_steps:stop=True;break
        if not args.smoke:
            metrics=evaluate(net,val_loader,device);emit('validation',epoch=epoch+1,step=step,**metrics)
            if args.ego_cache and metrics['ego_ADE_xyz_m'] < best_ego:
                best_ego = metrics['ego_ADE_xyz_m']
                if rank == 0:
                    atomic_save({'model':net.state_dict(),'config':config,'epoch':epoch,'step':step,'best_ego':best_ego},outdir/'best_ego.pt')
            if metrics['recall_at_2m_conf05']>best_recall:
                best_recall=metrics['recall_at_2m_conf05']
                if rank==0:
                    atomic_save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,'batch':batch_id,'config':config,'best_recall':best_recall,'best_ego':best_ego},outdir/'best_recall.pt')
        if rank==0:
            atomic_save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,'batch':batch_id,'config':config,'best_recall':best_recall,'best_ego':best_ego},outdir/'latest.pt')
        dist.barrier()
        if stop:break
    if args.smoke:
        net.eval()
        with torch.no_grad():
            final_prediction=net(**fixed[0]);final_loss=float(loss_fn(final_prediction,fixed[1])[0])
        assert final_loss<initial_loss*.8,('Failed small-data learning check',initial_loss,final_loss)
        checksum=hashlib.sha256(b''.join(p.detach().cpu().float().numpy().tobytes() for p in net.parameters())).hexdigest()
        hashes=[None]*world;dist.all_gather_object(hashes,checksum)
        assert len(set(hashes))==1,('DDP parameters diverged',hashes)
        checkpoint=torch.load(outdir/'latest.pt',map_location=device,weights_only=False)
        restored=Brain(predict_ego=bool(args.ego_cache)).to(device);restored.load_state_dict(checkpoint['model']);restored.eval()
        restored_optimizer=torch.optim.AdamW(restored.parameters());restored_optimizer.load_state_dict(checkpoint['optimizer'])
        with torch.no_grad():reloaded=restored(**fixed[0])
        assert all(torch.equal(reloaded[k],final_prediction[k]) for k in reloaded),'Checkpoint does not reproduce predictions'
        assert Brain.interface(reloaded).shape==(len(fixed[1]),6,256,3)
        emit('smoke_passed',initial_loss=initial_loss,final_loss=final_loss,ddp_hashes=hashes,checkpoint_restore=True,
             validation=evaluate(net,val_loader,device))
    emit('complete',step=step)
    dist.destroy_process_group()


if __name__=='__main__':main()
