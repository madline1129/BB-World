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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from model import Cerebellum, ForecastData, confusion, objective, summarize


def transfer(batch,device):
    x,y=batch
    return {k:v.to(device,non_blocking=True) for k,v in x.items()},y.to(device,non_blocking=True)


def atomic_save(payload,path):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)


@torch.no_grad()
def evaluate(model,loader,device,baseline=False):
    model.eval();hist=torch.zeros(3,18,18,dtype=torch.int64,device=device)
    base_hist=torch.zeros_like(hist);samples=torch.zeros((),device=device,dtype=torch.int64)
    for batch in loader:
        x,y=transfer(batch,device)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**x)
        pred=logits.argmax(2);del logits
        for j,k in enumerate([1,3,5]):hist[j]+=confusion(pred[:,k],y[:,k])
        if baseline:
            base=model.warp_current(x['current_occ'],x['future_to_current']).reshape_as(y).clamp_max(17)
            for j,k in enumerate([1,3,5]):base_hist[j]+=confusion(base[:,k],y[:,k])
        samples+=len(y)
    dist.all_reduce(hist);dist.all_reduce(samples)
    result={'samples':int(samples),'forecast':summarize(hist.cpu()),'confusion':hist.cpu().tolist()}
    if baseline:
        dist.all_reduce(base_hist);result['static_transport']=summarize(base_hist.cpu())
    return result


def geometry_checks(net,x):
    identity=torch.eye(4,device=x['current_occ'].device)[None,None].expand(len(x['current_occ']),1,-1,-1).clone()
    assert torch.equal(net.warp_current(x['current_occ'],identity),x['current_occ'].long())
    identity[:,:,0,3]=.4
    shifted=net.warp_current(x['current_occ'],identity)
    assert torch.equal(shifted[:,:-1],x['current_occ'][:,1:].long())
    assert torch.all(shifted[:,-1]==18)
    # A translated future ego moves current XY points in the opposite direction.
    centers=torch.zeros(1,1,2,device=identity.device)
    future=torch.zeros(1,1,6,2,device=identity.device);future[:,:,:,0]=3.2
    pose=torch.eye(4,device=identity.device)[None,None].expand(1,6,-1,-1).clone()
    pose[:,:,0,3]=1.6
    raster=net.motion_raster(centers,future,torch.ones(1,1,device=identity.device),pose).reshape(1,6,4,50,50)
    src=raster[0,0,0].argmax();dst=raster[0,0,1].argmax()
    assert int(dst//50-src//50)==2,'Future-coordinate motion direction is incorrect'
    assert torch.count_nonzero(net.motion_raster(centers,future,torch.zeros(1,1,device=identity.device),pose))==0
    # Independent metric oracle: one correct actor, one actor predicted free, one correct free.
    h=confusion(torch.tensor([4,17,17]),torch.tensor([4,4,17]))
    metrics=summarize(h[None].expand(3,-1,-1))['average']
    assert metrics['mIoU_corrected']==50 and metrics['IoU']==50


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--seed',type=int,default=20261028);p.add_argument('--smoke',action='store_true')
    p.add_argument('--ego-source',choices=['gt','predicted'],default='gt');p.add_argument('--max-steps',type=int,default=0);p.add_argument('--resume');a=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',device_id=device)
    random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def emit(kind,**kw):
        if rank==0:
            row={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'kind':kind,**kw}
            print(json.dumps(row),flush=True)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    limit=2*a.batch_size if a.smoke else 0
    train=ForecastData(Path(a.cache)/'train.pt',limit,ego_source=a.ego_source)
    val=ForecastData(Path(a.cache)/'val.pt',4 if a.smoke else 0,ego_source=a.ego_source)
    assert train.data['brain_sha256']==val.data['brain_sha256']
    assert not set(train.data['tokens'])&set(val.data['tokens'])
    sampler=DistributedSampler(train,shuffle=True,seed=a.seed)
    loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    val_loader=DataLoader(Subset(val,list(range(rank,len(val),world))),batch_size=a.batch_size,
                          num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    net=Cerebellum().to(device);model=DDP(net,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    step=start_epoch=start_batch=0;best=-1
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=False)
        assert ck['brain_sha256']==train.data['brain_sha256']
        assert ck['config'].get('ego_source','gt') == a.ego_source, 'Cannot resume across ego protocols'
        net.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        step=ck['step'];start_epoch=ck['epoch'];start_batch=ck['batch']+1;best=ck.get('best',-1)
    config={**vars(a),'world_size':world,'global_batch':world*a.batch_size,'parameters':sum(x.numel() for x in net.parameters()),
            'brain_sha256':train.data['brain_sha256'],'brain_frozen':True,'future_GT_actor_inputs':False,
            'future_ego_pose_condition':True,'future_GT_ego_pose_input':a.ego_source=='gt','no_vae':True,'torch':torch.__version__,
            'metric_protocol':'full grid, no visibility masks, 1/2/3s average; native zero-excluding and corrected mIoU',
            'cache_audit':json.loads((Path(a.cache)/'audit.json').read_text()),
            'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}}
    if rank==0:(out/'config.json').write_text(json.dumps(config,indent=2))
    emit('start',train_windows=len(train),val_windows=len(val),parameters=config['parameters'],global_batch=config['global_batch'])
    fixed=transfer(next(iter(loader)),device) if a.smoke else None
    if a.smoke:
        geometry_checks(net,fixed[0]);net.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):initial=float(objective(net(**fixed[0]),fixed[1])[0])
        emit('geometry_and_metric_checks_passed',initial_loss=initial)
    total=a.max_steps or a.epochs*len(loader);tick=time.monotonic();stop=False
    for epoch in range(start_epoch,a.epochs):
        sampler.set_epoch(epoch);model.train()
        for batch_id,batch in enumerate(loader):
            if epoch==start_epoch and batch_id<start_batch:continue
            step+=1
            scale=1 if a.smoke else min(1,step/200)*(.1+.9*.5*(1+math.cos(math.pi*min(step,total)/total)))
            for group in optimizer.param_groups:group['lr']=a.lr*scale
            x,y=transfer(batch,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(**x);loss,parts=objective(logits,y)
            assert torch.isfinite(loss).item(),f'Nonfinite loss rank={rank} step={step}'
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            if step==1:
                norms={name:float(param.grad.norm()) for name,param in [('embedding',net.embedding.weight),('motion_condition',net.condition[0].weight),('decoder',net.output.weight)]}
                assert all(x>0 for x in norms.values()),norms
                emit('gradient_paths_verified',**norms)
            optimizer.step();del logits
            if step==1 or step%10==0:
                stats=torch.stack([loss.detach(),*parts.values()]).float();dist.all_reduce(stats);stats/=world
                emit('train',epoch=epoch+1,step=step,loss=float(stats[0]),ce=float(stats[1]),dice_loss=float(stats[2]),
                     lr=optimizer.param_groups[0]['lr'],gradient_norm=float(grad),elapsed_s=round(time.monotonic()-tick,2),
                     peak_memory_gb=round(torch.cuda.max_memory_allocated()/1e9,2))
            if rank==0 and (step==20 or step%1000==0):
                atomic_save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,'batch':batch_id,
                             'config':config,'best':best,'brain_sha256':train.data['brain_sha256']},out/'latest.pt')
            if a.max_steps and step>=a.max_steps:stop=True;break
        if not a.smoke:
            metrics=evaluate(net,val_loader,device,baseline=epoch==0)
            assert metrics['samples']==len(val)
            if rank==0:(out/f'validation-epoch{epoch+1:02d}.json').write_text(json.dumps(metrics,indent=2))
            emit('validation',epoch=epoch+1,step=step,samples=metrics['samples'],forecast=metrics['forecast'],static_transport=metrics.get('static_transport'))
            score=metrics['forecast']['average']['mIoU_corrected']
            if score>best:
                best=score
                if rank==0:atomic_save({'model':net.state_dict(),'step':step,'epoch':epoch,'config':config,'brain_sha256':train.data['brain_sha256']},out/'best_miou.pt')
        if rank==0:
            atomic_save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,'batch':batch_id,
                         'config':config,'best':best,'brain_sha256':train.data['brain_sha256']},out/'latest.pt')
        dist.barrier()
        if stop:break
    if a.smoke:
        net.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=net(**fixed[0]);final=float(objective(pred,fixed[1])[0])
            x0={**fixed[0],'confidence':torch.zeros_like(fixed[0]['confidence'])}
            ablated=net(**x0)
            sensitivity=float((pred.float()-ablated.float()).abs().mean())
            assert sensitivity>1e-6,'Motion condition has no effect'
            expected=pred.argmax(2).cpu();del pred,ablated
        assert final<.8*initial,('No small-data learning',initial,final)
        digest=hashlib.sha256(b''.join(x.detach().cpu().float().numpy().tobytes() for x in net.parameters())).hexdigest()
        hashes=[None]*world;dist.all_gather_object(hashes,digest);assert len(set(hashes))==1
        ck=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        restored=Cerebellum().to(device);restored.load_state_dict(ck['model']);restored.eval()
        opt2=torch.optim.AdamW(restored.parameters());opt2.load_state_dict(ck['optimizer'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actual=restored(**fixed[0]).argmax(2).cpu()
        assert torch.equal(expected,actual),'Restored checkpoint changes semantic predictions'
        del restored,opt2,ck
        metrics=evaluate(net,val_loader,device,baseline=True)
        if rank==0:(out/'smoke_validation.json').write_text(json.dumps(metrics,indent=2))
        emit('smoke_passed',initial_loss=initial,final_loss=final,condition_sensitivity=sensitivity,ddp_hashes=hashes,checkpoint_restore=True)
    emit('complete',step=step);dist.destroy_process_group()


if __name__=='__main__':main()
