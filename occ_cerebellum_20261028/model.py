"""Direct semantic Occ decoder conditioned by frozen-brain predictions; no VAE."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from frozen_brain import OccActors


class ForecastData(Dataset):
    def __init__(self,cache,limit=0,ego_source="gt"):
        self.data=torch.load(cache,map_location='cpu',weights_only=False)
        if ego_source not in ('gt', 'predicted'):
            raise ValueError(ego_source)
        self.pose_key = 'predicted_future_to_current' if ego_source == 'predicted' else 'future_to_current'
        if self.pose_key not in self.data:
            raise ValueError('Predicted ego poses are absent; refusing GT fallback')
        self.length=min(limit,len(self.data['tokens'])) if limit else len(self.data['tokens'])
    def __len__(self):return self.length
    def __getitem__(self,i):
        d=self.data
        inputs={'current_occ':torch.from_numpy(OccActors.load_occ(d['current_paths'][i]).copy()),
                'future_to_current':torch.from_numpy(d[self.pose_key][i]),
                'centers':torch.from_numpy(d['centers'][i]),'future':torch.from_numpy(d['future'][i]),
                'confidence':torch.from_numpy(d['confidence'][i])}
        targets=torch.from_numpy(np.stack([OccActors.load_occ(p) for p in d['future_paths'][i]]))
        return inputs,targets


class Residual(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c),nn.SiLU(),
                               nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c))
    def forward(self,x):return F.silu(x+self.net(x))


def block(cin,cout,stride=1):
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,stride=stride),nn.GroupNorm(8,cout),nn.SiLU(),Residual(cout))


class SpatialFiLM(nn.Module):
    def __init__(self,c):
        super().__init__();self.proj=nn.Conv2d(32,2*c,1)
        nn.init.normal_(self.proj.weight,std=.01);nn.init.zeros_(self.proj.bias)
    def forward(self,x,condition):
        gamma,beta=F.interpolate(self.proj(condition),size=x.shape[-2:],mode='bilinear',align_corners=False).chunk(2,1)
        return x*(1+.1*torch.tanh(gamma))+.1*beta


class Cerebellum(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(19,4)
        self.enc0=block(64,32);self.enc1=block(32,64,2);self.enc2=block(64,128,2)
        self.condition=nn.Sequential(nn.Conv2d(4,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU())
        self.pose_time=nn.Sequential(nn.Linear(13,128),nn.SiLU(),nn.Linear(128,256))
        self.film2=SpatialFiLM(128);self.film1=SpatialFiLM(64);self.film0=SpatialFiLM(32)
        self.temporal=nn.Conv3d(128,128,(3,1,1),padding=(1,0,0))
        self.mid=Residual(128)
        self.up1=block(128+64,64);self.up0=block(64+32,32)
        self.output=nn.Conv2d(32,18*16,1)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)
        x=torch.arange(200)*.4-39.8;z=torch.arange(16)*.4-.8
        self.register_buffer('voxel_xyz',torch.stack(torch.meshgrid(x,x,z,indexing='ij'),-1),persistent=False)
        # Centers corresponding to stride-4 CNN cells in the future ego frame.
        a=torch.arange(50)*1.6-39.8
        self.register_buffer('bev_xy',torch.stack(torch.meshgrid(a,a,indexing='ij'),-1),persistent=False)

    def warp_current(self,current_occ,future_to_current):
        """Sample O0 at T_current<-future * future voxel centers. Unknown is 18."""
        b,t=future_to_current.shape[:2]
        with torch.autocast(device_type=current_occ.device.type,enabled=False):
            transform=future_to_current.float().reshape(b*t,4,4)
            xyz=torch.einsum('bij,xyzj->bxyzi',transform[:,:3,:3],self.voxel_xyz.float())
            xyz=xyz+transform[:,:3,3][:,None,None,None,:]
            grid=torch.stack(((xyz[...,2]-2.2)/3.2,xyz[...,1]/40,xyz[...,0]/40),-1)
            source=current_occ[:,None].expand(-1,t,-1,-1,-1).reshape(b*t,1,200,200,16).float()+1
            warped=F.grid_sample(source,grid,mode='nearest',padding_mode='zeros',align_corners=False)[:,0]-1
        return torch.where(warped<0,18,warped).long()

    def motion_raster(self,centers,future,confidence,future_to_current):
        """Confidence-weighted source/target density + source-associated dx/dy, in future ego XY."""
        with torch.autocast(device_type=centers.device.type,enabled=False):
            b,t=future_to_current.shape[:2];n=centers.shape[1]
            inv=torch.linalg.inv(future_to_current.float())
            src=torch.cat([centers.float(),torch.zeros(b,n,1,device=centers.device),torch.ones(b,n,1,device=centers.device)],-1)
            dst=future.float().permute(0,2,1,3)
            dst=torch.cat([dst,torch.zeros(b,t,n,1,device=centers.device),torch.ones(b,t,n,1,device=centers.device)],-1)
            source=torch.einsum('btij,bnj->btni',inv,src)[...,:2]
            target=torch.einsum('btij,btnj->btni',inv,dst)[...,:2]
            weights=(confidence.float()*(confidence>=.5))[:,None,:,None,None]
            def splat(x):
                diff=x[:,:,:,None,None,:]-self.bev_xy[None,None,None]
                return torch.exp(-diff.square().sum(-1)/(2*1.2**2))*weights
            gs=splat(source);gt=splat(target);density=gs.sum(2)
            flow=torch.einsum('btnxy,btnc->btcxy',gs,target-source)/(density[:, :,None].clamp_min(1e-4))
            raster=torch.cat([density[:,:,None].clamp_max(1),gt.sum(2)[:,:,None].clamp_max(1),
                              (flow/10).clamp(-4,4)],2)
        return raster.reshape(b*t,4,50,50)

    def forward(self,current_occ,future_to_current,centers,future,confidence):
        b,t=future_to_current.shape[:2];assert t==6
        warped=self.warp_current(current_occ,future_to_current)
        raster=self.motion_raster(centers,future,confidence,future_to_current)
        cond=self.condition(raster)
        x=self.embedding(warped).permute(0,3,4,1,2).reshape(b*t,64,200,200)
        s0=self.enc0(x);s1=self.enc1(s0);x=self.enc2(s1)
        pose=future_to_current[:,:,:3].float().clone();pose[:,:,:,3]/=40
        times=torch.arange(1,7,device=x.device).float()[None,:,None].expand(b,-1,-1)/6
        gamma,beta=self.pose_time(torch.cat([pose.flatten(2),times],-1)).reshape(b*t,256,1,1).chunk(2,1)
        x=self.film2(x,cond)*(1+.1*torch.tanh(gamma))+.1*beta
        temporal=x.reshape(b,t,128,50,50).permute(0,2,1,3,4)
        x=x+self.temporal(temporal).permute(0,2,1,3,4).reshape(b*t,128,50,50)
        x=self.mid(x)
        x=self.film1(self.up1(torch.cat([F.interpolate(x,size=(100,100),mode='bilinear',align_corners=False),s1],1)),cond)
        x=self.film0(self.up0(torch.cat([F.interpolate(x,size=(200,200),mode='bilinear',align_corners=False),s0],1)),cond)
        logits=self.output(x).reshape(b,t,18,16,200,200).permute(0,1,2,4,5,3).contiguous()
        # Static transport is a deterministic prior, not a future target or pretrained latent.
        prior=logits.new_zeros(b*t,18,200,200,16)
        prior.scatter_(1,warped.clamp_max(17)[:,None],2*(warped<18)[:,None].to(logits.dtype))
        logits=logits+prior.reshape(b,t,18,200,200,16)
        return logits


WEIGHTS=[.0727,.0692,.0838,.0681,.0601,.0741,.0823,.0688,.0773,.0681,.0641,.0527,.0655,.0563,.0558,.0541,.0538,.0468]


def objective(logits,target):
    # The complete 18-class grid is supervised; no camera/lidar visibility mask.
    b,t=target.shape[:2];logits=logits.reshape(b*t,18,-1).float();target=target.reshape(b*t,-1).long()
    weights=torch.tensor(WEIGHTS,device=logits.device)
    ce=F.cross_entropy(logits,target,weight=weights)
    prob=logits.softmax(1)
    predicted=prob.sum((0,2));correct=prob.gather(1,target[:,None])[:,0]
    intersection=torch.zeros(18,device=logits.device).scatter_add(0,target.flatten(),correct.flatten())
    truth=torch.bincount(target.flatten(),minlength=18).float()
    dice=(2*intersection+1e-5)/(predicted+truth+1e-5)
    valid=truth[:17]>0
    semantic_dice=1-dice[:17][valid].mean() if valid.any() else logits.sum()*0
    return ce+.5*semantic_dice,{'ce':ce.detach(),'semantic_dice_loss':semantic_dice.detach()}


@torch.no_grad()
def confusion(pred,target):
    return torch.bincount((target.long()*18+pred.long()).flatten(),minlength=18*18).reshape(18,18)


def summarize(hist):
    values=[]
    for h in hist.double():
        union=h.sum(0)+h.sum(1)-h.diag();iou=h.diag()/union.clamp_min(1)
        present=union[:17]>0;nonzero=present&(iou[:17]>0)
        occupied_intersection=h[:17,:17].sum();occupied_union=h.sum()-h[17,17]
        values.append({'mIoU_corrected':float(iou[:17][present].mean()*100),
                       'mIoU_IIWorld_native':float(iou[:17][nonzero].mean()*100) if nonzero.any() else None,
                       'IoU':float(occupied_intersection/occupied_union.clamp_min(1)*100),
                       'actor_class_mIoU':float(iou[[2,3,4,5,6,7,9,10]].mean()*100),
                       'class_IoU':(iou*100).tolist()})
    summary={'1s':values[0],'2s':values[1],'3s':values[2]}
    summary['average']={k:sum(v[k] for v in values)/3 for k in ['mIoU_corrected','IoU','actor_class_mIoU']}
    native=[v['mIoU_IIWorld_native'] for v in values]
    summary['average']['mIoU_IIWorld_native']=sum(native)/3 if all(x is not None for x in native) else None
    return summary
