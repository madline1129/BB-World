import pickle
from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


class OccActors(Dataset):
    def __init__(self, manifest, limit=0):
        with open(manifest, 'rb') as f:
            self.rows = pickle.load(f)
        if limit:
            self.rows = self.rows[:limit]

    @staticmethod
    @lru_cache(maxsize=32)
    def load_occ(path):
        with np.load(path) as f:
            a = f['semantics'].astype(np.uint8)
        assert a.shape == (200, 200, 16) and a.max() <= 17
        return a

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        inputs = {'occ': torch.from_numpy(np.stack([self.load_occ(p) for p in r['history_paths']])),
                  'current_to_history': torch.from_numpy(r['current_to_history']),
                  'history_times': torch.from_numpy(r['history_times'])}
        target = {k: torch.from_numpy(r[k]) for k in ['classes', 'centers', 'future', 'valid']}
        return inputs, target


def collate(batch):
    return {k: torch.stack([x[0][k] for x in batch]) for k in batch[0][0]}, [x[1] for x in batch]


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c),nn.GELU(),
                                 nn.Conv2d(c,c,3,padding=1),nn.GroupNorm(8,c))

    def forward(self,x):
        return F.gelu(x+self.net(x))


class Brain(nn.Module):
    """One end-to-end raw Occ network. No actor/box/identity inputs or frozen tokenizer."""
    def __init__(self, width=128, queries=256):
        super().__init__()
        assert queries == 256
        self.embedding = nn.Embedding(19,4)  # 18 is unknown after geometric warp.
        self.spatial = nn.Sequential(nn.Conv2d(64,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU(),Block(64),
                                     nn.Conv2d(64,width,3,stride=2,padding=1),nn.GroupNorm(8,width),nn.GELU(),Block(width))
        self.temporal = nn.Sequential(nn.Conv2d(width*5,width,1),nn.GroupNorm(8,width),nn.GELU(),Block(width),Block(width))
        self.time = nn.Linear(1,width)
        self.position = nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,width))
        self.queries = nn.Parameter(torch.randn(queries,width)*.02)
        layer = nn.TransformerDecoderLayer(width,8,width*4,dropout=.1,batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(layer,4,norm=nn.LayerNorm(width))
        self.class_head = nn.Linear(width,9)
        self.center_head = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,2))
        self.motion_head = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,12))
        nn.init.zeros_(self.center_head[-1].weight); nn.init.zeros_(self.center_head[-1].bias)
        nn.init.zeros_(self.motion_head[-1].weight); nn.init.zeros_(self.motion_head[-1].bias)
        axis = torch.linspace(-37.5,37.5,16)
        self.register_buffer('anchors',torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2))
        axis = torch.arange(50)*1.6-39.2
        self.register_buffer('positions',torch.stack(torch.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)/40)
        x = torch.arange(200)*.4-39.8
        z = torch.arange(16)*.4-.8
        xyz = torch.stack(torch.meshgrid(x,x,z,indexing='ij'),-1)
        self.register_buffer('voxel_xyz',xyz,persistent=False)

    def align(self, occ, current_to_history):
        b,t = occ.shape[:2]
        with torch.autocast(device_type=occ.device.type, enabled=False):
            transform = current_to_history.float().reshape(b*t,4,4)
            points = torch.einsum('bij,xyzj->bxyzi',transform[:,:3,:3],self.voxel_xyz.float()) + transform[:,:3,3,None,None,None].permute(0,2,3,4,1)
            # grid_sample consumes W,H,D = z,y,x; voxel array is X,Y,Z.
            grid = torch.stack(((points[...,2]-2.2)/3.2,points[...,1]/40,points[...,0]/40),-1)
            # +1 reserves zero padding; map padding to a distinct unknown label.
            aligned = F.grid_sample(occ.reshape(b*t,1,200,200,16).float()+1,grid,
                                    mode='nearest',padding_mode='zeros',align_corners=False)[:,0]-1
            aligned = torch.where(aligned < 0,18,aligned).long()
        return aligned

    def forward(self, occ, current_to_history, history_times):
        b,t = occ.shape[:2]
        assert t == 5
        aligned = self.align(occ,current_to_history)
        x = self.embedding(aligned).permute(0,3,4,1,2).reshape(b*t,64,200,200)
        x = self.spatial(x).reshape(b,t,-1,50,50)
        x = x+self.time(history_times[...,None]).unsqueeze(-1).unsqueeze(-1)
        x = self.temporal(x.flatten(1,2)).flatten(2).transpose(1,2)
        memory = x+self.position(self.positions)[None]
        q = self.queries + self.position(self.anchors/40)
        h = self.decoder(q[None].expand(b,-1,-1),memory)
        center = self.anchors[None]+20*torch.tanh(self.center_head(h))
        future = center[:,:,None]+self.motion_head(h).reshape(b,256,6,2).cumsum(2)
        return {'logits':self.class_head(h),'centers':center,'future':future}

    @staticmethod
    def interface(out):
        # p is current actor detection confidence, NOT calibrated future-mode probability.
        confidence = 1-out['logits'].softmax(-1)[...,-1]
        return torch.cat([out['future'].permute(0,2,1,3),confidence[:,None,:,None].expand(-1,6,-1,-1)],-1)


def match(out, targets):
    pairs=[]
    for i,g in enumerate(targets):
        if not len(g['classes']):
            pairs.append((np.array([],dtype=int),np.array([],dtype=int)));continue
        with torch.no_grad():
            cost=torch.cdist(out['centers'][i].float(),g['centers'][:,:2].float(),p=1)/5
            cost=cost-2*out['logits'][i].float().softmax(-1)[:,g['classes']]
            pairs.append(linear_sum_assignment(cost.cpu().numpy()))
    return pairs


def objective(out,targets):
    pairs=match(out,targets)
    labels=torch.full(out['logits'].shape[:2],8,device=out['logits'].device,dtype=torch.long)
    center_sum=out['centers'].sum()*0
    future_sum=out['future'].sum()*0
    displacement_sum=out['future'].sum()*0
    count,nvalid=0,0
    for i,(pi,gi) in enumerate(pairs):
        if len(pi)==0:continue
        g=targets[i];labels[i,pi]=g['classes'][gi]
        center_sum=center_sum+F.smooth_l1_loss(out['centers'][i,pi].float(),g['centers'][gi,:2],reduction='sum')
        valid=g['valid'][gi]
        pred=out['future'][i,pi].float();truth=g['future'][gi,:,:2]
        future_sum=future_sum+F.smooth_l1_loss(pred[valid],truth[valid],reduction='sum')
        disp=pred-out['centers'][i,pi,None].float()
        true_disp=truth-g['centers'][gi,None,:2]
        displacement_sum=displacement_sum+F.smooth_l1_loss(disp[valid],true_disp[valid],reduction='sum')
        count+=len(pi);nvalid+=int(valid.sum())
    weights=out['logits'].new_ones(9,dtype=torch.float32);weights[-1]=.1
    cls=F.cross_entropy(out['logits'].float().transpose(1,2),labels,weight=weights)
    loc=center_sum/max(count*2,1);traj=future_sum/max(nvalid*2,1);motion=displacement_sum/max(nvalid*2,1)
    loss=2*cls+loc+.5*traj+.5*motion
    return loss,{'class':cls.detach(),'center_m':loc.detach(),'future_m':traj.detach(),'motion_m':motion.detach()}
