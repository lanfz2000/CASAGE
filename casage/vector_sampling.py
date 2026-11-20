"""Padded batched sampling; pixel-center-aligned token sampling, explicit nearest-center boundary rule."""
import torch
from torch.nn import functional as F

def cell_rows(counts,device):
    total=sum(counts)
    bid=torch.tensor([b for b,n in enumerate(counts) for _ in range(n)],device=device,dtype=torch.long)
    rid=torch.tensor([j for n in counts for j in range(n)],device=device,dtype=torch.long)
    return bid,rid,total,max(counts,default=0)

def sample_points(maps,xy,counts,img_size=224):
    bid,rid,total,max_n=cell_rows(counts,xy.device)
    if total==0:return maps.new_empty((0,xy.shape[1],maps.shape[1]),dtype=torch.float32)
    grid=torch.zeros(len(counts),max_n,xy.shape[1],2,device=xy.device,dtype=torch.float32)
    # Pixel-index convention: patch j covers pixels14*j..14*j+13,
    # so its receptive-field center is14*j+6.5. Clamp beyond the first/last
    # token center to the nearest available central token (no added halo readout).
    side_h,side_w=maps.shape[-2:]
    size=xy.new_tensor([side_w,side_h],dtype=torch.float32)
    assert side_h>1 and side_w>1
    token_index=((xy.float()+.5)/img_size*size-.5)
    token_index=torch.minimum(torch.maximum(token_index,torch.zeros_like(token_index)),size-1)
    grid[bid,rid]=2*token_index/(size-1)-1
    sampled=F.grid_sample(maps.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=True)
    return sampled.permute(0,2,3,1)[bid,rid]
