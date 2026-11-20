"""Cell-adaptive spatial aggregation for CASAGE."""
import math

import torch
from torch import nn


from .vector_sampling import sample_points


class AdaptiveGrid(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.predict=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.GELU(),nn.Linear(128,12))
        nn.init.zeros_(self.predict[-1].weight);nn.init.zeros_(self.predict[-1].bias)
        self.register_buffer('base_offsets',torch.tensor([[-7.,0.],[7.,0.],[0.,-7.],[0.,7.]]))
        self.gate=nn.Parameter(torch.tensor(-2.))
    def forward(self,q,xy,tokens,counts):
        v=self.predict(q);offset=self.base_offsets+7*torch.tanh(v[:,:8].reshape(-1,4,2).float())
        points=(xy[:,None,:]+offset).clamp(0,223)
        side=math.isqrt(tokens.shape[1]);maps=tokens.transpose(1,2).reshape(len(counts),-1,side,side)
        samples=sample_points(maps,points,counts)
        weights=v[:,8:].float().softmax(-1)
        pooled=(samples*weights.unsqueeze(-1)).sum(1)
        return q+torch.sigmoid(self.gate)*(pooled-q)
