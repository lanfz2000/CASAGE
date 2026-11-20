import torch
class FullBatchShard:
 def __init__(self,n,rank,world=2,global_batch=64):
  self.n=n;self.rank=rank;self.world=world;self.global_batch=global_batch;self.epoch=0
 def set_epoch(self,epoch):self.epoch=epoch
 def __len__(self):return (self.n+self.global_batch-1)//self.global_batch
 def __iter__(self):
  ids=torch.randperm(self.n,generator=torch.Generator().manual_seed(self.epoch)).tolist()
  for s in range(0,self.n,self.global_batch):yield ids[s:s+self.global_batch][self.rank::self.world]
