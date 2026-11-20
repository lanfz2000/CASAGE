"""Training-label table availability, not physical assay membership or a model input."""
import torch
def available_for_loss(batch, counts, idxs, device):
 available=batch['target_available'].to(device=device,dtype=torch.bool,non_blocking=True)
 assert available.ndim==2 and available.shape[0]==len(counts)
 patch=available[:,idxs]
 cells=torch.repeat_interleave(patch,torch.as_tensor(counts,device=device),dim=0)
 return cells,patch
