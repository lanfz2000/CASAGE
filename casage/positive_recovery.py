"""Raw-logit absolute-expression regression for all measured targets.
Legacy function name and config switch retained solely for caller compatibility.
Masking/global normalization are applied by the caller; inference remains ReLU.
"""
import torch
import torch.nn.functional as F
def positive_recovery_errors(raw, target):
    with torch.autocast(device_type=raw.device.type, enabled=False):
        return F.huber_loss(raw.float(), target.float(), delta=1.0, reduction='none')
