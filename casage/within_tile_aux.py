"""Loss-only cell-relative supervision from already supplied training targets."""
import torch
from torch.nn import functional as F


def auxiliary_values(pred, target, mask, counts, mode):
    """Return selected Huber elements; the caller applies the global denominator.

    Inputs are flattened real cells. Both arms use the same provided-target/global
    nonzero mask and require >=2 valid cells per tile/gene. Predictions are the
    raw head outputs; the absolute regression loss also uses raw outputs.
    No operation here runs in the prediction path or consumes random numbers.
    """
    assert mode in ('centered', 'uncentered')
    assert pred.shape == target.shape == mask.shape and mask.dtype == torch.bool
    assert sum(counts) == len(pred)
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = pred.float(), target.float()
        tile = torch.repeat_interleave(torch.arange(len(counts), device=p.device),
                                       torch.as_tensor(counts, device=p.device))
        shape = (len(counts), p.shape[1])
        valid_counts = p.new_zeros(shape).index_add(0, tile, mask.float())
        eligible = mask & (valid_counts[tile] >= 2)
        if mode == 'centered':
            # Weighted means ensure absent labels receive zero gradients even if
            # availability were to differ between cells in a future dataset.
            ps = p.new_zeros(shape).index_add(0, tile, p.masked_fill(~mask, 0))
            ts = t.new_zeros(shape).index_add(0, tile, t.masked_fill(~mask, 0))
            divisor = valid_counts.clamp_min(1)
            p = p - (ps / divisor)[tile]
            t = t - (ts / divisor)[tile]
        return F.huber_loss(p[eligible], t[eligible], delta=1., reduction='none')
