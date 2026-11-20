"""Two-part CLS targets on the transformed expression scale."""
import torch
import torch.nn.functional as F


def targets(cell_values, counts):
    counts = torch.as_tensor(counts, device=cell_values.device, dtype=torch.long)
    index = torch.repeat_interleave(torch.arange(len(counts), device=cell_values.device), counts,
                                   output_size=len(cell_values))
    shape = (len(counts), cell_values.shape[1])
    totals = torch.zeros(shape, device=cell_values.device, dtype=torch.float32)
    positives = torch.zeros_like(totals)
    totals.index_add_(0, index, cell_values.float())
    positives.index_add_(0, index, (cell_values > 0).float())
    return totals, positives / counts[:, None].clamp_min(1), totals / positives.clamp_min(1), positives > 0


def terms(detection_logits, intensity_logits, fraction, positive_mean, available, has_positive):
    detection = F.binary_cross_entropy_with_logits(
        detection_logits[available].float(), fraction[available], reduction='none')
    selected = available & has_positive
    intensity = F.huber_loss(F.softplus(intensity_logits[selected].float()),
                             positive_mean[selected], delta=1.0, reduction='none')
    return detection, intensity
