import torch
import numpy as np

IGNORE_INDEX = 255

def remap_mask_array(mask_array, mapping_dict, ignore_value=IGNORE_INDEX):
    out = np.ones_like(mask_array, dtype=np.uint8) * ignore_value
    for orig, new in mapping_dict.items():
        out[mask_array == orig] = new
    return out

def compute_miou(preds, labels, num_classes, ignore_index=IGNORE_INDEX):
    preds_flat = preds.view(-1).cpu()
    labels_flat = labels.view(-1).cpu()
    mask = labels_flat != ignore_index
    preds_flat = preds_flat[mask]
    labels_flat = labels_flat[mask]

    if preds_flat.numel() == 0:
        return 0.0

    k = (labels_flat * num_classes + preds_flat).to(torch.long)
    conf = torch.bincount(k, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
    tp = conf.diag().float()
    gt = conf.sum(dim=1).float()
    pred = conf.sum(dim=0).float()
    union = gt + pred - tp
    iou = (tp / union).nan_to_num(0.0)
    miou = iou.mean().item()
    return miou
