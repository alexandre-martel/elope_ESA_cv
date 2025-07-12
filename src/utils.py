import numpy as np
import torch

def events_to_image(events, shape):
    """
    Crée une image en niveaux de gris à partir des événements.
    Accumule séparément les polarités +1 / -1, puis normalise.
    """
    if events.numel() == 0:
        return torch.zeros(shape, dtype=torch.float32, device=events.device)

    x = events[:, 0].long()
    y = events[:, 1].long()
    p = events[:, 2]

    x = torch.clamp(x, 0, shape[1] - 1)
    y = torch.clamp(y, 0, shape[0] - 1)

    pos_mask = (p > 0)
    neg_mask = ~pos_mask

    # Indices plats pour accumulation
    indices_pos = y[pos_mask] * shape[1] + x[pos_mask]
    indices_neg = y[neg_mask] * shape[1] + x[neg_mask]

    img_pos = torch.zeros(shape[0] * shape[1], dtype=torch.float32, device=events.device)
    img_neg = torch.zeros_like(img_pos)

    img_pos.index_add_(0, indices_pos, torch.ones_like(indices_pos, dtype=torch.float32))
    img_neg.index_add_(0, indices_neg, torch.ones_like(indices_neg, dtype=torch.float32))

    img_pos = img_pos.view(shape)
    img_neg = img_neg.view(shape)

    # Fusion avec normalisation simple : [0, 1] float image
    img = img_pos - img_neg
    img = img - img.min()
    img = img / (img.max() + 1e-6)  # Évite la division par zéro

    return img

def get_closest_range(range_meter, target_ts):
    """
    Trouve la valeur de range la plus proche temporellement de target_ts (en secondes), vectorisé.
    """
    idx = torch.argmin(torch.abs(range_meter[:, 0] - target_ts))
    return range_meter[idx, 1]

def events_to_4ch_image(events, shape, t_start, t_end, device):
    H, W = shape
    if events.shape[0] == 0:
        return torch.zeros((4, H, W), device=device)

    x = events[:, 0].long().clamp(0, W - 1)
    y = events[:, 1].long().clamp(0, H - 1)
    p = events[:, 2]
    t = events[:, 3]

    # Masks
    pos_mask = (p > 0)
    neg_mask = ~pos_mask

    # Init
    img_pos = torch.zeros(H, W, device=device)
    img_neg = torch.zeros(H, W, device=device)
    ts_pos = torch.zeros(H, W, device=device)
    ts_neg = torch.zeros(H, W, device=device)

    # Pos events
    idx_pos = y[pos_mask] * W + x[pos_mask]
    img_pos_flat = img_pos.view(-1)
    img_pos_flat.index_add_(0, idx_pos, torch.ones_like(idx_pos, dtype=torch.float32))

    ts_flat_pos = ts_pos.view(-1)
    ts_update_pos = torch.zeros_like(ts_flat_pos)
    ts_update_pos.index_put_((idx_pos,), t[pos_mask], accumulate=False)
    ts_pos = torch.maximum(ts_pos, ts_update_pos.view(H, W))

    # Neg events
    idx_neg = y[neg_mask] * W + x[neg_mask]
    img_neg_flat = img_neg.view(-1)
    img_neg_flat.index_add_(0, idx_neg, torch.ones_like(idx_neg, dtype=torch.float32))

    ts_flat_neg = ts_neg.view(-1)
    ts_update_neg = torch.zeros_like(ts_flat_neg)
    ts_update_neg.index_put_((idx_neg,), t[neg_mask], accumulate=False)
    ts_neg = torch.maximum(ts_neg, ts_update_neg.view(H, W))

    # Normalize timestamps
    ts_range = (t_end - t_start) + 1e-6
    ts_pos = (ts_pos - t_start) / ts_range
    ts_neg = (ts_neg - t_start) / ts_range

    return torch.stack([img_pos, img_neg, ts_pos, ts_neg], dim=0)