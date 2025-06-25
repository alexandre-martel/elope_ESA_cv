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