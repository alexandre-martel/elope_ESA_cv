import numpy as np
import torch

def events_to_voxel_grid(events, shape=(200, 200), bins=5, device='cpu'):
    """
    Convert event stream to voxel grid using PyTorch (GPU-compatible, no loops).

    Args:
        events: torch.Tensor (N, 4) with columns (x, y, p, t)
        shape: tuple (H, W)
        bins: number of temporal bins
        device: 'cpu' or 'cuda'

    Returns:
        voxel: torch.Tensor of shape (bins, H, W)
    """
    H, W = shape
    events = events.to(device)

    x = events[:, 0].long()
    y = events[:, 1].long()
    p = events[:, 2].float()
    t = events[:, 3].float()

    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, p, t = x[mask], y[mask], p[mask], t[mask]

    t_min = torch.min(t)
    t_max = torch.max(t)
    dt = (t_max - t_min) / bins if t_max > t_min else torch.tensor(1.0, device=device)
    b = ((t - t_min) / dt).floor().long().clamp(max=bins - 1)

    polarity = torch.where(p > 0, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))
    voxel = torch.zeros((bins, H, W), dtype=torch.float32, device=device)

    indices = b * (H * W) + y * W + x
    updates = polarity

    voxel = voxel.view(-1)
    voxel.index_add_(0, indices, updates)
    voxel = voxel.view(bins, H, W)

    return voxel

def get_closest_range(rangemeter: torch.Tensor, t_query: float) -> torch.Tensor:
    t_query_tensor = t_query.clone().detach()
    idx = torch.argmin(torch.abs(rangemeter[:, 0] - t_query_tensor))
    return rangemeter[idx, 1]