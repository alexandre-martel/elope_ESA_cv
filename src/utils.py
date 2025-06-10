import numpy as np
import torch

def events_to_voxel_grid(events, shape=(200, 200), bins=5):
    """
    Convert event stream to voxel grid.

    Args:
        events: numpy array (N, 4) where columns are (x, y, p, t) from ELOPE data.
        shape: image shape (H, W)
        bins: number of temporal bins

    Returns:
        voxel grid: Tensor (bins, H, W)
    """
    H, W = shape
    voxel = torch.zeros((bins, H, W), dtype=torch.float32)
    
    # Normalize timestamps
    t = events[:, 3].astype(np.float32)
    t_min, t_max = t.min(), t.max()
    dt = (t_max - t_min) / bins if t_max > t_min else 1.0

    for x, y, p, ts in events:
        x, y = int(x), int(y)
        if 0 <= x < W and 0 <= y < H:
            b = int((ts - t_min) / dt)
            b = min(b, bins - 1)
            polarity = 1.0 if p else -1.0
            voxel[b, y, x] += polarity

    return voxel

def get_closest_range(rangemeter, t_query):
    idx = np.argmin(np.abs(rangemeter[:,0] - t_query))
    return rangemeter[:,1][idx]