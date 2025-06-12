import os
import numpy as np
import torch
import time
from torch.utils.data import Dataset

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
    t_query_tensor = torch.tensor(t_query, dtype=torch.float32, device=rangemeter.device)
    idx = torch.argmin(torch.abs(rangemeter[:, 0] - t_query_tensor))
    return rangemeter[idx, 1]

class EventVelocityDataset(Dataset):
    def __init__(self, folder_path, shape=(200, 200), bins=5, window_size=200, device='cpu'):
    
        self.shape = shape
        self.bins = bins
        self.window_size = window_size  # 200 events before/after
        self.samples = []
        self.device = device
        self.sequences = []

        npz_files = sorted([f for f in os.listdir(folder_path)])

        for file_idx, file_name in enumerate(npz_files):
            path = os.path.join(folder_path, file_name)
            data = np.load(path)

            events_np = np.stack([
                data["events"]['x'],
                data["events"]['y'],
                data["events"]['p'].astype(np.int8),
                data["events"]['t'].astype(np.float32),
            ], axis=-1)

            events = torch.from_numpy(events_np).to(torch.float32)
            timestamps = torch.from_numpy(data["timestamps"]).to(torch.float32)
            traj = torch.from_numpy(data["traj"]).to(torch.float32)
            range_meter = torch.from_numpy(data["range_meter"]).to(torch.float32)

            ts_us = timestamps * 1e6
            event_times = events[:, 3]

            self.sequences.append({
                "events": events,
                "event_times": event_times,
                "traj": traj,
                "timestamps": timestamps,
                "ts_us": ts_us,
                "range_meter": range_meter,
            })

            for ts_idx in range(len(ts_us)):
                for bin_idx in range(bins):
                    self.samples.append((file_idx, ts_idx, bin_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        start_time = time.time()

        seq_idx, ts_idx, bin_idx = self.samples[index]

        
        seq = self.sequences[seq_idx]

        t_center = seq["ts_us"][ts_idx]

        print(type(seq), type(seq["ts_us"]), type(t_center))
        t_start = t_center - self.window_size * 1000
        t_end = t_center + self.window_size * 1000

        mask = (seq["event_times"] >= t_start) & (seq["event_times"] <= t_end)
        events_slice = seq["events"][mask]

        if len(events_slice) == 0:
            voxel = torch.zeros((self.bins, *self.shape), dtype=torch.float32, device=self.device)
        else:
            voxel = events_to_voxel_grid(events_slice.to(self.device), shape=self.shape, bins=self.bins, device=self.device)

        bin_image = voxel[bin_idx]
        velocity = seq["traj"][ts_idx, 3:6].to(self.device)
        ts = seq["timestamps"][ts_idx]
        range_val = get_closest_range(seq["range_meter"], ts)

        elapsed = time.time() - start_time
        print(f"⏱ Temps d'accès à l'échantillon {index} : {elapsed:.6f} secondes")

        return bin_image.unsqueeze(0), range_val, velocity
    

if __name__ == "__main__":

    print("Répertoire courant :", os.getcwd())
    folder_path = "./elope_dataset/train"  # À adapter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = EventVelocityDataset(folder_path, device=device)

    # Récupérer et chronométrer un seul élément du dataset
    index = 0
    print(f"\n🔍 Extraction de l'échantillon {index} du dataset...")
    image, range_val, velocity = dataset[index]

    print(f"\n✅ Donnée extraite :")
    print(f" - Image shape : {image.shape}")
    print(f" - Range       : {range_val.item():.2f} m")
    print(f" - Velocity    : {velocity.numpy()}")

  
"""
before : 0.110896 s

after 1 : 0.025563 s

after 2 : 0.010902 s
"""