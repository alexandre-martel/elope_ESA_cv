import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import events_to_image, get_closest_range

class EventVelocityDataset(Dataset):
    def __init__(self, folder_path, shape=(200, 200), window_duration_ms=400, n_windows=200, device='cpu'):

        self.shape = shape
        self.window_duration_us = window_duration_ms * 1000
        self.n_windows = n_windows
        self.samples = []
        self.device = device
        self.sequences = []

        npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])

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

            t_min = event_times[0].item()
            t_max = event_times[-1].item()
            total_duration = t_max - t_min

            max_possible = total_duration - self.window_duration_us
            if max_possible <= 0 or n_windows <= 0:
                continue 

            stride_us = max_possible / (n_windows - 1) if n_windows > 1 else 0

            for i in range(n_windows):
                t_start = t_min + i * stride_us
                t_end = t_start + self.window_duration_us
                if t_end <= t_max:
                    self.samples.append((file_idx, t_start, t_end))



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        seq_idx, t_start, t_end = self.samples[index]
        seq = self.sequences[seq_idx]

        mask = (seq["event_times"] >= t_start) & (seq["event_times"] <= t_end)
        events_slice = seq["events"][mask]

        # --- Nouveau traitement pour 2 canaux (pos/neg) ---
        if events_slice.numel() == 0:
            image = torch.zeros((2, *self.shape), dtype=torch.float32, device=self.device)
        else:
            x = events_slice[:, 0].long().clamp(0, self.shape[1] - 1)
            y = events_slice[:, 1].long().clamp(0, self.shape[0] - 1)
            p = events_slice[:, 2]

            pos_mask = (p > 0)
            neg_mask = ~pos_mask

            img_pos = torch.zeros(self.shape[0] * self.shape[1], dtype=torch.float32, device=self.device)
            img_neg = torch.zeros_like(img_pos)

            indices_pos = y[pos_mask] * self.shape[1] + x[pos_mask]
            indices_neg = y[neg_mask] * self.shape[1] + x[neg_mask]

            img_pos.index_add_(0, indices_pos, torch.ones_like(indices_pos, dtype=torch.float32))
            img_neg.index_add_(0, indices_neg, torch.ones_like(indices_neg, dtype=torch.float32))

            img_pos = img_pos.view(self.shape)
            img_neg = img_neg.view(self.shape)

            image = torch.stack([img_pos, img_neg], dim=0)  # shape (2, H, W)

        # Moyenne des 2 vitesses les plus proches du centre temporel
        t_center = (t_start + t_end) / 2.0 / 1e6  # en secondes
        timestamps = seq["timestamps"]
        traj = seq["traj"]

        time_diffs = torch.abs(timestamps - t_center)
        closest_indices = torch.topk(time_diffs, k=2, largest=False).indices
        velocities = traj[closest_indices, 3:6]
        velocity_mean = velocities.mean(dim=0).to(self.device)

        range_val = get_closest_range(seq["range_meter"], t_center)

        

        return image, range_val, velocity_mean