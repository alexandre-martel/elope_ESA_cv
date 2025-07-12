import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from model import eraft
import cv2
import json
from utils.dsec_utils import VoxelGrid
from utils.visualization import visualize_optical_flow
import torch.nn.functional as F

class EventVelocityDataset(Dataset):
    def __init__(self, folder_path, shape=(200, 200), n_velocity_ts=100, device='cpu', num_bins=15, max_ts=1000):
        self.shape = shape
        self.device = device
        self.num_bins = num_bins
        self.n_velocity_ts = n_velocity_ts
        self.FACTEUR = max_ts  # assez grand pour séparer les index (ex: 10000 timestamps max)
        self.sequences = []
        self.samples = []

        npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])

        for seq_id, file_name in enumerate(npz_files):
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

            self.sequences.append({
                "events": events,
                "event_times": events[:, 3],
                "traj": traj,
                "timestamps": timestamps,
                "range_meter": range_meter,
            })

            for ts_start_idx in range(len(timestamps) - n_velocity_ts):
                index = seq_id * self.FACTEUR + ts_start_idx
                self.samples.append(index)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        start_time = time.time()

        seq_id = index // self.FACTEUR
        ts_start_idx = index % self.FACTEUR
        seq = self.sequences[seq_id]

        timestamps = seq["timestamps"]
        traj = seq["traj"]

        t_start = timestamps[ts_start_idx].item() * 1e6
        t_end = timestamps[ts_start_idx + self.n_velocity_ts - 1].item() * 1e6
        t_mid = (t_start + t_end) / 2

        def voxelize_events(t0, t1):
            mask = (seq["event_times"] >= t0) & (seq["event_times"] < t1)
            events_slice = seq["events"][mask]
            if events_slice.shape[0] < 2:
                return torch.zeros((self.num_bins, *self.shape), dtype=torch.float32, device=self.device)

            events_dict = {
                'x': events_slice[:, 0],
                'y': events_slice[:, 1],
                't': events_slice[:, 3],
                'p': events_slice[:, 2]
            }
            voxelizer = VoxelGrid((self.num_bins, *self.shape), normalize=False)
            return voxelizer.convert(events_dict).to(self.device)

        image1 = voxelize_events(t_start, t_mid)
        image2 = voxelize_events(t_mid, t_end)

        velocity = traj[ts_start_idx:ts_start_idx + self.n_velocity_ts, 3:6].mean(dim=0).to(self.device)
        range_val = get_closest_range(seq["range_meter"], (t_start + t_end) / 2.0 / 1e6)

        elapsed = time.time() - start_time
        print(f"⏱ Accès index {index} (seq {seq_id}, ts {ts_start_idx}) : {elapsed:.6f} s")

        return image1, image2, range_val, velocity


def get_closest_range(range_meter, target_ts):
    """
    Trouve la valeur de range la plus proche temporellement de target_ts (en secondes), vectorisé.
    """
    idx = torch.argmin(torch.abs(range_meter[:, 0] - target_ts))
    return range_meter[idx, 1]

def visualize_flow(flow, title="Optical Flow"):
    """
    Visualisation simple du flow (exemple basique).
    flow: tensor (2, H, W)
    """
    flow = flow.cpu().numpy()
    hsv = flow_to_hsv(flow)
    plt.imshow(hsv)
    plt.title(title)
    plt.axis('off')
    plt.show()


def flow_to_hsv(flow):
    """
    Convertit un flow 2D en image HSV pour visualisation.
    flow : np.array (2, H, W)
    """
    import numpy as np
    H, W = flow.shape[1:]
    fx, fy = flow[0], flow[1]
    mag, ang = np.sqrt(fx**2 + fy**2), np.arctan2(fy, fx)
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = (ang + 3.14159265) * (180 / (2 * 3.14159265))  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = np.minimum(mag * 1000, 255)  # Value (amplitude)

    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return hsv

class FlowVisualizerEvent:
    def __init__(self, dataset, model, device='cpu'):
        self.dataset = dataset
        self.model = model.to(device).eval()
        self.device = device

    def visualize(self, index):
        print(f"🔍 Visualisation index {index}")

        # Charger un échantillon
        image1, image2, range_val, velocity = self.dataset[index]
        print(f"📏 Range: {range_val.item():.2f} m")
        print(f"🛸 Vitesse cible : {velocity.cpu().numpy()}")

        # Préparer les images pour le modèle
        image1 = image1.unsqueeze(0).to(self.device)
        image2 = image2.unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, flow_predictions = self.model(image1, image2)
            flow_pred = flow_predictions[-1].squeeze(0).cpu().numpy()  # (2, H, W)

        # ➤ Visualisation Optical Flow
        flow_img, (fmin, fmax) = visualize_optical_flow(flow_pred, return_image=True)
        plt.imshow(flow_img)
        plt.title(f"Flow ∈ [{fmin:.2f}, {fmax:.2f}] — v={velocity.cpu().numpy()}")
        plt.axis('off')
        plt.show()

        # ➤ Visualisation des deux images voxelisées
        def plot_voxel(img, title):
            proj = img.sum(dim=0).cpu().numpy()
            plt.imshow(proj, cmap='plasma')
            plt.title(title)
            plt.axis('off')
            plt.show()

        plot_voxel(image1[0], "🟠 Image Voxelisée t1")
        plot_voxel(image2[0], "🔵 Image Voxelisée t2")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_folder = "./elope_dataset/train"
    config_path = "./E-RAFT/config/mvsec_20.json"

    # 🔧 Charger config E-RAFT
    config = json.load(open(config_path))
    checkpoint_path = config['test']['checkpoint']
    voxel_bins = config['data_loader']['test']['args']['num_voxel_bins']

    # 📦 Charger un échantillon du dataset
    dataset = EventVelocityDataset(data_folder, device=device)
    


    # 🔁 Charger le modèle ERAFT
    model = eraft.ERAFT(config=config, n_first_channels=voxel_bins)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()


    visualizer = FlowVisualizerEvent(dataset, model, device=device)
    visualizer.visualize(5000)
