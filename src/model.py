import torch
import torch.nn as nn
import pytorch_lightning as pl

class FlowEncoder(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class VelocityFromFlow(nn.Module):
    def __init__(self, use_range=True):
        super().__init__()
        self.backbone = FlowEncoder(in_channels=2)
        self.use_range = use_range
        self.mlp = nn.Sequential(
            nn.Linear(2 + (1 if use_range else 0), 64), nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, voxel_grid, range_value=None):
        B = voxel_grid.size(0)
        flow_map = self.backbone(voxel_grid)
        flow_avg = flow_map.mean(dim=[2, 3])  # (B, 2)

        if self.use_range:
            x = torch.cat([flow_avg, range_value.view(B, 1)], dim=1)
        else:
            x = flow_avg
        return self.mlp(x)
    
class VelocityLightningModule(pl.LightningModule):
    def __init__(self, use_range=True, lr=1e-3):
        super().__init__()
        self.model = VelocityFromFlow(use_range=use_range)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.use_range = use_range

    def forward(self, voxels, ranges=None):
        return self.model(voxels, ranges)

    def training_step(self, batch, batch_idx):
        voxels, ranges, velocities = batch
        preds = self(voxels, ranges)
        loss = self.criterion(preds, velocities)
        self.log("train_loss", loss,  on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        voxels, ranges, velocities = batch
        preds = self(voxels, ranges)
        loss = self.criterion(preds, velocities)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
