import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import EventVelocityDataset
from model import VelocityLightningModule
import argparse
import os

def main(args):

    print(f"Mode: {args.mode}")
    print(f"Data folder: {args.folder_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Use range meter: {args.use_range}")

    os.makedirs(args.output_dir, exist_ok=True)
    # Load training data
    dataset = EventVelocityDataset(args.folder_path)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(args.mode=="train"))
    
    model = VelocityLightningModule(use_range=args.use_range)
    print(f"[INFO] Model initialized, use_range: {args.use_range} --------")

    if args.checkpoint_path:
        print(f"[INFO] Loading checkpoint from {args.checkpoint_path}")
        model = model.load_from_checkpoint(args.checkpoint_path)

    logger = TensorBoardLogger(args.output_dir, name="velocity_model")
    print(f"[INFO] Logger initialized, saving to: {logger.log_dir} --------")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        default_root_dir=args.output_dir,
    )
    print(f"[INFO] Trainer initialized on {'gpu' if torch.cuda.is_available() else 'cpu'}, starting training... --------") 

    
    if args.mode == "train":
        trainer.fit(model, dataloader)
    elif args.mode == "validate":
        trainer.validate(model, dataloaders=dataloader)
    elif args.mode == "test":
        results = trainer.test(model, dataloaders=dataloader)
        print("Test results:", results)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, validate or test velocity prediction model.")
    parser.add_argument("--mode", choices=["train", "validate", "test"], required=True, help="Mode to run")
    parser.add_argument("--folder_path", type=str, required=True, help="Dataset folder for train/validate/test")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Folder to save logs, checkpoints, results")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path for validate/test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (only for train)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--use_range", action="store_true", help="Use range meter data")

    args = parser.parse_args()
    main(args)
