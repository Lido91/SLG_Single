"""
Training entry point for the dedicated new_contrastive_codex pipeline.

Usage:
    python -m contrastive.train_codex --config contrastive/configs/new_contrastive_codex.yaml
"""

import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contrastive.dataset import (  # noqa: E402
    ContrastiveH2SDataset,
    PrecomputedContrastiveDataset,
    contrastive_collate,
    precomputed_collate,
)
from contrastive.model_codex import NewContrastiveCodexModel  # noqa: E402


class CodexHeadSaver(pl.Callback):
    """Persist the best projection heads and temperature parameter."""

    def __init__(self, dirpath: str):
        super().__init__()
        self.dirpath = dirpath
        self.best_r1 = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        avg_r1 = trainer.callback_metrics.get("val/avg_R@1")
        if avg_r1 is None:
            return

        avg_r1 = avg_r1.detach().item() if isinstance(avg_r1, torch.Tensor) else float(avg_r1)
        if avg_r1 <= self.best_r1 or not trainer.is_global_zero:
            return

        self.best_r1 = avg_r1
        os.makedirs(self.dirpath, exist_ok=True)
        if hasattr(pl_module, "speech_head"):
            torch.save(pl_module.speech_head.state_dict(), os.path.join(self.dirpath, "speech_head_best.pt"))
        if hasattr(pl_module, "text_head"):
            torch.save(pl_module.text_head.state_dict(), os.path.join(self.dirpath, "text_head_best.pt"))
        torch.save(pl_module.motion_head.state_dict(), os.path.join(self.dirpath, "motion_head_best.pt"))
        torch.save({"logit_scale": pl_module.logit_scale.data}, os.path.join(self.dirpath, "logit_scale_best.pt"))


def load_config(path: str) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def load_tensor(path: str) -> torch.Tensor:
    try:
        tensor = torch.load(path, weights_only=True)
    except TypeError:
        tensor = torch.load(path)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    return tensor


def main():
    parser = argparse.ArgumentParser(description="Train the new_contrastive_codex pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--use_gpus", type=str, default=None, help="Comma-separated GPU ids")
    parser.add_argument("--nodebug", action="store_true", help="Skip sanity validation steps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    vae_cfg = cfg["vae"]
    logger_cfg = cfg.get("logger", {})

    if args.use_gpus is not None:
        train_cfg["devices"] = [int(gpu_id) for gpu_id in args.use_gpus.split(",")]

    mean = load_tensor(data_cfg["mean_path"])
    std = load_tensor(data_cfg["std_path"])
    preload_features = data_cfg.get("preload_features", False)

    common_data_kwargs = dict(
        data_root=data_cfg["root"],
        mean=mean,
        std=std,
        max_motion_length=data_cfg["max_motion_length"],
        min_motion_length=data_cfg["min_motion_length"],
        audio_dir=data_cfg.get("audio_dir", data_cfg["root"]),
        dataset_name=data_cfg["dataset_name"],
        unit_length=data_cfg.get("unit_length", 4),
        youtube3d_root=data_cfg.get("youtube3d_root"),
    )

    if preload_features:
        feat_dir = data_cfg.get(
            "precomputed_feat_dir",
            os.path.join(data_cfg["root"], "precomputed_feats"),
        )
        enable_speech = model_cfg.get("enable_speech", True)
        enable_text = model_cfg.get("enable_text", True)
        precomputed_kwargs = dict(
            feat_root=feat_dir,
            speech_type=model_cfg.get("speech_encoder_type", "hubert-large"),
            text_type=model_cfg.get("text_encoder_type", "clip"),
            require_speech=enable_speech,
            require_text=enable_text,
        )
        train_dataset = PrecomputedContrastiveDataset(
            split="train",
            **precomputed_kwargs,
            **common_data_kwargs,
        )
        val_dataset = PrecomputedContrastiveDataset(
            split="val",
            **precomputed_kwargs,
            **common_data_kwargs,
        )
        collate_fn = precomputed_collate
    else:
        train_dataset = ContrastiveH2SDataset(split="train", **common_data_kwargs)
        val_dataset = ContrastiveH2SDataset(split="val", **common_data_kwargs)
        collate_fn = contrastive_collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 16),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 16),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    head_cfg = model_cfg.get("projection_head", {})
    loss_cfg = model_cfg.get("loss", {})
    vae_config = {"params": vae_cfg["params"]}

    model = NewContrastiveCodexModel(
        speech_encoder_type=model_cfg.get("speech_encoder_type", "hubert-large"),
        text_encoder_type=model_cfg.get("text_encoder_type", "clip"),
        vae_config=vae_config,
        vae_checkpoint=vae_cfg.get("checkpoint"),
        proj_dim=model_cfg.get("proj_dim", 512),
        head_num_layers=head_cfg.get("num_layers", 4),
        head_nhead=head_cfg.get("nhead", 8),
        head_ff_dim=head_cfg.get("ff_dim", 2048),
        head_dropout=head_cfg.get("dropout", 0.1),
        speech_max_seq_len=head_cfg.get("speech_max_seq_len", 1500),
        text_max_seq_len=head_cfg.get("text_max_seq_len", 77),
        motion_max_seq_len=head_cfg.get("motion_max_seq_len", 128),
        temperature_init=loss_cfg.get("temperature_init", 0.07),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_epochs=train_cfg.get("max_epochs", 100),
        preload_features=preload_features,
        enable_speech=model_cfg.get("enable_speech", True),
        enable_text=model_cfg.get("enable_text", True),
    )

    exp_name = logger_cfg.get("name", "new_contrastive_codex")
    exp_dir = os.path.join("experiments/contrastive_codex", exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    logger = WandbLogger(
        project=logger_cfg.get("project", "SContrastiveG"),
        name=exp_name,
        save_dir=exp_dir,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="contrastive-codex-{epoch:02d}",
        auto_insert_metric_name=False,
        monitor="val/avg_R@1",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    head_saver = CodexHeadSaver(ckpt_dir)

    devices = train_cfg.get("devices", [0])
    if isinstance(devices, str):
        devices = [int(gpu_id.strip()) for gpu_id in devices.split(",") if gpu_id.strip()]
    num_devices = devices if isinstance(devices, int) else len(devices)

    trainer_kwargs = dict(
        max_epochs=train_cfg.get("max_epochs", 100),
        accelerator="gpu",
        devices=devices,
        strategy="ddp" if num_devices > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, head_saver],
        precision=train_cfg.get("precision", "32-true"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
    )
    if args.nodebug:
        trainer_kwargs["num_sanity_val_steps"] = 0
        trainer_kwargs["enable_model_summary"] = False

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
