"""
PyTorch Lightning Module for Text-to-Motion Training
Adapted from UniMuMo's transformer_model.py

Key Features:
- Training with per-codebook loss (using pre-computed tokens)
- CFG dropout during training
- Validation with generation and VQ-VAE decoding (on-the-fly)
- Learning rate scheduling
- Metric logging (MRMetrics for motion reconstruction)
"""

import typing as tp
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pytorch_lightning as pl

from mGPT.archs.tm_model import TextToMotionLM, build_text_to_motion_model
from mGPT.config import instantiate_from_config
from mGPT.metrics.mr import MRMetrics


class TextToMotionLightning(pl.LightningModule):
    """
    PyTorch Lightning module for Text-to-Motion training.

    Training uses pre-computed tokens from Text2MotionDatasetCB.
    Validation uses raw motion from Text2MotionDatasetEval + VQ-VAE for decoding.

    Args:
        n_q: Number of codebooks
        card: Codebook vocabulary size
        dim: Transformer dimension
        num_heads: Attention heads
        num_layers: Transformer layers
        t5_name: T5 model name
        t5_finetune: Whether to finetune T5
        cfg_dropout: CFG dropout rate
        cfg_coef: CFG coefficient
        pattern_type: Codebook pattern type
        lr: Learning rate
        weight_decay: Weight decay
        warmup_steps: Warmup steps
        max_steps: Maximum training steps
        emb_lr: Separate learning rate for embeddings
        vae_cfg: VQ-VAE config for validation (optional)
    """
    def __init__(
        self,
        # Model config
        n_q: int = 6,
        card: int = 512,
        dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        hidden_scale: int = 4,
        dropout: float = 0.1,
        layer_scale: tp.Optional[float] = None,
        # T5 config
        t5_name: str = "google/flan-t5-base",
        t5_finetune: bool = False,
        # CFG config
        cfg_dropout: float = 0.1,
        cfg_coef: float = 3.0,
        # Pattern config
        pattern_type: str = 'delayed',
        delays: tp.Optional[tp.List[int]] = None,
        # Training config
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        emb_lr: tp.Optional[float] = None,
        # Generation config
        gen_temp: float = 1.0,
        gen_top_k: int = 250,
        gen_top_p: float = 0.0,
        gen_max_len: int = 256,
        # VAE config for validation
        vae_cfg: tp.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build TM Transformer model
        self.model = TextToMotionLM(
            n_q=n_q,
            card=card,
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_scale=hidden_scale,
            dropout=dropout,
            layer_scale=layer_scale,
            t5_name=t5_name,
            t5_finetune=t5_finetune,
            cfg_dropout=cfg_dropout,
            cfg_coef=cfg_coef,
            pattern_type=pattern_type,
            delays=delays,
            emb_lr=emb_lr,
        )

        # VQ-VAE for validation (decode generated tokens to motion)
        self.vae = None
        if vae_cfg is not None:
            self.vae = instantiate_from_config(vae_cfg)
            # Freeze VAE
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        # Training config
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.emb_lr = emb_lr

        # Generation config
        self.gen_temp = gen_temp
        self.gen_top_k = gen_top_k
        self.gen_top_p = gen_top_p
        self.gen_max_len = gen_max_len

        # Codebook size for clamping
        self.card = card

        # Metrics (will be initialized in setup() when datamodule is available)
        self.metrics = None
        self.feats2joints = None
        self.njoints = kwargs.get('njoints', 127)  # Default for How2Sign
        self.jointstype = kwargs.get('jointstype', 'smplxh2s')

    def setup(self, stage: str):
        """Called by Lightning with datamodule available."""
        # Get feats2joints from datamodule (like mgpt.py)
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            datamodule = self.trainer.datamodule
            if hasattr(datamodule, 'feats2joints'):
                self.feats2joints = datamodule.feats2joints
                print(f"[tm_lightning] feats2joints loaded from datamodule")
            if hasattr(datamodule, 'njoints'):
                self.njoints = datamodule.njoints
                print(f"[tm_lightning] njoints={self.njoints}")

        # Initialize metrics (will be moved to device by Lightning)
        if self.metrics is None:
            self.metrics = MRMetrics(
                njoints=self.njoints,
                jointstype=self.jointstype,
                dist_sync_on_step=True,
            )
            print(f"[tm_lightning] MRMetrics initialized")

    def forward(self, codes: torch.LongTensor, texts: tp.List[str]):
        """Forward pass for training."""
        return self.model(codes, texts)

    def training_step(self, batch, batch_idx):
        """Training step."""
        codes = batch['motion_codes']  # [B, K, T]
        texts = batch['texts']         # List of strings

        loss, loss_per_codebook = self.model(codes, texts)

        # Log metrics
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        for k, loss_k in enumerate(loss_per_codebook):
            self.log(f'train/loss_cb{k}', loss_k, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with on-the-fly generation and VAE decoding.

        Follows mgpt.py's val_t2m_forward() pattern:
        1. Get raw motion features from Text2MotionDatasetEval
        2. Generate motion tokens from text using TM Transformer
        3. Decode tokens to motion using VQ-VAE
        4. Compute joints/vertices and update metrics
        """
        # Batch from Text2MotionDatasetEval (val_collate_fn):
        # {'text': texts, 'motion': feats, 'length': lengths, 'name': names, 'src': srcs}
        feats_ref = batch['motion']  # Raw motion features [B, T, C]
        texts = batch['text']        # List of strings
        lengths = batch['length']    # List of lengths
        names = batch['name']        # List of sample names
        srcs = batch['src']          # List of sources

        B = feats_ref.shape[0]
        C = feats_ref.shape[-1]  # Feature dim (133)

        if self.vae is None:
            # No VAE loaded - skip validation
            self.log('val/loss', 0.0, prog_bar=True, sync_dist=True)
            return None

        with torch.no_grad():
            # Generate motion tokens from text using TM Transformer
            # Returns [B, K, T] tensor
            outputs_tokens = self.generate(
                texts=texts,
                max_gen_len=self.gen_max_len,
                use_sampling=True,
                temp=self.gen_temp,
                top_k=self.gen_top_k,
                cfg_coef=self.model.cfg_coef,
                show_progress=False,
            )

            # Allocate output tensor based on max generated token length
            # Each token = 4 frames (UNIT_LEN)
            max_token_len = outputs_tokens.shape[-1]  # T dimension
            feats_rst = torch.zeros(B, max_token_len * 4, C).to(feats_ref.device)

            # Decode generated tokens and track lengths
            lengths_rst = []
            for i in range(B):
                # Get tokens for this sample: [K, T] -> [T, K] for VAE
                tokens_i = outputs_tokens[i].permute(1, 0)  # [T, K]

                # Clamp tokens to valid range
                tokens_i = torch.clamp(tokens_i, 0, self.card - 1)

                if tokens_i.shape[0] > 1:
                    # Decode: VAE expects [1, T, K] or [T, K]
                    motion = self.vae.decode(tokens_i)  # [1, T*4, C]
                else:
                    motion = torch.zeros(1, 4, C).to(feats_ref.device)

                feats_rst[i:i+1, :motion.shape[1], :] = motion
                lengths_rst.append(motion.shape[1])

            # Convert features to joints/vertices for metrics (like mgpt.py)
            # Skip metrics if feats2joints not available
            if self.feats2joints is not None and self.metrics is not None:
                # Use min length to avoid padding issues - like mgpt.py does
                # The generated motion length should match reference for fair comparison
                T_ref = feats_ref.shape[1]
                T_rst = feats_rst.shape[1]
                T_min = min(T_ref, T_rst)

                # Truncate to minimum length (avoid padding)
                feats_ref_truncated = feats_ref[:, :T_min, :]
                feats_rst_truncated = feats_rst[:, :T_min, :]

                # Adjust lengths to not exceed T_min
                lengths_truncated = [min(l, T_min) for l in lengths_rst]

                # Get joints and vertices
                feats2joints_result_ref = self.feats2joints(feats_ref_truncated)
                feats2joints_result_rst = self.feats2joints(feats_rst_truncated)

                # Handle both tuple (vertices, joints) and single tensor return
                if isinstance(feats2joints_result_ref, tuple):
                    vertices_ref, joints_ref = feats2joints_result_ref
                    vertices_rst, joints_rst = feats2joints_result_rst
                else:
                    joints_ref = feats2joints_result_ref
                    joints_rst = feats2joints_result_rst
                    vertices_ref = joints_ref  # Use joints as vertices if not available
                    vertices_rst = joints_rst

                # Debug: print shapes on first validation batch
                if batch_idx == 0:
                    print(f"[tm_lightning] validation shapes:")
                    print(f"  feats_ref_truncated: {feats_ref_truncated.shape}")
                    print(f"  feats_rst_truncated: {feats_rst_truncated.shape}")
                    print(f"  joints_ref: {joints_ref.shape}, joints_rst: {joints_rst.shape}")
                    print(f"  vertices_ref: {vertices_ref.shape}, vertices_rst: {vertices_rst.shape}")
                    print(f"  lengths_truncated: {lengths_truncated}")

                # Update metrics
                self.metrics.update(
                    feats_rst=feats_rst_truncated,
                    feats_ref=feats_ref_truncated,
                    joints_rst=joints_rst,
                    joints_ref=joints_ref,
                    vertices_rst=vertices_rst,
                    vertices_ref=vertices_ref,
                    lengths=lengths_truncated,
                    src=srcs,
                    name=names,
                )

                if batch_idx == 0:
                    print(f"[tm_lightning] metrics.update() called successfully")

        # Return results for potential further processing
        return {
            'm_ref': feats_ref,
            'm_rst': feats_rst,
            'lengths': lengths,
            'lengths_rst': lengths_rst,
            'texts': texts,
        }

    def on_validation_epoch_end(self):
        """Log validation metrics at end of epoch."""
        print(f"[tm_lightning] on_validation_epoch_end called, sanity_checking={self.trainer.sanity_checking}")

        if self.metrics is not None:
            # Compute metrics (always compute, but only log if not sanity checking)
            metrics_dict = self.metrics.compute(sanity_flag=self.trainer.sanity_checking)
            print(f"[tm_lightning] metrics computed: {list(metrics_dict.keys())}")

            # Log all metrics (skip during sanity check)
            if not self.trainer.sanity_checking:
                for name, value in metrics_dict.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value.mean().item()
                    self.log(f'val/{name}', value, prog_bar=False, sync_dist=True)
                print(f"[tm_lightning] metrics logged")

    @torch.no_grad()
    def generate(
        self,
        texts: tp.List[str],
        max_gen_len: tp.Optional[int] = None,
        use_sampling: bool = True,
        temp: tp.Optional[float] = None,
        top_k: tp.Optional[int] = None,
        top_p: tp.Optional[float] = None,
        cfg_coef: tp.Optional[float] = None,
        show_progress: bool = False,
    ) -> torch.LongTensor:
        """
        Generate motion codes from text.

        Args:
            texts: List of text descriptions
            max_gen_len: Maximum generation length
            use_sampling: Whether to sample
            temp: Temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            cfg_coef: CFG coefficient
            show_progress: Whether to show progress bar

        Returns:
            Generated motion codes [B, K, T]
        """
        return self.model.generate(
            texts=texts,
            max_gen_len=max_gen_len or self.gen_max_len,
            use_sampling=use_sampling,
            temp=temp or self.gen_temp,
            top_k=top_k or self.gen_top_k,
            top_p=top_p or self.gen_top_p,
            cfg_coef=cfg_coef or self.model.cfg_coef,
            show_progress=show_progress,
        )

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Separate embedding parameters if different LR
        if self.emb_lr is not None:
            emb_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if 'emb' in name:
                    emb_params.append(param)
                else:
                    other_params.append(param)

            param_groups = [
                {'params': other_params, 'lr': self.lr},
                {'params': emb_params, 'lr': self.emb_lr},
            ]
        else:
            param_groups = self.model.parameters()

        # Optimizer
        optimizer = AdamW(
            param_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        # Scheduler: linear warmup + cosine decay
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=self.lr * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


def tm_collate_fn(batch):
    """
    Collate function for TM Transformer TRAINING DataLoader.

    Input batch format (from Text2MotionDatasetCB):
        (caption, m_tokens, m_length, name, None, None, None, all_captions, tasks, src)
        - m_tokens: [T, K] tensor of motion codes

    Output format for TM Transformer:
        {'motion_codes': [B, K, T], 'texts': List[str]}
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None and b[1] is not None]
    if len(batch) == 0:
        return None

    captions = [b[0] for b in batch]
    tokens_list = [b[1] for b in batch]  # List of [T, K] tensors

    # Debug: print token shape for first batch
    if tokens_list and hasattr(tm_collate_fn, '_first_batch'):
        pass
    elif tokens_list:
        tm_collate_fn._first_batch = True
        print(f"[tm_collate_fn] First token shape: {tokens_list[0].shape} (expected [T, K])")

    # Get dimensions
    K = tokens_list[0].shape[-1]  # Number of codebooks
    max_len = max(t.shape[0] for t in tokens_list)  # Max time length

    # Pad and transpose: [T, K] -> [K, T]
    padded_codes = torch.zeros(len(batch), K, max_len, dtype=torch.long)
    for i, tokens in enumerate(tokens_list):
        T = tokens.shape[0]
        # tokens is [T, K], transpose to [K, T]
        padded_codes[i, :, :T] = tokens.permute(1, 0)

    return {
        'motion_codes': padded_codes,  # [B, K, T]
        'texts': captions,             # List[str]
    }


def tm_val_collate_fn(batch):
    """
    Collate function for TM Transformer VALIDATION DataLoader.

    Input batch format (from Text2MotionDatasetEval):
        (text, motion, m_length, name, None, None, word_tokens, all_captions, None, src)
        - motion: [T, C] tensor of raw motion features (normalized)

    Output format for validation:
        {'text': List[str], 'motion': [B, T, C], 'length': List[int], 'name': List[str], 'src': List[str]}
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None and b[1] is not None]
    if len(batch) == 0:
        return None

    texts = [b[0] for b in batch]
    motions = [b[1] for b in batch]  # List of [T, C] tensors
    lengths = [b[2] for b in batch]
    names = [b[3] for b in batch]
    srcs = [b[9] for b in batch]

    # Get dimensions
    C = motions[0].shape[-1]  # Feature dim (133)
    max_len = max(m.shape[0] for m in motions)

    # Pad motions
    padded_motions = torch.zeros(len(batch), max_len, C)
    for i, motion in enumerate(motions):
        T = motion.shape[0]
        padded_motions[i, :T, :] = motion

    return {
        'text': texts,              # List[str]
        'motion': padded_motions,   # [B, T, C]
        'length': lengths,          # List[int]
        'name': names,              # List[str]
        'src': srcs,                # List[str]
    }


def create_tm_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """
    Create DataLoader for TM Transformer training using existing H2S datasets.

    Args:
        dataset: Text2MotionDatasetCB instance (from H2SDataModule)
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle

    Returns:
        DataLoader with tm_collate_fn
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=tm_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
