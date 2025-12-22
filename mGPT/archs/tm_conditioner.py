"""
T5 Text Conditioner for Text-to-Motion Generation
Adapted from UniMuMo's conditioners.py

Key Features:
- T5EncoderModel for text encoding
- Projects T5 embeddings to transformer dimension
- CFG dropout support
- Attention mask handling
"""

import logging
import typing as tp
import warnings
from copy import deepcopy

import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer


logger = logging.getLogger(__name__)


class TorchAutocast:
    """TorchAutocast utility class."""
    def __init__(self, enabled: bool = True, device_type: str = 'cuda',
                 dtype: torch.dtype = torch.float16):
        self.enabled = enabled
        self.device_type = device_type
        self.dtype = dtype

    def __enter__(self):
        if self.enabled:
            self._autocast = torch.autocast(device_type=self.device_type, dtype=self.dtype)
            return self._autocast.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled:
            return self._autocast.__exit__(*args)


class T5Conditioner(nn.Module):
    """
    T5-based text conditioner for motion generation.

    Uses T5EncoderModel to encode text descriptions and projects
    the output to the motion transformer's hidden dimension.

    Args:
        name: T5 model name (e.g., 'google/flan-t5-base')
        output_dim: Output dimension (motion transformer hidden dim)
        finetune: Whether to finetune T5 during training
        device: Device for T5
        autocast_dtype: Autocast dtype for T5 inference
        word_dropout: Word dropout probability
    """
    MODELS = [
        "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
        "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
        "google/flan-t5-xl", "google/flan-t5-xxl"
    ]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
        self,
        name: str = "google/flan-t5-base",
        output_dim: int = 768,
        finetune: bool = False,
        device: str = 'cuda',
        autocast_dtype: tp.Optional[str] = 'float32',
        word_dropout: float = 0.0,
    ):
        super().__init__()

        assert name in self.MODELS, f"Unrecognized T5 model: {name}. Available: {self.MODELS}"

        self.name = name
        self.dim = self.MODELS_DIMS[name]
        self.output_dim = output_dim
        self.finetune = finetune
        self.word_dropout = word_dropout

        # Setup autocast
        if autocast_dtype is None or device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            if device != 'cpu':
                logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=device, dtype=dtype)

        # Load T5
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                logger.info(f"Loading T5 model: {name}")
                self.tokenizer = T5Tokenizer.from_pretrained(name)
                t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)

        if finetune:
            self.t5 = t5
        else:
            self.t5 = t5
            for p in self.t5.parameters():
                p.requires_grad = False
            self.t5.eval()

        # Output projection to motion transformer dimension
        self.output_proj = nn.Linear(self.dim, output_dim)

    def tokenize(self, texts: tp.List[tp.Optional[str]], device: torch.device
                ) -> tp.Dict[str, torch.Tensor]:
        """
        Tokenize text inputs.

        Args:
            texts: List of text strings (None for empty conditions)
            device: Target device

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Replace None with empty string
        entries = [t if t is not None else "" for t in texts]

        # Apply word dropout during training
        if self.word_dropout > 0.0 and self.training:
            import random
            new_entries = []
            for entry in entries:
                words = [w for w in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        # Tokenize
        inputs = self.tokenizer(
            entries,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]
               ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text inputs.

        Args:
            inputs: Tokenized inputs from tokenize()

        Returns:
            Tuple of (embeddings, mask):
                - embeddings: [B, L, output_dim]
                - mask: [B, L] attention mask
        """
        mask = inputs['attention_mask']

        with torch.set_grad_enabled(self.finetune):
            with self.autocast:
                embeds = self.t5(**inputs).last_hidden_state

        # Project to output dimension
        embeds = self.output_proj(embeds.to(self.output_proj.weight.dtype))

        # Zero out padding positions
        embeds = embeds * mask.unsqueeze(-1)

        return embeds, mask

    def encode(self, texts: tp.List[tp.Optional[str]], device: torch.device
              ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to tokenize and encode in one call.

        Args:
            texts: List of text strings
            device: Target device

        Returns:
            Tuple of (embeddings, mask)
        """
        inputs = self.tokenize(texts, device)
        return self.forward(inputs)


class CFGDropout(nn.Module):
    """
    Classifier-Free Guidance dropout.

    During training, randomly drops all conditions with probability p.
    This enables unconditional generation for CFG at inference.

    Args:
        p: Dropout probability
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, condition: torch.Tensor, mask: torch.Tensor
               ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CFG dropout.

        Args:
            condition: Condition embeddings [B, L, D]
            mask: Attention mask [B, L]

        Returns:
            Tuple of (condition, mask) with some samples nullified
        """
        if not self.training or self.p == 0:
            return condition, mask

        B = condition.shape[0]
        device = condition.device

        # Randomly select which samples to drop
        drop_mask = torch.rand(B, device=device) < self.p

        # Zero out dropped samples
        condition = condition * (~drop_mask).float().view(B, 1, 1)
        mask = mask * (~drop_mask).float().view(B, 1)

        return condition, mask


class ConditionProvider(nn.Module):
    """
    Unified condition provider that handles text encoding and CFG.

    Args:
        t5_name: T5 model name
        output_dim: Output dimension
        finetune: Whether to finetune T5
        cfg_dropout: CFG dropout probability
        device: Device
    """
    def __init__(
        self,
        t5_name: str = "google/flan-t5-base",
        output_dim: int = 768,
        finetune: bool = False,
        cfg_dropout: float = 0.1,
        device: str = 'cuda',
        autocast_dtype: str = 'float32',
    ):
        super().__init__()

        self.t5_conditioner = T5Conditioner(
            name=t5_name,
            output_dim=output_dim,
            finetune=finetune,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        self.cfg_dropout = CFGDropout(p=cfg_dropout)

    def forward(
        self,
        texts: tp.List[tp.Optional[str]],
        device: torch.device,
        apply_cfg_dropout: bool = True,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text conditions.

        Args:
            texts: List of text descriptions
            device: Target device
            apply_cfg_dropout: Whether to apply CFG dropout

        Returns:
            Tuple of (embeddings, mask)
        """
        # Encode text
        condition, mask = self.t5_conditioner.encode(texts, device)

        # Apply CFG dropout
        if apply_cfg_dropout:
            condition, mask = self.cfg_dropout(condition, mask)

        return condition, mask

    def prepare_cfg_conditions(
        self,
        texts: tp.List[tp.Optional[str]],
        device: torch.device,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare conditions for CFG inference.

        Returns [conditional; unconditional] concatenated along batch dimension.

        Args:
            texts: List of text descriptions
            device: Target device

        Returns:
            Tuple of (embeddings, mask) with shape [2B, L, D] and [2B, L]
        """
        # Encode conditional
        cond_embeds, cond_mask = self.t5_conditioner.encode(texts, device)

        # Create unconditional (null) conditions
        uncond_texts = [""] * len(texts)
        uncond_embeds, uncond_mask = self.t5_conditioner.encode(uncond_texts, device)

        # Pad to same length if needed
        max_len = max(cond_embeds.shape[1], uncond_embeds.shape[1])

        if cond_embeds.shape[1] < max_len:
            pad = torch.zeros(
                cond_embeds.shape[0], max_len - cond_embeds.shape[1], cond_embeds.shape[2],
                device=device, dtype=cond_embeds.dtype
            )
            cond_embeds = torch.cat([cond_embeds, pad], dim=1)
            cond_mask = torch.cat([
                cond_mask,
                torch.zeros(cond_mask.shape[0], max_len - cond_mask.shape[1], device=device)
            ], dim=1)

        if uncond_embeds.shape[1] < max_len:
            pad = torch.zeros(
                uncond_embeds.shape[0], max_len - uncond_embeds.shape[1], uncond_embeds.shape[2],
                device=device, dtype=uncond_embeds.dtype
            )
            uncond_embeds = torch.cat([uncond_embeds, pad], dim=1)
            uncond_mask = torch.cat([
                uncond_mask,
                torch.zeros(uncond_mask.shape[0], max_len - uncond_mask.shape[1], device=device)
            ], dim=1)

        # Concatenate [conditional, unconditional]
        embeds = torch.cat([cond_embeds, uncond_embeds], dim=0)
        mask = torch.cat([cond_mask, uncond_mask], dim=0)

        return embeds, mask
