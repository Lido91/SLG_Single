"""
QFormer Base Model - Adapted for MotionGPT

Adapted from LaMP's QFormer_Base.py with imports modified for MotionGPT structure.
"""
import contextlib
import logging
import torch
import torch.nn as nn
from transformers import BertTokenizer
from .QFormer import BertConfig, BertLMHeadModel
from .basemodel import BaseModel


class QFormer_Base(BaseModel):
    """
    Base class for QFormer-based models.

    Provides utilities for:
    - Initializing BERT tokenizer
    - Initializing QFormer architecture
    - Initializing motion encoder
    """

    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        """Initialize BERT tokenizer with special tokens."""
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        """
        Conditional autocasting for mixed precision training.
        Only enabled on GPU.
        """
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        """
        Initialize QFormer with cross-attention to vision/motion features.

        Args:
            num_query_token: Number of learnable query tokens (e.g., 32)
            vision_width: Dimension of vision/motion features (e.g., 1408 for motion encoder output)
            cross_attention_freq: Insert cross-attention every N blocks (default: 2)

        Returns:
            Qformer: BertLMHeadModel with cross-attention
            query_tokens: Learnable query embeddings [1, num_query_token, hidden_size]
        """
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # Insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        return Qformer, query_tokens

    @classmethod
    def init_motion_encoder_from_vae(cls, vae_encoder):
        """
        Use the encoder from a pretrained RVQ-VAE as the motion encoder.

        Args:
            vae_encoder: The encoder module from RVQVae

        Returns:
            Motion encoder (frozen)
        """
        # The encoder is already instantiated, just return it
        # It will be frozen in the LaMP model
        return vae_encoder

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """
        Get optimizer parameter groups with layer-wise learning rate decay.

        Args:
            weight_decay: Weight decay coefficient
            lr_scale: Base learning rate scale

        Returns:
            List of parameter groups with lr_scale and weight_decay
        """
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": 1.0
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": 1.0
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        optim_params = list(parameter_group_vars.values())
        return optim_params


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
