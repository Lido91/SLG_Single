"""
VQ-VAE with Residual Vector Quantization

Extends the standard VQ-VAE with multi-stage residual quantization
for improved reconstruction quality and hierarchical representation.
"""

from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution

from .mgpt_vq import Encoder, Decoder  # Reuse existing encoder/decoder
from .tools.residual_vq import ResidualVQ, ResidualVQ_2D


class RVQVae(nn.Module):
    """
    VQ-VAE with Residual Vector Quantization

    Architecture:
        Input [B, T, D]
            ↓ preprocess
        [B, D, T]
            ↓ Encoder
        [B, code_dim, T']
            ↓ ResidualVQ (num_quantizers stages)
        [B, code_dim, T'], indices[num_quantizers × B*T']
            ↓ Decoder
        [B, D, T]
            ↓ postprocess
        Output [B, T, D]

    Args:
        nfeats: Number of input features (pose dimension)
        num_quantizers: Number of residual quantization stages (default: 6)
        quantizer: Base quantizer type ('ema_reset', 'orig', 'ema', 'reset')
        code_num: Codebook size per quantizer (default: 512)
        code_dim: Code dimension (default: 512)
        output_emb_width: Embedding width (default: 512)
        down_t: Number of temporal downsampling layers (default: 3)
        stride_t: Stride for temporal downsampling (default: 2)
        width: Channel width (default: 512)
        depth: ResNet depth (default: 3)
        dilation_growth_rate: Dilation growth rate (default: 3)
        norm: Normalization type (default: None)
        activation: Activation function (default: 'relu')
        quantize_dropout_prob: Quantizer dropout probability (default: 0.2)
        quantize_dropout_cutoff_index: Keep first N quantizers (default: 0)
        shared_codebook: Share codebook across stages (default: False)
    """

    def __init__(
        self,
        nfeats: int,
        num_quantizers: int = 6,
        quantizer: str = "ema_reset",
        code_num: int = 512,
        code_dim: int = 512,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        norm: Optional[str] = None,
        activation: str = "relu",
        quantize_dropout_prob: float = 0.2,
        quantize_dropout_cutoff_index: int = 0,
        shared_codebook: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.code_num = code_num
        self.code_dim = code_dim
        self.num_quantizers = num_quantizers
        self.quantize_dropout_prob = quantize_dropout_prob

        # Encoder: [B, D, T] -> [B, code_dim, T']
        self.encoder = Encoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm
        )

        # Residual Vector Quantizer
        self.quantizer = ResidualVQ(
            num_quantizers=num_quantizers,
            shared_codebook=shared_codebook,
            quantize_dropout_prob=quantize_dropout_prob,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
            nb_code=code_num,
            code_dim=code_dim,
            quantizer_type=quantizer,
            mu=0.99
        )

        # Decoder: [B, code_dim, T'] -> [B, D, T]
        self.decoder = Decoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm
        )

    def preprocess(self, x: Tensor) -> Tensor:
        """(B, T, D) -> (B, D, T)"""
        return x.permute(0, 2, 1)

    def postprocess(self, x: Tensor) -> Tensor:
        """(B, D, T) -> (B, T, D)"""
        return x.permute(0, 2, 1)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through R-VQVAE

        Args:
            features: Input motion features [B, T, D]

        Returns:
            x_out: Reconstructed motion [B, T, D]
            loss: Commitment loss
            perplexity: Codebook usage metric
        """
        # Preprocess: [B, T, D] -> [B, D, T]
        x_in = self.preprocess(features)

        # Encode: [B, D, T] -> [B, code_dim, T']
        x_encoder = self.encoder(x_in)

        # Residual quantization: [B, code_dim, T'] -> [B, code_dim, T'], indices, loss, perplexity
        x_quantized, all_indices, loss, perplexity = self.quantizer(x_encoder)

        # Decode: [B, code_dim, T'] -> [B, D, T]
        x_decoder = self.decoder(x_quantized)

        # Postprocess: [B, D, T] -> [B, T, D]
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity

    def encode(
        self,
        features: Tensor,
    ) -> Tuple[Tensor, None]:
        """
        Encode motion to discrete codes

        Args:
            features: Input motion features [B, T, D]

        Returns:
            code_idx: Discrete codes [B, T, num_quantizers]
            dist: None (for compatibility)
        """
        B, T, D = features.shape

        # Preprocess and encode
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)  # [B, code_dim, T']

        # Get discrete codes for all quantizers
        all_indices = self.quantizer.quantize(x_encoder)  # List of [B*T'] tensors

        # Stack indices: List of [B*T'] -> [B, T', num_quantizers]
        T_prime = x_encoder.shape[-1]
        code_idx_list = []
        for indices in all_indices:
            # Reshape [B*T'] -> [B, T']
            indices_reshaped = indices.view(B, T_prime)
            code_idx_list.append(indices_reshaped)

        # Stack along last dimension: [B, T', num_quantizers]
        code_idx = torch.stack(code_idx_list, dim=-1)

        return code_idx, None

    def encode_continuous(self, features: Tensor) -> Tensor:
        """
        Extract continuous embeddings BEFORE quantization
        For contrastive learning between text and motion

        Args:
            features: [B, T, D] motion features

        Returns:
            embeddings: [B, code_dim] pooled continuous embeddings
        """
        B, T, D = features.shape
        x_in = self.preprocess(features)  # [B, D, T]
        x_encoder = self.encoder(x_in)     # [B, code_dim, T']

        # Pool over time dimension to get fixed-size representation
        x_pooled = x_encoder.mean(dim=-1)  # [B, code_dim]

        return x_pooled

    def decode(self, code_idx: Tensor) -> Tensor:
        """
        Decode discrete codes to motion

        Args:
            code_idx: Discrete codes [B, T', num_quantizers] or [T', num_quantizers]
                      Can contain -1 for padding positions

        Returns:
            x_out: Reconstructed motion [B, T, D]
        """
        # Handle different input shapes
        if code_idx.dim() == 1:
            # [T'] -> [1, T', num_quantizers] - replicate across quantizers (Q0 only mode)
            code_idx = code_idx.unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.num_quantizers)
        elif code_idx.dim() == 2:
            # Could be [B, T'] or [T', num_quantizers]
            if code_idx.shape[-1] == self.num_quantizers:
                # [T', num_quantizers] -> [1, T', num_quantizers]
                code_idx = code_idx.unsqueeze(0)
            else:
                # [B, T'] -> [B, T', num_quantizers] - replicate across quantizers (Q0 only mode)
                code_idx = code_idx.unsqueeze(-1).repeat(1, 1, self.num_quantizers)

        B, T_prime, input_num_quantizers = code_idx.shape

        # Use get_codes_from_indices which handles -1 padding correctly
        # This method masks out -1 indices and sets their embeddings to 0
        all_codes = self.quantizer.get_codes_from_indices(code_idx)  # [num_quantizers, B, T', code_dim]

        # Sum across quantizers: [num_quantizers, B, T', code_dim] -> [B, T', code_dim]
        x_quantized = all_codes.sum(dim=0)  # [B, T', code_dim]

        # Permute to [B, code_dim, T'] for decoder
        x_quantized = x_quantized.permute(0, 2, 1).contiguous()  # [B, code_dim, T']

        # Decode: [B, code_dim, T'] -> [B, D, T]
        x_decoder = self.decoder(x_quantized)

        # Postprocess: [B, D, T] -> [B, T, D]
        x_out = self.postprocess(x_decoder)

        return x_out

    def decode_q0_only(self, q0_idx: Tensor) -> Tensor:
        """
        Decode using only Q0 tokens (for MoMask Stage 2 validation).

        The RVQ-VAE is trained with quantizer dropout, so the decoder
        can reconstruct coarse motion from Q0 embeddings alone.
        Q1-Q5 contributions are set to zero.

        Args:
            q0_idx: Q0 token indices [B, T'] or [T']

        Returns:
            x_out: Reconstructed motion [B, T, D]
        """
        # Handle input shapes
        if q0_idx.dim() == 1:
            q0_idx = q0_idx.unsqueeze(0)  # [T'] -> [1, T']

        B, T_prime = q0_idx.shape

        # Get Q0 embeddings only
        quantizer_0 = self.quantizer._get_quantizer(0)
        indices_flat = q0_idx.reshape(-1)  # [B*T']
        z_q0 = quantizer_0.dequantize(indices_flat)  # [B*T', code_dim]

        # Reshape to [B, code_dim, T']
        x_quantized = z_q0.view(B, T_prime, self.code_dim).permute(0, 2, 1).contiguous()

        # Q1-Q5 contributions are zero (simulating quantizer dropout)
        # x_quantized already contains only Q0, which is what we want

        # Decode: [B, code_dim, T'] -> [B, D, T]
        x_decoder = self.decoder(x_quantized)

        # Postprocess: [B, D, T] -> [B, T, D]
        x_out = self.postprocess(x_decoder)

        return x_out

    def decode_partial(self, code_idx: Tensor) -> Tensor:
        """
        Decode using a subset of quantizers (e.g., first 3 of 6).

        This is used by HierarchicalRVQGPT which generates only Q0, Q1, Q2 codes.
        The remaining quantizers (Q3-Q5) are treated as zero contribution.

        Args:
            code_idx: Discrete codes [T', n_quantizers] or [B, T', n_quantizers]
                      where n_quantizers <= self.num_quantizers

        Returns:
            x_out: Reconstructed motion [B, T, D]
        """
        # Handle input shapes
        if code_idx.dim() == 2:
            # [T', n_quantizers] -> [1, T', n_quantizers]
            code_idx = code_idx.unsqueeze(0)

        B, T_prime, n_quantizers = code_idx.shape

        # Dequantize only the provided quantizers
        x_quantized = None
        for i in range(n_quantizers):
            indices = code_idx[:, :, i]  # [B, T']
            indices_flat = indices.reshape(-1)  # [B*T']

            quantizer = self.quantizer._get_quantizer(i)
            z_q = quantizer.dequantize(indices_flat)  # [B*T', code_dim]

            # Reshape to [B, code_dim, T']
            z_q = z_q.view(B, T_prime, self.code_dim).permute(0, 2, 1).contiguous()

            if x_quantized is None:
                x_quantized = z_q
            else:
                x_quantized = x_quantized + z_q

        # Decode: [B, code_dim, T'] -> [B, D, T]
        x_decoder = self.decoder(x_quantized)

        # Postprocess: [B, D, T] -> [B, T, D]
        x_out = self.postprocess(x_decoder)

        return x_out

    def get_num_quantizers(self) -> int:
        """Return the number of quantizers"""
        return self.num_quantizers

    def get_codebook_size(self) -> int:
        """Return the codebook size per quantizer"""
        return self.code_num
