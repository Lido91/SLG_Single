"""
VQ-VAE with Finite Scalar Quantization (FSQ) for Sign Language Motion

Replaces traditional VQ-VAE codebook with FSQ for:
- No codebook collapse
- Simpler training (no EMA updates)
- Deterministic quantization
- Better stability
"""

from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution

from .mgpt_vq import Encoder, Decoder  # Reuse existing encoder/decoder
from .tools.vq.FSQ import FSQ


class FSQVae(nn.Module):
    """
    VQ-VAE with Finite Scalar Quantization

    Architecture:
        Input [B, T, D]
            ↓ preprocess
        [B, D, T]
            ↓ Encoder
        [B, code_dim, T']
            ↓ FSQ (finite scalar quantization)
        [B, code_dim, T'], indices[B, T']
            ↓ Decoder
        [B, D, T]
            ↓ postprocess
        Output [B, T, D]

    Args:
        nfeats: Number of input features (e.g., 133 for SMPLX)
        fsq_levels: Quantization levels per dimension (e.g., [5,5,5,5] for 625 codes)
        code_dim: Latent code dimension (should match len(fsq_levels))
        output_emb_width: Encoder output width (default: 512)
        down_t: Number of temporal downsampling layers (default: 3)
        stride_t: Stride for temporal downsampling (default: 2)
        width: Channel width (default: 512)
        depth: ResNet depth (default: 3)
        dilation_growth_rate: Dilation growth rate (default: 3)
        norm: Normalization type (default: None)
        activation: Activation function (default: 'relu')
    """

    def __init__(
        self,
        nfeats: int,
        fsq_levels: List[int] = [5, 5, 5, 5],
        code_dim: int = 4,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        norm: Optional[str] = None,
        activation: str = "relu",
        use_projection: bool = True,
        **kwargs
    ) -> None:
        super().__init__()

        # Validate FSQ configuration
        assert len(fsq_levels) == code_dim, f"fsq_levels length {len(fsq_levels)} must match code_dim {code_dim}"

        self.code_dim = code_dim
        self.fsq_levels = fsq_levels
        self.code_num = int(torch.tensor(fsq_levels).prod().item())  # Total codebook size
        self.nfeats = nfeats
        self.use_projection = use_projection

        # Determine encoder output dimension
        # If use_projection=False, encoder outputs code_dim directly (more efficient)
        # If use_projection=True, encoder outputs output_emb_width and FSQ projects to code_dim
        encoder_output_dim = output_emb_width if use_projection else code_dim

        # Encoder: [B, nfeats, T] -> [B, encoder_output_dim, T']
        self.encoder = Encoder(
            nfeats,
            encoder_output_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm
        )

        # FSQ Quantizer: [B, encoder_output_dim, T'] -> [B, code_dim, T']
        # When use_projection=False and encoder_output_dim=code_dim, FSQ uses Identity projection
        self.quantizer = FSQ(
            levels=fsq_levels,
            dim=encoder_output_dim,  # Input dimension
            num_codebooks=1,
            force_quantization_f32=True,
            return_indices=True
        )

        # Decoder: [B, encoder_output_dim, T'] -> [B, nfeats, T]
        self.decoder = Decoder(
            nfeats,
            encoder_output_dim,
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
        Forward pass through FSQ-VQVAE

        Args:
            features: Input motion features [B, T, D]

        Returns:
            x_out: Reconstructed motion [B, T, D]
            loss: Commitment loss (0 for FSQ)
            perplexity: Codebook usage metric
        """
        # Preprocess: [B, T, D] -> [B, D, T]
        x_in = self.preprocess(features)

        # Encode: [B, D, T] -> [B, output_emb_width, T']
        x_encoder = self.encoder(x_in)

        # Reshape for FSQ: [B, C, T'] -> [B, T', C]
        x_encoder = x_encoder.permute(0, 2, 1)

        # FSQ quantization: [B, T', C] -> [B, T', C], indices [B, T']
        x_quantized, indices, (loss, perplexity) = self.quantizer(x_encoder, return_loss_breakdown=True)

        # Reshape back: [B, T', C] -> [B, C, T']
        x_quantized = x_quantized.permute(0, 2, 1)

        # Decode: [B, C, T'] -> [B, D, T]
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
            code_idx: Discrete codes [B, T']
            dist: None (for compatibility)
        """
        B, T, D = features.shape

        # Preprocess and encode
        x_in = self.preprocess(features)  # [B, D, T]
        x_encoder = self.encoder(x_in)    # [B, C, T']

        # Reshape for FSQ
        x_encoder = x_encoder.permute(0, 2, 1)  # [B, T', C]

        # Get discrete codes
        _, indices, _ = self.quantizer(x_encoder)  # [B, T']

        return indices, None

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
        x_encoder = self.encoder(x_in)    # [B, C, T']

        # Pool over time dimension to get fixed-size representation
        x_pooled = x_encoder.mean(dim=-1)  # [B, C]

        return x_pooled

    def decode(self, code_idx: Tensor) -> Tensor:
        """
        Decode discrete codes to motion

        Args:
            code_idx: Discrete codes [B, T'] or [T']

        Returns:
            x_out: Reconstructed motion [B, T, D]
        """
        # Handle different input shapes
        if code_idx.dim() == 1:
            code_idx = code_idx.unsqueeze(0)  # [T'] -> [1, T']

        B, T_prime = code_idx.shape

        # Decode indices to continuous codes: [B, T'] -> [B, T', C]
        x_d = self.quantizer.get_codes_from_indices(code_idx)

        # Reshape for decoder: [B, T', C] -> [B, C, T']
        x_d = x_d.permute(0, 2, 1)

        # Decode: [B, C, T'] -> [B, D, T]
        x_decoder = self.decoder(x_d)

        # Postprocess: [B, D, T] -> [B, T, D]
        x_out = self.postprocess(x_decoder)

        return x_out

    def get_codebook_size(self) -> int:
        """Return the codebook size"""
        return self.code_num
