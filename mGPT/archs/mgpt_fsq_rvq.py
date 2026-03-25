"""
Residual Vector Quantization with Finite Scalar Quantization (FSQ)

Combines the hierarchical representation of RVQ with the simplicity of FSQ.
Each quantizer stage uses FSQ instead of learned codebooks.
"""

from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution

from .mgpt_vq import Encoder, Decoder  # Reuse existing encoder/decoder
from .tools.vq.FSQ import FSQ


class FSQResidualVQ(nn.Module):
    """
    Residual VQ with FSQ quantizers at each stage

    Instead of learned codebooks, uses FSQ for each quantizer.
    This provides:
    - Hierarchical representation (coarse to fine)
    - No codebook collapse at any stage
    - Simpler training
    - Deterministic quantization

    Args:
        num_quantizers: Number of residual quantization stages
        fsq_levels: Quantization levels per dimension for each stage
        code_dim: Code dimension
        shared_codebook: Whether to share FSQ across stages (usually False)
        quantize_dropout_prob: Probability of dropping later quantizers
        quantize_dropout_cutoff_index: Always keep first N quantizers
    """

    def __init__(
        self,
        num_quantizers: int = 6,
        fsq_levels: List[int] = [5, 5, 5, 5],
        code_dim: int = 4,
        shared_codebook: bool = False,
        quantize_dropout_prob: float = 0.2,
        quantize_dropout_cutoff_index: int = 0,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.quantize_dropout_prob = quantize_dropout_prob
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.code_dim = code_dim
        self.codebook_size = int(torch.tensor(fsq_levels).prod().item())

        # Create FSQ quantizers for each stage
        if shared_codebook:
            # Share a single FSQ across all stages
            self.quantizers = nn.ModuleList([
                FSQ(
                    levels=fsq_levels,
                    dim=code_dim,
                    num_codebooks=1,
                    force_quantization_f32=True,
                    return_indices=True
                )
            ] * num_quantizers)
        else:
            # Separate FSQ for each stage
            self.quantizers = nn.ModuleList([
                FSQ(
                    levels=fsq_levels,
                    dim=code_dim,
                    num_codebooks=1,
                    force_quantization_f32=True,
                    return_indices=True
                )
                for _ in range(num_quantizers)
            ])

    def _get_quantizer(self, index: int) -> FSQ:
        """Get quantizer at given index"""
        return self.quantizers[index]

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], Tensor, Tensor]:
        """
        Residual quantization with FSQ

        Args:
            x: Input tensor [B, code_dim, T']

        Returns:
            - Quantized output (sum of all stages)
            - List of indices for each stage [num_quantizers × [B*T']]
            - Total commitment loss (0 for FSQ)
            - Average perplexity across stages
        """
        B, C, T_prime = x.shape

        # Reshape to [B, T', C] for FSQ
        x = x.permute(0, 2, 1)  # [B, T', C]

        # Determine how many quantizers to use (for dropout during training)
        if self.training and self.quantize_dropout_prob > 0:
            n_quantizers = torch.randint(
                self.quantize_dropout_cutoff_index + 1,
                self.num_quantizers + 1,
                (1,)
            ).item()
            if torch.rand(1).item() > self.quantize_dropout_prob:
                n_quantizers = self.num_quantizers
        else:
            n_quantizers = self.num_quantizers

        residual = x.clone()
        x_quantized_sum = torch.zeros_like(x)
        all_indices = []
        total_perplexity = 0.0
        total_loss = torch.tensor(0.0, device=x.device)

        for i in range(n_quantizers):
            # Quantize residual
            quantized, indices, (loss, perplexity) = self.quantizers[i](
                residual, return_loss_breakdown=True
            )

            # Accumulate quantized output
            x_quantized_sum = x_quantized_sum + quantized

            # Update residual
            residual = residual - quantized

            # Store indices (flatten for compatibility)
            all_indices.append(indices.reshape(-1))

            # Accumulate metrics
            total_loss = total_loss + loss
            total_perplexity = total_perplexity + perplexity

        # Average perplexity
        avg_perplexity = total_perplexity / n_quantizers

        # Reshape back to [B, C, T']
        x_quantized_sum = x_quantized_sum.permute(0, 2, 1)

        return x_quantized_sum, all_indices, total_loss, avg_perplexity

    def quantize(self, x: Tensor) -> List[Tensor]:
        """
        Quantize input and return indices for all stages

        Args:
            x: Input tensor [B, code_dim, T']

        Returns:
            List of indices [num_quantizers × [B*T']]
        """
        B, C, T_prime = x.shape

        # Reshape to [B, T', C]
        x = x.permute(0, 2, 1)

        residual = x.clone()
        all_indices = []

        for i in range(self.num_quantizers):
            # Quantize residual
            quantized, indices, _ = self.quantizers[i](residual)

            # Update residual
            residual = residual - quantized

            # Store indices
            all_indices.append(indices.reshape(-1))

        return all_indices

    def dequantize(self, all_indices: List[Tensor]) -> Tensor:
        """
        Decode indices from all stages

        Args:
            all_indices: List of indices [num_quantizers × [B*T']]

        Returns:
            Quantized tensor [B, code_dim, T']
        """
        # Infer shape from first indices
        B_T_prime = all_indices[0].shape[0]

        # Decode each stage and sum
        x_quantized_sum = None

        for i, indices in enumerate(all_indices):
            # Decode indices to continuous codes
            quantized = self.quantizers[i].get_codes_from_indices(
                indices.view(-1)
            )  # [B*T', C]

            if x_quantized_sum is None:
                x_quantized_sum = quantized
            else:
                x_quantized_sum = x_quantized_sum + quantized

        # Reshape to [B, T', C] and then [B, C, T']
        # We need to infer B and T'
        # For now, assume B=1 (will be fixed in RVQVae.decode)
        return x_quantized_sum


class FSQRVQVae(nn.Module):
    """
    VQ-VAE with Residual FSQ

    Architecture:
        Input [B, T, D]
            ↓ Encoder
        [B, code_dim, T']
            ↓ Residual FSQ (num_quantizers stages)
        [B, code_dim, T'], indices[num_quantizers × B*T']
            ↓ Decoder
        Output [B, T, D]

    Args:
        nfeats: Number of input features (e.g., 133 for SMPLX)
        num_quantizers: Number of residual quantization stages
        fsq_levels: Quantization levels per dimension
        code_dim: Latent code dimension
        output_emb_width: Encoder output width
        ... (other encoder/decoder parameters)
    """

    def __init__(
        self,
        nfeats: int,
        num_quantizers: int = 6,
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
        quantize_dropout_prob: float = 0.2,
        quantize_dropout_cutoff_index: int = 0,
        shared_codebook: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.code_dim = code_dim
        self.num_quantizers = num_quantizers
        self.code_num = int(torch.tensor(fsq_levels).prod().item())

        # Encoder: [B, nfeats, T] -> [B, code_dim, T']
        # Note: We use code_dim as output_emb_width for FSQ
        self.encoder = Encoder(
            nfeats,
            code_dim,  # Output matches FSQ input
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm
        )

        # Residual FSQ Quantizer
        self.quantizer = FSQResidualVQ(
            num_quantizers=num_quantizers,
            fsq_levels=fsq_levels,
            code_dim=code_dim,
            shared_codebook=shared_codebook,
            quantize_dropout_prob=quantize_dropout_prob,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index
        )

        # Decoder: [B, code_dim, T'] -> [B, nfeats, T]
        self.decoder = Decoder(
            nfeats,
            code_dim,  # Input matches FSQ output
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
        Forward pass through FSQ-RVQVAE

        Args:
            features: Input motion features [B, T, D]

        Returns:
            x_out: Reconstructed motion [B, T, D]
            loss: Commitment loss (0 for FSQ)
            perplexity: Average perplexity across stages
        """
        # Preprocess: [B, T, D] -> [B, D, T]
        x_in = self.preprocess(features)

        # Encode: [B, D, T] -> [B, code_dim, T']
        x_encoder = self.encoder(x_in)

        # Residual quantization
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
            code_idx: Discrete codes [B, T', num_quantizers]
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
        """
        B, T, D = features.shape
        x_in = self.preprocess(features)  # [B, D, T]
        x_encoder = self.encoder(x_in)     # [B, code_dim, T']

        # Pool over time
        x_pooled = x_encoder.mean(dim=-1)  # [B, code_dim]

        return x_pooled

    def decode(self, code_idx: Tensor) -> Tensor:
        """
        Decode discrete codes to motion

        Args:
            code_idx: Discrete codes [B, T', num_quantizers] or [T', num_quantizers]

        Returns:
            x_out: Reconstructed motion [B, T, D]
        """
        # Handle different input shapes
        if code_idx.dim() == 1:
            # [T'] -> [1, T', num_quantizers]
            code_idx = code_idx.unsqueeze(0).unsqueeze(-1).repeat(1, 1, self.num_quantizers)
        elif code_idx.dim() == 2:
            # [T', num_quantizers] -> [1, T', num_quantizers]
            if code_idx.shape[-1] == self.num_quantizers:
                code_idx = code_idx.unsqueeze(0)
            else:
                # [B, T'] -> [B, T', num_quantizers]
                code_idx = code_idx.unsqueeze(-1).repeat(1, 1, self.num_quantizers)

        B, T_prime, num_quantizers = code_idx.shape

        # Split codes into list: [B, T', num_quantizers] -> List of [B*T']
        all_indices = []
        for i in range(num_quantizers):
            indices = code_idx[:, :, i]  # [B, T']
            indices_flat = indices.reshape(-1)  # [B*T']
            all_indices.append(indices_flat)

        # Dequantize: sum all stages
        x_quantized = None
        for i, indices in enumerate(all_indices):
            quantizer = self.quantizer._get_quantizer(i)
            z_q = quantizer.get_codes_from_indices(indices)  # [B*T', code_dim]

            # Reshape to [B, T', code_dim]
            z_q = z_q.view(B, T_prime, self.code_dim)

            if x_quantized is None:
                x_quantized = z_q
            else:
                x_quantized = x_quantized + z_q

        # Reshape for decoder: [B, T', code_dim] -> [B, code_dim, T']
        x_quantized = x_quantized.permute(0, 2, 1)

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
