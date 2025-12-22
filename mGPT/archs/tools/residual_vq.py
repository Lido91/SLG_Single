"""
Residual Vector Quantization for Motion Tokenization

Based on:
- SoundStream: https://arxiv.org/pdf/2107.03312.pdf
- SemTalk implementation

Multi-stage quantization where each stage quantizes the residual
from the previous stage, enabling hierarchical representation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization

    Applies multiple quantization stages sequentially:
    1. Quantize input: z_0 = Q_0(x)
    2. Compute residual: r_0 = x - z_0
    3. Quantize residual: z_1 = Q_1(r_0)
    4. Compute residual: r_1 = r_0 - z_1
    5. Repeat for N stages
    6. Final output: sum(z_0, z_1, ..., z_N)

    Args:
        num_quantizers: Number of quantization stages (default: 6)
        shared_codebook: Whether to share codebook across stages (default: False)
        quantize_dropout_prob: Probability of dropping quantizers during training (default: 0.2)
        quantize_dropout_cutoff_index: Always keep first N quantizers (default: 0)
        nb_code: Codebook size per quantizer (default: 512)
        code_dim: Code dimension (default: 512)
        quantizer_type: Type of quantizer ('ema_reset', 'orig', 'ema', 'reset')
        mu: EMA decay rate for EMA-based quantizers (default: 0.99)
    """

    def __init__(
        self,
        num_quantizers: int = 6,
        shared_codebook: bool = False,
        quantize_dropout_prob: float = 0.2,
        quantize_dropout_cutoff_index: int = 0,
        nb_code: int = 512,
        code_dim: int = 512,
        quantizer_type: str = 'ema_reset',
        mu: float = 0.99,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.shared_codebook = shared_codebook
        self.quantize_dropout_prob = quantize_dropout_prob
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.nb_code = nb_code
        self.code_dim = code_dim

        # Create quantizers
        if shared_codebook:
            # Single shared quantizer
            self.quantizers = nn.ModuleList([
                self._create_quantizer(quantizer_type, nb_code, code_dim, mu)
                for _ in range(1)
            ])
        else:
            # Separate quantizer for each stage
            self.quantizers = nn.ModuleList([
                self._create_quantizer(quantizer_type, nb_code, code_dim, mu)
                for _ in range(num_quantizers)
            ])

    def _create_quantizer(self, quantizer_type: str, nb_code: int, code_dim: int, mu: float):
        """Create a single quantizer based on type"""
        if quantizer_type == "ema_reset":
            return QuantizeEMAReset(nb_code, code_dim, mu=mu)
        elif quantizer_type == "orig":
            return Quantizer(nb_code, code_dim, beta=1.0)
        elif quantizer_type == "ema":
            return QuantizeEMA(nb_code, code_dim, mu=mu)
        elif quantizer_type == "reset":
            return QuantizeReset(nb_code, code_dim)
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}")

    def _get_quantizer(self, index: int):
        """Get quantizer at index (handles shared codebook case)"""
        if self.shared_codebook:
            return self.quantizers[0]
        else:
            return self.quantizers[index]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual quantization

        Args:
            x: Input tensor [B, C, T]

        Returns:
            quantized_out: Accumulated quantized output [B, C, T]
            all_indices: List of code indices per stage, each [B*T]
            commit_loss: Mean commitment loss across stages
            perplexity: Mean perplexity across stages
        """
        B, C, T = x.shape

        # Determine number of quantizers to use (with dropout)
        num_quantizers_to_use = self.num_quantizers
        if self.training and self.quantize_dropout_prob > 0:
            # Randomly determine how many quantizers to use
            # Always keep at least cutoff_index + 1 quantizers
            min_quantizers = self.quantize_dropout_cutoff_index + 1
            if torch.rand(1).item() < self.quantize_dropout_prob:
                num_quantizers_to_use = torch.randint(
                    min_quantizers,
                    self.num_quantizers + 1,
                    (1,)
                ).item()

        # Initialize
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices = []
        all_losses = []
        all_perplexities = []

        # Residual quantization loop
        for i in range(num_quantizers_to_use):
            quantizer = self._get_quantizer(i)

            # Quantize current residual
            z_q, loss, perplexity = quantizer(residual)

            # Store indices for this stage
            # Need to extract indices from quantizer
            with torch.no_grad():
                indices = self._extract_indices(quantizer, residual)
                all_indices.append(indices)

            # Update residual (detach to stop gradient flow)
            residual = residual - z_q.detach()

            # Accumulate quantized output
            quantized_out = quantized_out + z_q

            # Collect losses and perplexities
            all_losses.append(loss)
            all_perplexities.append(perplexity)

        # Aggregate losses and perplexities
        commit_loss = torch.mean(torch.stack(all_losses))
        mean_perplexity = torch.mean(torch.stack(all_perplexities))

        return quantized_out, all_indices, commit_loss, mean_perplexity

    def _extract_indices(self, quantizer, x: torch.Tensor) -> torch.Tensor:
        """
        Extract discrete indices from quantizer

        Args:
            quantizer: Quantizer module
            x: Input tensor [B, C, T]

        Returns:
            indices: Discrete codes [B*T] or [B, T]
        """
        # Preprocess to [B*T, C]
        x_preprocessed = quantizer.preprocess(x)

        # Quantize to get indices
        indices = quantizer.quantize(x_preprocessed)

        return indices

    def quantize(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Quantize input and return discrete codes for all stages

        Args:
            x: Input tensor [B, C, T]

        Returns:
            all_indices: List of code indices per stage, each [B*T]
        """
        residual = x
        all_indices = []

        for i in range(self.num_quantizers):
            quantizer = self._get_quantizer(i)

            # Get quantized version
            z_q, _, _ = quantizer(residual)

            # Extract indices
            with torch.no_grad():
                indices = self._extract_indices(quantizer, residual)
                all_indices.append(indices)

            # Update residual
            residual = residual - z_q.detach()

        return all_indices

    def dequantize(self, all_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Dequantize codes back to continuous representation

        Args:
            all_indices: List of code indices per stage, each [B*T] or [B, T]

        Returns:
            x: Reconstructed tensor [B, C, T]
        """
        quantized_out = None

        for i, indices in enumerate(all_indices):
            quantizer = self._get_quantizer(i)

            # Dequantize this stage
            z_q = quantizer.dequantize(indices)

            # Reshape to [B, C, T] if needed
            if z_q.dim() == 2:
                # [B*T, C] -> need to infer B and T
                # This is tricky - we'll assume we can infer from indices shape
                B_times_T, C = z_q.shape
                # For now, reshape to [1, C, B*T] as placeholder
                z_q = z_q.view(1, -1, C).permute(0, 2, 1).contiguous()

            # Accumulate
            if quantized_out is None:
                quantized_out = z_q
            else:
                quantized_out = quantized_out + z_q

        return quantized_out

    @property
    def codebooks(self) -> torch.Tensor:
        """
        Get all codebook embeddings stacked

        Returns:
            codebooks: [num_quantizers, nb_code, code_dim]
        """
        codebooks = []
        for i in range(self.num_quantizers):
            quantizer = self._get_quantizer(i)
            # Each quantizer has an embedding table
            codebooks.append(quantizer.codebook)

        return torch.stack(codebooks, dim=0)  # [num_q, nb_code, code_dim]

    def get_codes_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get codebook embeddings from discrete indices (ReMoMask-style interface)

        This method is compatible with ReMoMask's forward_decoder() interface.
        It looks up the actual codebook vectors for given indices.

        Args:
            indices: [B, T', num_quantizers] - Discrete code indices

        Returns:
            all_codes: [num_quantizers, B, T', code_dim] - Codebook embeddings

        Example:
            >>> indices = torch.randint(0, 512, (2, 49, 6))  # B=2, T'=49, Q=6
            >>> codes = rvq.get_codes_from_indices(indices)
            >>> codes.shape
            torch.Size([6, 2, 49, 512])  # [Q, B, T', C]
        """
        from einops import repeat

        B, T, num_quant = indices.shape
        assert num_quant <= self.num_quantizers, \
            f"Indices have {num_quant} quantizers but model has {self.num_quantizers}"

        # Handle quantizer dropout (pad with -1 if fewer quantizers)
        if num_quant < self.num_quantizers:
            padding = torch.full(
                (B, T, self.num_quantizers - num_quant),
                -1,
                dtype=indices.dtype,
                device=indices.device
            )
            indices = torch.cat([indices, padding], dim=-1)

        # Get codebooks: [num_q, nb_code, code_dim]
        codebooks = self.codebooks

        # Prepare for gathering
        codebooks = repeat(codebooks, 'q c d -> q b c d', b=B)  # [Q, B, nb_code, code_dim]
        gather_indices = repeat(indices, 'b t q -> q b t d', d=self.code_dim)  # [Q, B, T', code_dim]

        # Handle dropout (-1 indices)
        mask = gather_indices == -1
        gather_indices = gather_indices.masked_fill(mask, 0)  # Replace -1 with 0 (dummy)

        # Gather codes from codebooks
        all_codes = codebooks.gather(2, gather_indices)  # [Q, B, T', code_dim]

        # Mask out dropped quantizers
        all_codes = all_codes.masked_fill(mask, 0.0)

        return all_codes  # [num_quantizers, B, T', code_dim]


class ResidualVQ_2D(nn.Module):
    """
    2D Residual Vector Quantization for joint-temporal representation

    Similar to ResidualVQ but operates on [B, C, J, T] tensors where:
    - B: batch size
    - C: channel dimension
    - J: number of joints
    - T: time steps

    This is useful for models that want to quantize across both spatial (joints)
    and temporal dimensions simultaneously.
    """

    def __init__(
        self,
        num_quantizers: int = 6,
        shared_codebook: bool = False,
        quantize_dropout_prob: float = 0.2,
        quantize_dropout_cutoff_index: int = 0,
        nb_code: int = 512,
        code_dim: int = 512,
        quantizer_type: str = 'ema_reset',
        mu: float = 0.99,
        **kwargs
    ):
        super().__init__()

        # Use the 1D ResidualVQ as the core
        self.residual_vq = ResidualVQ(
            num_quantizers=num_quantizers,
            shared_codebook=shared_codebook,
            quantize_dropout_prob=quantize_dropout_prob,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
            nb_code=nb_code,
            code_dim=code_dim,
            quantizer_type=quantizer_type,
            mu=mu,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass for 2D input

        Args:
            x: Input tensor [B, C, J, T]

        Returns:
            quantized_out: Accumulated quantized output [B, C, J, T]
            all_indices: List of code indices per stage
            commit_loss: Mean commitment loss
            perplexity: Mean perplexity
        """
        B, C, J, T = x.shape

        # Reshape to [B, C, J*T] for processing
        x_flat = x.reshape(B, C, J * T)

        # Apply residual quantization
        quantized_out, all_indices, commit_loss, perplexity = self.residual_vq(x_flat)

        # Reshape back to [B, C, J, T]
        quantized_out = quantized_out.reshape(B, C, J, T)

        return quantized_out, all_indices, commit_loss, perplexity

    def quantize(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Quantize 2D input"""
        B, C, J, T = x.shape
        x_flat = x.reshape(B, C, J * T)
        return self.residual_vq.quantize(x_flat)

    def dequantize(self, all_indices: List[torch.Tensor], shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Dequantize codes back to 2D representation

        Args:
            all_indices: List of code indices per stage
            shape: Target shape [B, C, J, T]

        Returns:
            x: Reconstructed tensor [B, C, J, T]
        """
        B, C, J, T = shape
        x_flat = self.residual_vq.dequantize(all_indices)
        return x_flat.reshape(B, C, J, T)

