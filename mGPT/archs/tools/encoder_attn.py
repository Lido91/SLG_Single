"""
Encoder and Decoder with optional self-attention for HRVQVAE.

These modules extend the standard Conv1D encoder/decoder with optional
self-attention layers for better temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import Resnet1D


def length_to_mask(length, max_len=None, device=None):
    """Convert sequence lengths to attention mask."""
    if device is None:
        device = length.device if isinstance(length, torch.Tensor) else "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)

    length = length.to(device)

    if max_len is None:
        max_len = int(length.max().item())

    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


class SelfAttention1D(nn.Module):
    """
    Self-attention module for 1D sequences.

    Input: [B, C, T]
    Output: [B, C, T]
    """
    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, C, T]
            mask: [B, T] boolean mask (True = valid, False = padding)
        Returns:
            [B, C, T]
        """
        B, C, T = x.shape

        # Transpose for attention: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)

        # Layer norm
        x_norm = self.norm(x)

        # QKV projection
        qkv = self.qkv(x_norm).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, T, T]

        # Apply mask if provided
        if mask is not None:
            # mask: [B, T] -> [B, 1, 1, T] for broadcasting
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)  # [B, T, C]
        out = self.proj(out)
        out = self.dropout(out)

        # Residual connection
        out = x + out

        # Transpose back: [B, T, C] -> [B, C, T]
        return out.permute(0, 2, 1)


class EncoderAttn(nn.Module):
    """
    Encoder with optional self-attention layers.

    Architecture:
        Conv1d -> ReLU -> [DownBlock + ResNet + (optional)Attention] x down_t -> Conv1d

    Args:
        input_emb_width: Input feature dimension
        output_emb_width: Output embedding dimension
        down_t: Number of downsampling stages
        stride_t: Stride for downsampling
        width: Channel width
        depth: ResNet depth per stage
        dilation_growth_rate: Dilation growth rate
        activation: Activation function
        norm: Normalization type
        use_attn: Whether to use self-attention
        num_heads: Number of attention heads
    """
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        use_attn=False,
        num_heads=8
    ):
        super().__init__()

        self.down_t = down_t
        self.stride_t = stride_t
        self.use_attn = use_attn

        # Initial conv
        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.init_conv = nn.Conv1d(input_emb_width, width, 3, 1, 1)
        self.init_act = nn.ReLU()

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.resnet_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList() if use_attn else None

        for i in range(down_t):
            # Downsample conv
            self.down_blocks.append(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t)
            )
            # ResNet block
            self.resnet_blocks.append(
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm)
            )
            # Optional attention
            if use_attn:
                self.attn_blocks.append(SelfAttention1D(width, num_heads=num_heads))

        # Output conv
        self.out_conv = nn.Conv1d(width, output_emb_width, 3, 1, 1)

    def forward(self, x, m_lens=None):
        """
        Args:
            x: [B, C, T] input features
            m_lens: [B] sequence lengths (optional)
        Returns:
            [B, output_emb_width, T'] encoded features
        """
        # Initial conv
        x = self.init_conv(x)
        x = self.init_act(x)

        # Downsampling with optional attention
        for i in range(self.down_t):
            x = self.down_blocks[i](x)
            x = self.resnet_blocks[i](x)

            if self.use_attn and self.attn_blocks is not None:
                # Create mask for attention if lengths provided
                if m_lens is not None:
                    # Adjust lengths for downsampling
                    current_lens = m_lens // (self.stride_t ** (i + 1))
                    mask = length_to_mask(current_lens, x.shape[-1], x.device)
                else:
                    mask = None
                x = self.attn_blocks[i](x, mask)

        # Output conv
        x = self.out_conv(x)

        return x


class DecoderAttn(nn.Module):
    """
    Decoder with optional self-attention layers.

    Architecture:
        Conv1d -> ReLU -> [(optional)Attention + ResNet + Upsample + Conv1d] x down_t -> Conv1d -> Conv1d

    Args:
        input_emb_width: Output feature dimension (motion features)
        output_emb_width: Input embedding dimension (from quantizer)
        down_t: Number of upsampling stages
        stride_t: Upsample scale factor
        width: Channel width
        depth: ResNet depth per stage
        dilation_growth_rate: Dilation growth rate
        activation: Activation function
        norm: Normalization type
        use_attn: Whether to use self-attention
        num_heads: Number of attention heads
    """
    def __init__(
        self,
        input_emb_width=3,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        use_attn=False,
        num_heads=8
    ):
        super().__init__()

        self.down_t = down_t
        self.stride_t = stride_t
        self.use_attn = use_attn

        # Initial conv
        self.init_conv = nn.Conv1d(output_emb_width, width, 3, 1, 1)
        self.init_act = nn.ReLU()

        # Upsampling blocks
        self.attn_blocks = nn.ModuleList() if use_attn else None
        self.resnet_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(down_t):
            # Optional attention
            if use_attn:
                self.attn_blocks.append(SelfAttention1D(width, num_heads=num_heads))
            # ResNet block (reverse dilation)
            self.resnet_blocks.append(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True,
                        activation=activation, norm=norm)
            )
            # Upsample
            self.upsample_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
            # Conv after upsample
            self.up_convs.append(nn.Conv1d(width, width, 3, 1, 1))

        # Output convs
        self.out_conv1 = nn.Conv1d(width, width, 3, 1, 1)
        self.out_act = nn.ReLU()
        self.out_conv2 = nn.Conv1d(width, input_emb_width, 3, 1, 1)

    def forward(self, x, m_lens=None):
        """
        Args:
            x: [B, C, T'] quantized features
            m_lens: [B] target sequence lengths (optional, for attention mask)
        Returns:
            [B, input_emb_width, T] decoded motion features
        """
        # Initial conv
        x = self.init_conv(x)
        x = self.init_act(x)

        # Upsampling with optional attention
        for i in range(self.down_t):
            if self.use_attn and self.attn_blocks is not None:
                # Create mask for attention if lengths provided
                if m_lens is not None:
                    # Current temporal resolution
                    current_lens = m_lens // (self.stride_t ** (self.down_t - i))
                    mask = length_to_mask(current_lens, x.shape[-1], x.device)
                else:
                    mask = None
                x = self.attn_blocks[i](x, mask)

            x = self.resnet_blocks[i](x)
            x = self.upsample_blocks[i](x)
            x = self.up_convs[i](x)

        # Output convs
        x = self.out_conv1(x)
        x = self.out_act(x)
        x = self.out_conv2(x)

        return x
