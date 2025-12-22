"""
Hierarchical Residual VQ-VAE (HRVQVAE) for MotionGPT.

This is a direct port of vq/rvq_model.py to be compatible with MotionGPT's
config system while maintaining the exact same model structure.
"""

import random
import torch
import torch.nn as nn
from .tools.encoder_attn import EncoderAttn, DecoderAttn, length_to_mask
from .tools.hr_quantize import HRQuantizeEMAReset, HRQuantizeEMAResetV2


class HRVQVae(nn.Module):
    """
    Hierarchical Residual VQ-VAE - matches vq/rvq_model.py structure exactly.

    Args:
        nfeats: Input feature dimension (replaces input_width)
        down_t: Number of temporal downsampling stages
        stride_t: Stride for downsampling
        width: Channel width in encoder/decoder
        depth: ResNet depth per stage
        dilation_growth_rate: Dilation growth rate
        activation: Activation function
        use_attn: Whether to use self-attention
        norm: Normalization type

        # Quantizer params (replaces args.quantizer.*)
        code_num: Codebook size (nb_code)
        code_dim: Code embedding dimension
        mu: EMA decay rate
        scales: List of temporal scale factors
        quantizer_version: 'v1' or 'v2'
        share_quant_resi: Phi sharing mode for V2
        quant_resi: Residual ratio for Phi in V2
        start_drop: Start index for quantizer dropout
        quantize_dropout_prob: Probability of dropping quantizers
    """
    def __init__(self,
                 nfeats=263,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 use_attn=False,
                 norm=None,
                 # Quantizer params
                 code_num=512,
                 code_dim=512,
                 mu=0.99,
                 scales=[1, 2, 4, 8],
                 quantizer_version='v1',
                 share_quant_resi=4,
                 quant_resi=0.5,
                 start_drop=1,
                 quantize_dropout_prob=0.0,
                 **kwargs):

        super().__init__()

        output_emb_width = code_dim

        self.encoder = EncoderAttn(nfeats, output_emb_width, down_t, stride_t, width, depth,
                                   dilation_growth_rate, activation=activation, norm=norm, use_attn=use_attn)
        self.decoder = DecoderAttn(nfeats, output_emb_width, down_t, stride_t, width, depth,
                                   dilation_growth_rate, activation=activation, norm=norm, use_attn=use_attn)

        # Store config for forward pass
        self.start_drop = start_drop
        self.quantize_dropout_prob = quantize_dropout_prob
        self.down_t = down_t
        self.code_dim = code_dim
        self.scales = scales

        if quantizer_version == 'v2':
            self.quantizer = HRQuantizeEMAResetV2(nb_code=code_num,
                                                  code_dim=code_dim,
                                                  mu=mu,
                                                  scales=scales,
                                                  share_quant_resi=share_quant_resi,
                                                  quant_resi=quant_resi)
        else:
            self.quantizer = HRQuantizeEMAReset(nb_code=code_num,
                                                code_dim=code_dim,
                                                mu=mu,
                                                scales=scales)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x, m_lens=None):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in, m_lens)

        if m_lens is not None:
            m_lens = m_lens // (2 ** self.down_t)

        code_idx, all_codes = self.quantizer.quantize_all(x_encoder, m_lens, return_latent=True)
        # code_idx: List of [B, T_scale] indices per scale
        # all_codes: [B, C, T] reconstructed features
        return code_idx, all_codes

    def forward(self, x, m_lengths=None):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in, m_lengths)

        if m_lengths is not None:
            m_lengths_enc = m_lengths // (2 ** self.down_t)
        else:
            m_lengths_enc = None

        # Quantization
        x_quantized, commit_loss, perplexity = self.quantizer(x_encoder,
                                                              temperature=0.5,
                                                              m_lens=m_lengths_enc,
                                                              start_drop=self.start_drop,
                                                              quantize_dropout_prob=self.quantize_dropout_prob)

        if m_lengths_enc is not None:
            x_quantized = x_quantized.permute(0, 2, 1)
            mask = length_to_mask(m_lengths_enc, x_quantized.shape[1], x_quantized.device)
            x_quantized[~mask] = 0
            x_quantized = x_quantized.permute(0, 2, 1)

        # Decoder
        x_out = self.decoder(x_quantized, m_lengths_enc)

        return x_out, commit_loss, perplexity

    def forward_decoder(self, x, m_lengths=None):
        x_d = self.quantizer.get_codes_from_indices(x)

        if len(x_d.shape) == 4:
            x_d = x_d.sum(dim=0)

        if m_lengths is not None:
            m_lengths_enc = m_lengths // (2 ** self.down_t)
            mask = length_to_mask(m_lengths_enc, x_d.shape[1], x_d.device)
            x_d[~mask] = 0
        else:
            m_lengths_enc = None

        x_d = x_d.permute(0, 2, 1)

        # Decoder
        x_out = self.decoder(x_d, m_lengths_enc)

        return x_out

    def decode(self, x, m_lengths=None):
        """Decode from continuous latent features."""
        if m_lengths is not None:
            x = x.permute(0, 2, 1)
            m_lengths_enc = m_lengths // (2 ** self.down_t)
            mask = length_to_mask(m_lengths_enc, x.shape[1], x.device)
            x[~mask] = 0
            x = x.permute(0, 2, 1)
        else:
            m_lengths_enc = None

        # Decoder
        x_out = self.decoder(x, m_lengths_enc)

        return x_out

    # ===================== VAR-related methods =====================
    def get_var_input(self, features, m_lens=None):
        """Get VAR training inputs from features."""
        code_idx_list, _ = self.encode(features, m_lens)
        return self.quantizer.idx_to_var_input(code_idx_list)

    def get_next_var_input(self, level, indices, code, T):
        """Get next VAR inference input."""
        return self.quantizer.get_next_var_input(level, indices, code, T)

    # ===================== Helper methods =====================
    def get_codebook(self):
        """Return the codebook embeddings."""
        return self.quantizer.codebook

    def get_num_scales(self):
        """Return the number of quantization scales."""
        return len(self.scales)

    def get_scales(self):
        """Return the scale factors."""
        return self.scales

    def get_codebook_size(self):
        """Return the codebook size."""
        return self.quantizer.nb_code
