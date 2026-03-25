"""
VQ-Style RVQ-VAE: RVQ-VAE with Contrastive + Mutual Information losses

Based on VQ-Style (arXiv 2602.02334):
- Contrastive loss on residual codebooks (Q1+) — organizes style quantizers by labels
- Mutual information (MI) loss on content codebook (Q0) — prevents label info leaking into Q0

Adapted for sign language: text/gloss cluster labels replace style labels.
Q0 = text-predictable content; Q1+ = signer-specific style.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import Tensor

from .mgpt_rvq import RVQVae


class RVQVaeVQStyle(RVQVae):
    """
    RVQ-VAE extended with VQ-Style contrastive and mutual information losses.

    Additional Args (beyond RVQVae):
        content_cutoff: Number of content quantizers (Q0..Q_{cutoff-1}). Default: 1
        lambda_con: Weight for contrastive loss. Default: 0.005
        lambda_mi: Weight for MI loss. Default: 0.02
        tau_con: Temperature for contrastive similarity. Default: 0.07
        tau_mi: Temperature for MI soft assignment. Default: 1.0
        num_clusters: Number of text clusters. Default: 64
        cluster_labels_path: Path to JSON mapping sample_name -> cluster_id
    """

    def __init__(
        self,
        content_cutoff: int = 1,
        lambda_con: float = 0.005,
        lambda_mi: float = 0.02,
        tau_con: float = 0.07,
        tau_mi: float = 1.0,
        num_clusters: int = 64,
        cluster_labels_path: str = '',
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.content_cutoff = content_cutoff
        self.lambda_con = lambda_con
        self.lambda_mi = lambda_mi
        self.tau_con = tau_con
        self.tau_mi = tau_mi
        self.num_clusters = num_clusters

        # Load pre-computed cluster labels
        self.cluster_labels = {}
        if cluster_labels_path:
            try:
                with open(cluster_labels_path, 'r') as f:
                    raw = json.load(f)
                # Convert values to int
                self.cluster_labels = {k: int(v) for k, v in raw.items()}
                print(f"[RVQVaeVQStyle] Loaded {len(self.cluster_labels)} cluster labels "
                      f"from {cluster_labels_path}")
            except Exception as e:
                print(f"[RVQVaeVQStyle] WARNING: Could not load cluster labels "
                      f"from {cluster_labels_path}: {e}")

    def _get_batch_labels(self, names: Optional[List[str]], device: torch.device) -> Optional[Tensor]:
        """
        Look up cluster labels for a batch of sample names.

        Returns:
            labels: [B] long tensor, or None if labels unavailable
        """
        if names is None or len(self.cluster_labels) == 0:
            return None

        labels = []
        for name in names:
            if name in self.cluster_labels:
                labels.append(self.cluster_labels[name])
            else:
                return None  # Missing label → skip VQ-Style losses for this batch
        return torch.tensor(labels, dtype=torch.long, device=device)

    def _contrastive_loss(self, all_z_q: List[Tensor], labels: Tensor) -> Tensor:
        """
        Multi-positive contrastive loss on style quantizer embeddings (Q_{cutoff}..Q_N).

        Args:
            all_z_q: List of per-quantizer embeddings, each [B, C, T']
            labels: [B] cluster labels

        Returns:
            loss: Scalar contrastive loss
        """
        # Select style quantizers: Q_{cutoff} onwards
        style_z_q = all_z_q[self.content_cutoff:]
        if len(style_z_q) == 0:
            return torch.tensor(0.0, device=labels.device)

        # Pool: mean over quantizers, then mean over time → [B, C]
        # Stack style quantizers: [num_style, B, C, T'] → mean over dim 0 → [B, C, T']
        style_pooled = torch.stack(style_z_q, dim=0).mean(dim=0)  # [B, C, T']
        style_pooled = style_pooled.mean(dim=-1)  # [B, C]

        # L2 normalize
        style_pooled = F.normalize(style_pooled, p=2, dim=-1)  # [B, C]

        B = style_pooled.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=labels.device)

        # Similarity matrix: [B, B]
        sim = torch.mm(style_pooled, style_pooled.t()) / self.tau_con  # [B, B]

        # Build positive mask: same label (excluding diagonal)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        diag_mask = ~torch.eye(B, dtype=torch.bool, device=labels.device)
        pos_mask = label_eq & diag_mask  # [B, B]

        # Check if any positives exist
        num_pos_per_sample = pos_mask.sum(dim=1)  # [B]
        has_positives = num_pos_per_sample > 0
        if not has_positives.any():
            return torch.tensor(0.0, device=labels.device)

        # Multi-positive InfoNCE:
        # For each anchor i with positives, loss = -mean_j(log(softmax(sim)[i,j]))
        # = -mean_j(sim[i,j] - logsumexp(sim[i,:]))
        # We mask out the diagonal from the denominator
        neg_inf = torch.finfo(sim.dtype).min
        sim_masked = sim.masked_fill(~diag_mask, neg_inf)  # mask diagonal
        log_softmax = sim_masked - torch.logsumexp(sim_masked, dim=1, keepdim=True)  # [B, B]

        # Average log-softmax over positive pairs per anchor
        pos_log_softmax = log_softmax * pos_mask.float()
        loss_per_sample = -pos_log_softmax.sum(dim=1) / num_pos_per_sample.clamp(min=1)

        # Average only over samples that have positives
        loss = loss_per_sample[has_positives].mean()
        return loss

    def _mi_loss(self, all_residuals: List[Tensor], labels: Tensor) -> Tensor:
        """
        Mutual information loss on content quantizers (Q0..Q_{cutoff-1}).

        Estimates MI(z; l) where z is the quantizer assignment and l is the label,
        using soft assignments from pre-quantization residuals.

        Args:
            all_residuals: List of pre-quantization residuals, each [B, C, T']
            labels: [B] cluster labels

        Returns:
            loss: Scalar MI estimate
        """
        mi_total = torch.tensor(0.0, device=labels.device)
        num_content = min(self.content_cutoff, len(all_residuals))

        if num_content == 0:
            return mi_total

        B = labels.shape[0]
        unique_labels = labels.unique()
        L = unique_labels.shape[0]

        if L < 2:
            return mi_total  # Need at least 2 labels for MI

        for i in range(num_content):
            residual = all_residuals[i]  # [B, C, T']
            quantizer = self.quantizer._get_quantizer(i)
            codebook = quantizer.codebook  # [nb_code, C]

            # Preprocess residual: [B, C, T'] → [B*T', C]
            B_r, C, T_prime = residual.shape
            r_flat = residual.permute(0, 2, 1).contiguous().view(-1, C)  # [B*T', C]

            # Soft assignment: q(z=c_k|r) = softmax(-||r - c_k||^2 / tau_mi)
            # Distance: [B*T', nb_code]
            dist = torch.sum(r_flat ** 2, dim=1, keepdim=True) \
                   - 2 * torch.mm(r_flat, codebook.t()) \
                   + torch.sum(codebook ** 2, dim=1, keepdim=True).t()  # [B*T', nb_code]
            soft_assign = F.softmax(-dist / self.tau_mi, dim=-1)  # [B*T', nb_code]

            # Reshape to [B, T', nb_code]
            soft_assign_bt = soft_assign.view(B_r, T_prime, -1)

            # p(z) — marginal over all samples and timesteps: [nb_code]
            p_z = soft_assign.mean(dim=0)  # [nb_code]

            # p(z|l) — conditional per label: average soft assignments for samples with label l
            # p(l) — label prior
            mi = torch.tensor(0.0, device=labels.device)
            for l_idx, l in enumerate(unique_labels):
                mask_l = (labels == l)  # [B]
                n_l = mask_l.sum().float()
                p_l = n_l / B  # p(l)

                # Select soft assignments for this label, average over samples and time
                # soft_assign_bt[mask_l] → [n_l, T', nb_code]
                p_z_given_l = soft_assign_bt[mask_l].mean(dim=(0, 1))  # [nb_code]

                # KL(p(z|l) || p(z)) = sum_k p(z|l) * log(p(z|l) / p(z))
                kl = F.kl_div(
                    (p_z + 1e-8).log(),
                    p_z_given_l + 1e-8,
                    reduction='sum',
                    log_target=False
                )
                mi = mi + p_l * kl

            mi_total = mi_total + mi

        # Average over content quantizers
        mi_total = mi_total / num_content
        return mi_total

    def forward(self, features: Tensor, names: Optional[List[str]] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through VQ-Style RVQ-VAE.

        Args:
            features: Input motion features [B, T, D]
            names: Optional list of sample names for cluster label lookup

        Returns:
            x_out: Reconstructed motion [B, T, D]
            loss_commit: Commitment loss
            perplexity: Codebook usage metric
            loss_con: Contrastive loss (weighted by lambda_con)
            loss_mi: MI loss (weighted by lambda_mi)
        """
        # Preprocess: [B, T, D] -> [B, D, T]
        x_in = self.preprocess(features)

        # Encode: [B, D, T] -> [B, code_dim, T']
        x_encoder = self.encoder(x_in)

        # Residual quantization with per-quantizer outputs
        x_quantized, all_indices, loss_commit, perplexity, all_z_q, all_residuals = \
            self.quantizer(x_encoder, return_per_quantizer=True)

        # Decode: [B, code_dim, T'] -> [B, D, T]
        x_decoder = self.decoder(x_quantized)

        # Postprocess: [B, D, T] -> [B, T, D]
        x_out = self.postprocess(x_decoder)

        # Compute VQ-Style losses
        device = features.device
        labels = self._get_batch_labels(names, device)

        if labels is not None and self.training:
            loss_con = self.lambda_con * self._contrastive_loss(all_z_q, labels)
            loss_mi = self.lambda_mi * self._mi_loss(all_residuals, labels)
        else:
            loss_con = torch.tensor(0.0, device=device)
            loss_mi = torch.tensor(0.0, device=device)

        return x_out, loss_commit, perplexity, loss_con, loss_mi
