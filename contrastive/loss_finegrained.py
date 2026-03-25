"""
Fine-grained sequence-level contrastive losses from paper 2507.23188.

Key differences from the original InfoNCE loss:
1. Sequence-level similarity using max over token pairs (not global mean)
2. Learnable token weights via Linear + Softmax
3. Bidirectional KL divergence instead of cross-entropy
4. Reconstruction loss for masked motion tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_level_similarity(
    tokens_x: torch.Tensor,
    tokens_y: torch.Tensor,
    mask_x: torch.Tensor = None,
    mask_y: torch.Tensor = None,
    weight_head_x: nn.Module = None,
    weight_head_y: nn.Module = None,
) -> torch.Tensor:
    """
    Compute fine-grained sequence-level similarity between two sets of token sequences.

    h(e_x, e_y) = 0.5 * sum_i w_x^i * max_j <e_x^i, e_y^j>
                + 0.5 * sum_j w_y^j * max_i <e_y^j, e_x^i>

    Args:
        tokens_x: [B1, L_x, C] L2-normalized token sequences from modality x
        tokens_y: [B2, L_y, C] L2-normalized token sequences from modality y
        mask_x: [B1, L_x] float mask (1=valid, 0=pad), or None
        mask_y: [B2, L_y] float mask (1=valid, 0=pad), or None
        weight_head_x: nn.Linear(C, 1) for computing token weights w_x
        weight_head_y: nn.Linear(C, 1) for computing token weights w_y

    Returns:
        sim_matrix: [B1, B2] pairwise similarity scores
    """
    B1, L_x, C = tokens_x.shape
    B2, L_y, _ = tokens_y.shape

    # Compute token weights via learned linear + masked softmax
    # w_x: [B1, L_x]
    if weight_head_x is not None:
        w_x = weight_head_x(tokens_x).squeeze(-1)  # [B1, L_x]
    else:
        w_x = torch.zeros(B1, L_x, device=tokens_x.device)
    if mask_x is not None:
        w_x = w_x.masked_fill(mask_x == 0, float('-inf'))
    w_x = F.softmax(w_x, dim=-1)  # [B1, L_x]
    w_x = w_x.nan_to_num(0.0)  # guard: all-padded → softmax(all -inf) → NaN → 0

    # w_y: [B2, L_y]
    if weight_head_y is not None:
        w_y = weight_head_y(tokens_y).squeeze(-1)  # [B2, L_y]
    else:
        w_y = torch.zeros(B2, L_y, device=tokens_y.device)
    if mask_y is not None:
        w_y = w_y.masked_fill(mask_y == 0, float('-inf'))
    w_y = F.softmax(w_y, dim=-1)  # [B2, L_y]
    w_y = w_y.nan_to_num(0.0)  # guard: all-padded → softmax(all -inf) → NaN → 0

    # Pairwise token similarity: [B1, B2, L_x, L_y]
    # tokens_x: [B1, L_x, C], tokens_y: [B2, L_y, C]
    # Use einsum for memory efficiency
    # sim[b1, b2, i, j] = <tokens_x[b1,i], tokens_y[b2,j]>
    token_sim = torch.einsum('bic,djc->bdij', tokens_x, tokens_y)  # [B1, B2, L_x, L_y]

    # Mask invalid positions before max
    if mask_y is not None:
        # For x->y direction: mask invalid y positions
        mask_y_expand = mask_y[None, :, None, :]  # [1, B2, 1, L_y]
        token_sim_xy = token_sim.masked_fill(mask_y_expand == 0, float('-inf'))
    else:
        token_sim_xy = token_sim

    if mask_x is not None:
        # For y->x direction: mask invalid x positions
        mask_x_expand = mask_x[:, None, :, None]  # [B1, 1, L_x, 1]
        token_sim_yx = token_sim.masked_fill(mask_x_expand == 0, float('-inf'))
    else:
        token_sim_yx = token_sim

    # x->y: for each x token, find max similarity with any y token
    max_sim_xy = token_sim_xy.max(dim=-1).values  # [B1, B2, L_x]
    # Zero out invalid x positions (use masked_fill to avoid -inf * 0 = NaN)
    if mask_x is not None:
        max_sim_xy = max_sim_xy.masked_fill(mask_x[:, None, :] == 0, 0.0)

    # y->x: for each y token, find max similarity with any x token
    max_sim_yx = token_sim_yx.max(dim=-2).values  # [B1, B2, L_y]
    # Zero out invalid y positions (use masked_fill to avoid -inf * 0 = NaN)
    if mask_y is not None:
        max_sim_yx = max_sim_yx.masked_fill(mask_y[None, :, :] == 0, 0.0)

    # Weighted sum
    # x->y direction: sum_i w_x^i * max_j sim(x_i, y_j)
    sim_x2y = (w_x[:, None, :] * max_sim_xy).sum(dim=-1)  # [B1, B2]

    # y->x direction: sum_j w_y^j * max_i sim(y_j, x_i)
    sim_y2x = (w_y[None, :, :] * max_sim_yx).sum(dim=-1)  # [B1, B2]

    # Average both directions
    sim_matrix = 0.5 * (sim_x2y + sim_y2x)  # [B1, B2]

    return sim_matrix


def bidirectional_kl_loss(
    sim_x2y: torch.Tensor,
    sim_y2x: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Bidirectional KL divergence alignment loss (Eq. 5 in paper).

    L_align^xy = KL(S_pred^x2y, S_target) + KL(S_pred^y2x, S_target^T)

    Args:
        sim_x2y: [B, B] similarity matrix (x->y predictions)
        sim_y2x: [B, B] similarity matrix (y->x predictions), typically sim_x2y.T
        temperature: temperature for softmax

    Returns:
        loss: scalar KL divergence loss
    """
    B = sim_x2y.shape[0]

    # Target: identity matrix (i-th sample matches i-th sample)
    target = torch.eye(B, device=sim_x2y.device)

    # Predicted distributions (softmax over rows)
    # temperature here is 1/τ (logit scaling factor), so we MULTIPLY to amplify logits
    pred_x2y = F.log_softmax(sim_x2y * temperature, dim=-1)  # [B, B]
    pred_y2x = F.log_softmax(sim_y2x * temperature, dim=-1)  # [B, B]

    # KL(target || pred) = sum target * (log target - log pred)
    # Since target is one-hot, this simplifies to -log pred at diagonal
    # But we use the full KL for numerical stability
    loss_x2y = F.kl_div(pred_x2y, target, reduction='batchmean')
    loss_y2x = F.kl_div(pred_y2x, target, reduction='batchmean')

    return loss_x2y + loss_y2x


def compute_alignment_loss(
    tokens_x: torch.Tensor,
    tokens_y: torch.Tensor,
    mask_x: torch.Tensor,
    mask_y: torch.Tensor,
    weight_head_x: nn.Module,
    weight_head_y: nn.Module,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Full alignment loss between two modalities: sequence-level similarity + KL loss.

    Args:
        tokens_x, tokens_y: [B, L, C] L2-normalized token sequences
        mask_x, mask_y: [B, L] masks
        weight_head_x, weight_head_y: token weight heads
        temperature: learnable temperature

    Returns:
        loss: scalar alignment loss
    """
    sim_matrix = sequence_level_similarity(
        tokens_x, tokens_y, mask_x, mask_y, weight_head_x, weight_head_y
    )  # [B, B]

    loss = bidirectional_kl_loss(sim_matrix, sim_matrix.T, temperature)
    return loss
