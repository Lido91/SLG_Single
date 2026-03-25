import torch
import torch.nn.functional as F


def symmetric_infonce(emb_a: torch.Tensor, emb_b: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (CLIP-style) between two sets of embeddings.

    Decorated with @custom_fwd to force fp32 execution even under AMP autocast.
    This prevents autocast from downcasting the matmul to fp16/bf16, which causes
    NaN when logit_scale is large (late in training).

    Args:
        emb_a: [B, D] L2-normalized embeddings from modality A
        emb_b: [B, D] L2-normalized embeddings from modality B
        logit_scale: scalar temperature (already exponentiated)

    Returns:
        loss: scalar symmetric cross-entropy loss
    """
    emb_a = emb_a.float()
    emb_b = emb_b.float()
    logit_scale = logit_scale.float()
    logits = logit_scale * (emb_a @ emb_b.T)  # [B, B], guaranteed fp32
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)
    return (loss_a2b + loss_b2a) / 2.0
