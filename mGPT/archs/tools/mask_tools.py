"""
Utility functions for MoMask-style masked generative modeling.

Includes:
- Masking strategies
- Noise schedules
- Sampling helpers
- Loss computation
"""

import torch
import torch.nn.functional as F
import math
from einops import rearrange


def lengths_to_mask(lengths, max_len):
    """
    Create a mask where valid (non-padding) positions are TRUE.

    Args:
        lengths: (B,) tensor of sequence lengths
        max_len: maximum sequence length

    Returns:
        mask: (B, max_len) boolean tensor, TRUE for valid positions
    """
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def get_pad_mask_idx(seq, pad_idx):
    """Return mask where padding positions are FALSE."""
    return (seq != pad_idx).unsqueeze(1)


def get_subsequent_mask(seq):
    """
    Create causal attention mask for autoregressive modeling.

    Returns:
        mask: (1, seq_len, seq_len) lower triangular boolean tensor
    """
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    """Decorator to temporarily set model to eval mode during generation."""
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def l2norm(t):
    return F.normalize(t, dim=-1)


def get_mask_subset_prob(mask, prob):
    """
    Get a random subset of TRUE positions in mask with given probability.

    Args:
        mask: (B, N) boolean tensor
        prob: probability of selecting each TRUE position

    Returns:
        subset_mask: (B, N) boolean tensor, subset of mask
    """
    subset_mask = torch.bernoulli(mask.float(), p=prob).bool() & mask
    return subset_mask


def get_mask_special_tokens(ids, special_ids):
    """Create mask for special tokens in ids."""
    mask = torch.zeros_like(ids).bool()
    for special_id in special_ids:
        mask |= (ids == special_id)
    return mask


def uniform(shape, device=None):
    """Sample uniform random values in [0, 1)."""
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    """Create a random boolean mask with given probability of TRUE."""
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


# Sampling helpers

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    """Sample Gumbel noise for Gumbel-Softmax sampling."""
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    """
    Gumbel-Softmax sampling.

    Args:
        t: logits tensor
        temperature: sampling temperature (higher = more random)
        dim: dimension to sample along

    Returns:
        samples: argmax of (logits/temp + gumbel_noise)
    """
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9, dim=-1):
    """
    Top-k filtering: set logits outside top-k to -inf.

    Args:
        logits: (B, N, V) tensor of logits
        thres: keep top (1-thres) fraction of values
        dim: dimension to apply top-k

    Returns:
        filtered_logits: logits with non-top-k values set to -inf
    """
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim=dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    return probs


# Noise schedules

def cosine_schedule(t):
    """
    Cosine noise schedule: more masking at t=0, less at t=1.

    Args:
        t: timestep in [0, 1]

    Returns:
        mask_ratio: cosine-scaled mask ratio
    """
    return torch.cos(t * math.pi * 0.5)


def scale_cosine_schedule(t, scale):
    """Scaled cosine schedule with adjustable range."""
    return torch.clip(scale * torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)


def q_schedule(bs, low, high, device):
    """
    Schedule for randomly sampling quantizer layers.

    Samples layer indices with cosine weighting (favors higher layers).

    Args:
        bs: batch size
        low: minimum layer index (inclusive)
        high: maximum layer index (exclusive)
        device: torch device

    Returns:
        layer_indices: (bs,) tensor of layer indices in [low, high)
    """
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low


def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    """
    Calculate loss and accuracy for token prediction.

    Args:
        pred: (B, V, N) predicted logits
        labels: (B, N) ground truth token indices
        ignore_index: index to ignore in loss computation
        smoothing: label smoothing factor
        tk: top-k for accuracy computation

    Returns:
        loss: cross-entropy loss
        pred_id: (B, N) predicted token indices
        acc: top-k accuracy
    """
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)

    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc


def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    """
    Calculate cross entropy loss with optional label smoothing.

    Args:
        pred: (B, V, N) predicted logits
        labels: (B, N) ground truth token indices
        ignore_index: index to ignore in loss
        smoothing: label smoothing factor

    Returns:
        loss: scalar loss value
    """
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss