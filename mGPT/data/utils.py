import torch
import rich
import pickle
import numpy as np


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    if isinstance(batch[0], np.ndarray):
        batch = [torch.tensor(b).float() for b in batch]

    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def humanml3d_collate(batch):
    # Filter out None and invalid batches (e.g., corrupted data from worker errors)
    notnone_batches = [b for b in batch if b is not None and isinstance(b, (tuple, list)) and len(b) >= 10]

    if len(notnone_batches) == 0:
        # Return empty batch if all samples failed
        print("[WARNING] humanml3d_collate: All samples in batch were None or invalid, skipping batch")
        return None

    EvalFlag = False if notnone_batches[0][5] is None else True

    # Sort by text length
    if EvalFlag:
        notnone_batches.sort(key=lambda x: x[5], reverse=True)

    # Motion only
    # Keep dtype: use .long() for tokens (LM stage), .float() for features (VAE stage)
    motion_batch = [b[1] for b in notnone_batches]
    # Check if motion data is long (tokens) or float (features) based on first element
    if motion_batch[0].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
        collated_motion = collate_tensors(motion_batch)  # Keep as long for tokens
    else:
        collated_motion = collate_tensors([m.float() for m in motion_batch])  # Convert to float for features

    adapted_batch = {
        "motion": collated_motion,
        "length": [b[2] for b in notnone_batches],
        "src": [b[9] for b in notnone_batches],
        "name": [b[3] for b in notnone_batches]
    }

    # Text and motion
    if notnone_batches[0][0] is not None:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "all_captions": [b[7] for b in notnone_batches],
        })

    # Evaluation related
    if EvalFlag:
        adapted_batch.update({
            "text": [b[0] for b in notnone_batches],
            "word_embs":
            collate_tensors(
                [torch.tensor(b[3]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors(
                [torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
        })

    # Tasks
    if len(notnone_batches[0]) >= 9:
        adapted_batch.update({"tasks": [b[8] for b in notnone_batches]})

    # Precomputed CLIP text features (element [7], reused from all_captions slot)
    clip_feats_list = [b[7] for b in notnone_batches
                       if b[7] is not None and isinstance(b[7], torch.Tensor)]
    if len(clip_feats_list) == len(notnone_batches):
        adapted_batch["clip_text_features"] = torch.stack(clip_feats_list)  # [B, 512]
    else:
        adapted_batch["clip_text_features"] = None

    # Audio / precomputed speech feature support
    # Check for precomputed speech features first (13th element, index 11)
    has_precomputed = (len(notnone_batches[0]) >= 13
                       and any(b[11] is not None for b in notnone_batches))

    if has_precomputed:
        # Precomputed speech features: pad [T_s, D] tensors
        # Find feat_dim from first non-None entry
        feat_dim = next(b[11].shape[-1] for b in notnone_batches if b[11] is not None)
        # Replace None with zero-length placeholder
        speech_feats_batch = [b[11] if b[11] is not None else torch.zeros(1, feat_dim) for b in notnone_batches]
        speech_lengths = [b[12] if b[12] is not None else 0 for b in notnone_batches]
        max_seq = max(f.shape[0] for f in speech_feats_batch)

        padded_feats = torch.zeros(len(speech_feats_batch), max_seq, feat_dim)
        speech_mask = torch.zeros(len(speech_feats_batch), max_seq)
        for i, feats in enumerate(speech_feats_batch):
            t = feats.shape[0]
            padded_feats[i, :t] = feats
            speech_mask[i, :t] = 1.0

        adapted_batch.update({
            "audio": None,
            "audio_lengths": None,
            "speech_feats": padded_feats,      # (B, max_seq, D)
            "speech_mask": speech_mask,         # (B, max_seq)
            "speech_lengths": speech_lengths,
        })
    elif len(notnone_batches[0]) >= 11 and any(b[10] is not None for b in notnone_batches):
        # Raw audio waveforms (some samples may have None audio)
        audio_batch = [b[10] if b[10] is not None else torch.zeros(1) for b in notnone_batches]
        audio_lengths = [len(audio) for audio in audio_batch]
        max_audio_len = max(audio_lengths)

        padded_audio = torch.zeros(len(audio_batch), max_audio_len)
        for i, audio in enumerate(audio_batch):
            padded_audio[i, :len(audio)] = audio

        adapted_batch.update({
            "audio": padded_audio,  # (B, max_audio_samples)
            "audio_lengths": audio_lengths,
            "speech_feats": None,
            "speech_mask": None,
        })
    else:
        adapted_batch.update({
            "audio": None,
            "audio_lengths": None,
            "speech_feats": None,
            "speech_mask": None,
        })

    return adapted_batch


def load_pkl(path, description=None, progressBar=False):
    if progressBar:
        with rich.progress.open(path, 'rb', description=description) as file:
            data = pickle.load(file)
    else:
        with open(path, 'rb') as file:
            data = pickle.load(file)
    return data
