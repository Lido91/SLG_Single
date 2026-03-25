"""
Pre-extract speech encoder features from audio files.

Saves features as .pt files so training doesn't need to run the speech encoder
at every iteration.

Usage:
    # Extract HuBERT-large features for Youtube3D (all splits)
    python extract_speech_features.py \
        --encoder hubert-large \
        --data_root /data/hwu/slg_data/Youtube3D \
        --dataset youtube3d \
        --gpu 0

    # Extract Whisper-medium features for Youtube3D
    python extract_speech_features.py \
        --encoder whisper-medium \
        --data_root /data/hwu/slg_data/Youtube3D \
        --dataset youtube3d \
        --gpu 0

    # Extract for How2Sign
    python extract_speech_features.py \
        --encoder whisper-medium \
        --data_root /data/hwu/slg_data/How2Sign \
        --dataset how2sign \
        --gpu 0

    # Extract specific split only
    python extract_speech_features.py \
        --encoder hubert-large \
        --data_root /data/hwu/slg_data/Youtube3D \
        --dataset youtube3d \
        --splits train \
        --gpu 0

Output structure:
    {data_root}/speech_features/{encoder_type}/{split}/{SENTENCE_NAME}.pt
    Each .pt file contains:
        - "features": (seq_len, dim) float16 tensor
        - "attention_mask": (seq_len,) int8 tensor
"""

import argparse
import os
import sys
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mGPT.archs.speech_encoder import SpeechEncoder
from mGPT.data.audio_utils import load_audio


def get_sample_names(data_root, split, dataset):
    """Load sample names from CSV for a given split."""
    if dataset == "youtube3d":
        csv_path = os.path.join(
            data_root, split, "re_aligned", f"youtube_asl_{split}.csv"
        )
    elif dataset == "how2sign":
        csv_path = os.path.join(
            data_root,
            split,
            "re_aligned",
            f"how2sign_realigned_{split}_preprocessed_fps.csv",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not os.path.exists(csv_path):
        print(f"WARNING: CSV not found: {csv_path}, skipping split '{split}'")
        return []

    csv = pd.read_csv(csv_path)
    csv["DURATION"] = csv["END_REALIGNED"] - csv["START_REALIGNED"]
    csv = csv[csv["DURATION"] < 30].reset_index(drop=True)
    names = csv["SENTENCE_NAME"].tolist()
    print(f"  [{split}] Found {len(names)} samples from {csv_path}")
    return names


def extract_features_for_split(
    encoder, data_root, split, dataset, output_dir, batch_size, device
):
    """Extract features for one split."""
    names = get_sample_names(data_root, split, dataset)
    if not names:
        return 0, 0

    audio_dir = os.path.join(data_root, "speech", f"{split}_wavs")
    if not os.path.isdir(audio_dir):
        print(f"WARNING: Audio directory not found: {audio_dir}, skipping")
        return 0, 0

    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    extracted = 0
    skipped = 0
    failed = 0

    # Process in batches for efficiency
    batch_names = []
    batch_audios = []

    pbar = tqdm(names, desc=f"  [{split}]", unit="sample")
    for name in pbar:
        # Skip if already extracted
        out_path = os.path.join(split_output_dir, f"{name}.pt")
        if os.path.exists(out_path):
            skipped += 1
            pbar.set_postfix(done=extracted, skip=skipped, fail=failed)
            continue

        # Load audio
        audio_path = os.path.join(audio_dir, f"{name}.wav")
        if not os.path.exists(audio_path):
            failed += 1
            pbar.set_postfix(done=extracted, skip=skipped, fail=failed)
            continue

        try:
            audio = load_audio(audio_path, target_sr=16000)
        except Exception as e:
            print(f"\n  WARNING: Failed to load {audio_path}: {e}")
            failed += 1
            continue

        batch_names.append(name)
        batch_audios.append(audio)

        # Process batch
        if len(batch_audios) >= batch_size:
            n = _process_batch(
                encoder, batch_names, batch_audios, split_output_dir, device
            )
            extracted += n
            batch_names = []
            batch_audios = []
            pbar.set_postfix(done=extracted, skip=skipped, fail=failed)

    # Process remaining
    if batch_audios:
        n = _process_batch(
            encoder, batch_names, batch_audios, split_output_dir, device
        )
        extracted += n

    print(
        f"  [{split}] Done: extracted={extracted}, skipped={skipped}, failed={failed}"
    )
    return extracted, skipped


def _process_batch(encoder, names, audios, output_dir, device):
    """Run encoder on a batch of audios and save results."""
    # Pad to same length
    max_len = max(a.shape[0] for a in audios)
    padded = torch.zeros(len(audios), max_len)
    for i, a in enumerate(audios):
        padded[i, : a.shape[0]] = a

    padded = padded.to(device)

    # Forward pass
    with torch.no_grad():
        features, attention_mask = encoder(padded, return_attention_mask=True)
        # features: (B, seq_len, dim)
        # attention_mask: (B, seq_len)

    # Save each sample individually (trim padding based on original audio length)
    saved = 0
    for i, name in enumerate(names):
        # Compute actual feature length for this sample
        # For waveform encoders: ~num_samples / 320
        # For whisper: fixed 1500
        mask_i = attention_mask[i]  # (seq_len,)
        feat_i = features[i]  # (seq_len, dim)

        # Trim to valid length using attention mask
        valid_len = mask_i.sum().item()
        if valid_len > 0:
            feat_i = feat_i[:valid_len]
            mask_i = mask_i[:valid_len]

        out_path = os.path.join(output_dir, f"{name}.pt")
        torch.save(
            {
                "features": feat_i.cpu().half(),  # fp16 to save space
                "attention_mask": mask_i.cpu().to(torch.int8),
            },
            out_path,
        )
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract speech encoder features from audio files"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="hubert-large",
        choices=[
            "hubert-base",
            "hubert-large",
            "hubert-xlarge",
            "wav2vec2-base",
            "wav2vec2-large",
            "wavlm-base",
            "wavlm-large",
            "whisper-base",
            "whisper-medium",
            "whisper-large-v3",
        ],
        help="Speech encoder type (default: hubert-large)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/hwu/slg_data/Youtube3D",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="youtube3d",
        choices=["youtube3d", "how2sign"],
        help="Dataset name (default: youtube3d)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for feature extraction (default: 8)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device id (default: 0)"
    )

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU available, using CPU (this will be slow)")

    # Output directory: {data_root}/speech_features/{encoder_type}/
    output_dir = os.path.join(args.data_root, "speech_features", args.encoder)
    print(f"\n{'='*60}")
    print(f"Speech Feature Extraction")
    print(f"{'='*60}")
    print(f"Encoder:    {args.encoder}")
    print(f"Dataset:    {args.dataset}")
    print(f"Data root:  {args.data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Splits:     {args.splits}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")

    # Load encoder
    print("Loading speech encoder...")
    encoder = SpeechEncoder(encoder_type=args.encoder, freeze=True)
    encoder = encoder.to(device)
    encoder.eval()
    print(f"Encoder loaded: output_dim={encoder.output_dim}D\n")

    # Process each split
    total_extracted = 0
    total_skipped = 0
    for split in args.splits:
        print(f"Processing split: {split}")
        n_ext, n_skip = extract_features_for_split(
            encoder,
            args.data_root,
            split,
            args.dataset,
            output_dir,
            args.batch_size,
            device,
        )
        total_extracted += n_ext
        total_skipped += n_skip
        print()

    print(f"{'='*60}")
    print(f"All done! Extracted: {total_extracted}, Skipped: {total_skipped}")
    print(f"Features saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
