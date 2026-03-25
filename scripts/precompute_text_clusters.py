"""
Precompute text cluster labels for VQ-Style training.

Encodes all text/gloss from How2Sign and/or YouTube3D CSVs with CLIP,
then runs k-means clustering to assign each sample a cluster ID.
Saves {sample_name: cluster_id} as JSON.

Usage (How2Sign only):
    python scripts/precompute_text_clusters.py \
        --data_root /data/hwu/slg_data/How2Sign \
        --num_clusters 64 \
        --output cluster_labels_h2s.json

Usage (YouTube3D only):
    python scripts/precompute_text_clusters.py \
        --data_root /data/hwu/slg_data/Youtube3D \
        --dataset youtube3d \
        --num_clusters 64 \
        --output cluster_labels_ytb.json

Usage (both combined):
    python scripts/precompute_text_clusters.py \
        --data_root /data/hwu/slg_data/How2Sign \
        --youtube3d_root /data/hwu/slg_data/Youtube3D \
        --num_clusters 128 \
        --output cluster_labels_h2s_ytb.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans


def _load_csv_texts(csv_path: str, samples: dict):
    """Load (name, text) pairs from a single CSV into samples dict."""
    if not os.path.exists(csv_path):
        print(f"  Skipping {csv_path} (not found)")
        return

    df = pd.read_csv(csv_path)
    # Determine text column
    text_col = None
    for col in ['SENTENCE', 'text', 'gloss']:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        print(f"  WARNING: No text column found in {csv_path}, columns: {list(df.columns)}")
        return

    # Determine name column
    name_col = None
    for col in ['SENTENCE_NAME', 'name', 'id']:
        if col in df.columns:
            name_col = col
            break
    if name_col is None:
        print(f"  WARNING: No name column found in {csv_path}")
        return

    count = 0
    for _, row in df.iterrows():
        name = str(row[name_col])
        text = str(row[text_col])
        if text and text != 'nan':
            samples[name] = text
            count += 1

    print(f"  Loaded {count} texts from {csv_path}")


def load_texts_h2s(data_root: str, samples: dict):
    """Load texts from How2Sign CSVs."""
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(
            data_root, split, 're_aligned',
            f'how2sign_realigned_{split}_preprocessed_fps.csv'
        )
        _load_csv_texts(csv_path, samples)


def load_texts_youtube3d(data_root: str, samples: dict):
    """Load texts from YouTube3D CSVs."""
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(
            data_root, split, 're_aligned',
            f'youtube_asl_{split}.csv'
        )
        _load_csv_texts(csv_path, samples)


def encode_texts_clip(texts: list, batch_size: int = 256):
    """Encode texts with CLIP text encoder."""
    try:
        import clip
    except ImportError:
        raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)  # [B, 512]
        emb = emb.float().cpu().numpy()
        all_embeddings.append(emb)
        if (i // batch_size) % 10 == 0:
            print(f"  Encoded {i + len(batch_texts)}/{len(texts)} texts")

    return np.concatenate(all_embeddings, axis=0)  # [N, 512]


def main():
    parser = argparse.ArgumentParser(description="Precompute text cluster labels for VQ-Style")
    parser.add_argument('--data_root', type=str, default='',
                        help='Path to How2Sign data root')
    parser.add_argument('--youtube3d_root', type=str, default='',
                        help='Path to YouTube3D data root')
    parser.add_argument('--dataset', type=str, default='auto',
                        choices=['auto', 'how2sign', 'youtube3d'],
                        help='Dataset to use. "auto" detects from data_root/youtube3d_root')
    parser.add_argument('--num_clusters', type=int, default=64,
                        help='Number of k-means clusters')
    parser.add_argument('--output', type=str, default='cluster_labels.json',
                        help='Output JSON path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for k-means')
    args = parser.parse_args()

    samples = {}

    # Determine which datasets to load
    if args.dataset == 'youtube3d':
        # YouTube3D only (data_root points to YouTube3D)
        root = args.youtube3d_root or args.data_root
        print(f"Loading YouTube3D texts from {root}...")
        load_texts_youtube3d(root, samples)
    elif args.dataset == 'how2sign':
        # How2Sign only
        print(f"Loading How2Sign texts from {args.data_root}...")
        load_texts_h2s(args.data_root, samples)
    else:
        # Auto: load whatever is provided
        if args.data_root:
            print(f"Loading How2Sign texts from {args.data_root}...")
            load_texts_h2s(args.data_root, samples)
        if args.youtube3d_root:
            print(f"Loading YouTube3D texts from {args.youtube3d_root}...")
            load_texts_youtube3d(args.youtube3d_root, samples)
    print(f"Total unique samples: {len(samples)}")

    if len(samples) == 0:
        print("ERROR: No samples found!")
        return

    names = list(samples.keys())
    texts = [samples[n] for n in names]

    print(f"Encoding {len(texts)} texts with CLIP...")
    embeddings = encode_texts_clip(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # L2 normalize before clustering
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    print(f"Running k-means with {args.num_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=args.num_clusters,
        random_state=args.seed,
        n_init=10,
        max_iter=300,
        verbose=1
    )
    cluster_ids = kmeans.fit_predict(embeddings)

    # Build output mapping
    label_map = {name: int(cid) for name, cid in zip(names, cluster_ids)}

    # Print cluster distribution
    unique, counts = np.unique(cluster_ids, return_counts=True)
    print(f"\nCluster distribution (min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, std={counts.std():.1f}):")
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  Cluster {u}: {c} samples")
    print(f"  ... ({len(unique)} clusters total)")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(label_map, f)
    print(f"\nSaved {len(label_map)} cluster labels to {args.output}")


if __name__ == '__main__':
    main()
