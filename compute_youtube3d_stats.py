"""
Compute mean and std statistics for YouTube3D dataset.
Run this script to generate youtube3d_mean.pt and youtube3d_std.pt

Usage:
    python compute_youtube3d_stats.py
"""

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

# Configuration
DATA_ROOT = "/data/hwu/slg_data/Youtube3D"
SPLIT = "train"  # Compute stats from training set only
OUTPUT_DIR = DATA_ROOT

# SMPL-X parameter keys in the ORDER expected by the model (same as How2Sign)
# Note: YouTube3D has shape (1, N), so we need to squeeze
keys = [
    'smplx_root_pose',      # (1, 3)  -> (3,)
    'smplx_body_pose',      # (1, 63) -> (63,)
    'smplx_lhand_pose',     # (1, 45) -> (45,)
    'smplx_rhand_pose',     # (1, 45) -> (45,)
    'smplx_jaw_pose',       # (1, 3)  -> (3,)
    'smplx_shape',          # (1, 10) -> (10,)
    'smplx_expr'            # (1, 10) -> (10,)
]  # Total: 179 dimensions


def compute_stats():
    poses_dir = os.path.join(DATA_ROOT, SPLIT, 'poses')

    if not os.path.exists(poses_dir):
        print(f"Error: Directory not found: {poses_dir}")
        print("Please check the DATA_ROOT and SPLIT variables.")
        return

    all_poses = []
    sample_dirs = os.listdir(poses_dir)
    print(f"Found {len(sample_dirs)} samples in {poses_dir}")

    for sample_name in tqdm(sample_dirs, desc="Loading samples"):
        sample_dir = os.path.join(poses_dir, sample_name)
        if not os.path.isdir(sample_dir):
            continue

        frame_files = [f for f in os.listdir(sample_dir) if f.endswith('.pkl')]

        for frame_file in frame_files:
            frame_path = os.path.join(sample_dir, frame_file)
            try:
                with open(frame_path, 'rb') as f:
                    poses = pickle.load(f)

                # Concatenate all SMPL-X parameters
                # YouTube3D format: each value is (1, N), squeeze to (N,)
                pose = np.concatenate([poses[key].squeeze(0) for key in keys], axis=0)
                all_poses.append(pose)
            except Exception as e:
                print(f"Warning: Failed to load {frame_path}: {e}")
                continue

    if len(all_poses) == 0:
        print("Error: No poses were loaded!")
        return

    print(f"\nLoaded {len(all_poses)} frames total")

    # Stack all poses and compute statistics
    all_poses = np.stack(all_poses)  # (N, 179)
    print(f"Poses array shape: {all_poses.shape}")

    # Compute mean and std
    mean = all_poses.mean(axis=0).astype(np.float32)
    std = all_poses.std(axis=0).astype(np.float32)

    # Convert to torch tensors
    mean_tensor = torch.from_numpy(mean)
    std_tensor = torch.from_numpy(std)

    # Save
    mean_path = os.path.join(OUTPUT_DIR, 'youtube3d_mean.pt')
    std_path = os.path.join(OUTPUT_DIR, 'youtube3d_std.pt')

    torch.save(mean_tensor, mean_path)
    torch.save(std_tensor, std_path)

    print(f"\nSaved statistics:")
    print(f"  Mean: {mean_path} (shape: {mean_tensor.shape})")
    print(f"  Std:  {std_path} (shape: {std_tensor.shape})")

    # Print some stats for verification
    print(f"\nStatistics summary:")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range:  [{std.min():.4f}, {std.max():.4f}]")
    print(f"  Std zeros (might cause issues): {(std == 0).sum()}")

    # Warn about zero std values
    if (std == 0).any():
        zero_indices = np.where(std == 0)[0]
        print(f"  Warning: Zero std at indices: {zero_indices}")
        print("  These dimensions have no variance and may cause division by zero.")


if __name__ == "__main__":
    compute_stats()
