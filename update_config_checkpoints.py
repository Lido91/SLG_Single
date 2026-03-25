#!/usr/bin/env python3
"""
Script to update config YAML files with the best checkpoint path (min-* checkpoint).

The script automatically finds the experiment directory by reading the NAME field
from the config file, then locates checkpoints with 'min-' prefix (saved when
the monitored metric reaches its minimum/best value).

Usage:
    python update_config_checkpoints.py --config configs/deto_h2s_rvq_3_youtube.yaml

Options:
    --config: Path to the config YAML file to update
    --exp_base: Base directory for experiments (default: experiments/mgpt)
    --update_vae: Update PRETRAINED_VAE field (default: True)
    --update_test: Update TEST.CHECKPOINTS field (default: True)
    --dry_run: Print changes without modifying the file (default: False)
    --list: List available experiment directories
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional, List


def extract_name_from_config(config_path: str) -> Optional[str]:
    """
    Extract the NAME field from a config YAML file.

    Args:
        config_path: Path to the config YAML file

    Returns:
        The NAME value, or None if not found
    """
    with open(config_path, 'r') as f:
        content = f.read()

    # Match NAME: value (with or without quotes)
    match = re.search(r'^NAME:\s*["\']?([^"\'\n#]+)["\']?', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def extract_stage_from_config(config_path: str) -> Optional[str]:
    """
    Extract the TRAIN.STAGE field from a config YAML file.

    Args:
        config_path: Path to the config YAML file

    Returns:
        The STAGE value (e.g., 'vae', 'lm'), or None if not found
    """
    with open(config_path, 'r') as f:
        content = f.read()

    # Match STAGE: value under TRAIN section
    match = re.search(r'^\s*STAGE:\s*["\']?([^"\'\n#]+)["\']?', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def find_experiment_dir(name: str, exp_base: str = "experiments/mgpt") -> Optional[str]:
    """
    Find the experiment directory that exactly matches the config NAME.

    The experiment directory should be: {exp_base}/{NAME}
    e.g., experiments/mgpt/DETO_RVQ_wholebody_3

    Args:
        name: The NAME field from the config
        exp_base: Base directory for experiments

    Returns:
        Path to the experiment directory, or None if not found
    """
    if not os.path.exists(exp_base):
        print(f"Error: Experiments base directory not found: {exp_base}")
        return None

    # The experiment directory should exactly match the NAME
    exp_dir = os.path.join(exp_base, name)

    if not os.path.exists(exp_dir):
        print(f"Error: Experiment directory not found: {exp_dir}")
        return None

    if not os.path.isdir(exp_dir):
        print(f"Error: {exp_dir} is not a directory")
        return None

    # Check if it has checkpoints directory
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        print(f"Error: Checkpoints directory not found: {ckpt_dir}")
        return None

    return exp_dir


def find_best_checkpoint(exp_dir: str) -> Optional[str]:
    """
    Find the best checkpoint file (with 'min-' prefix) in the experiment directory.

    The best checkpoint is the one with 'min-' prefix, which indicates the checkpoint
    saved when the metric was at its minimum (best) value.

    Returns:
        Path to the best checkpoint (min-* checkpoint), or None if not found
    """
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        return None

    # Find all checkpoints with 'min-' prefix (best metric checkpoints)
    min_ckpts = list(Path(checkpoint_dir).glob("min-*.ckpt"))

    if not min_ckpts:
        print(f"Warning: No 'min-' checkpoints found in {checkpoint_dir}")
        # Fallback: look for any checkpoint with epoch in name
        all_ckpts = [f for f in Path(checkpoint_dir).glob("*.ckpt") if f.name != "last.ckpt"]
        if all_ckpts:
            print(f"  Found {len(all_ckpts)} other checkpoint(s), using the latest one")
            # Sort by epoch number and return the highest
            all_ckpts.sort(key=lambda x: extract_epoch(x.name) or 0, reverse=True)
            return str(all_ckpts[0])
        return None

    # If multiple min-* checkpoints exist, pick the one with highest epoch
    # (most recent best checkpoint)
    best_ckpt = None
    best_epoch = -1

    for ckpt_file in min_ckpts:
        epoch = extract_epoch(ckpt_file.name)
        if epoch is not None and epoch > best_epoch:
            best_epoch = epoch
            best_ckpt = str(ckpt_file)
        elif best_ckpt is None:
            best_ckpt = str(ckpt_file)

    return best_ckpt


def extract_epoch(filename: str) -> Optional[int]:
    """Extract epoch number from checkpoint filename."""
    match = re.search(r'epoch[=_]?(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def update_config(
    config_path: str,
    exp_base: str = "experiments/mgpt",
    update_vae: bool = True,
    update_test: bool = True,
    dry_run: bool = False
) -> None:
    """
    Update config file with the best checkpoint path (min-* checkpoint).

    Args:
        config_path: Path to the config YAML file
        exp_base: Base directory for experiments
        update_vae: Whether to update PRETRAINED_VAE field
        update_test: Whether to update TEST.CHECKPOINTS field
        dry_run: If True, print changes without modifying file
    """
    # Extract NAME from config
    name = extract_name_from_config(config_path)
    if name is None:
        print(f"Error: Could not find NAME field in config: {config_path}")
        return

    print(f"Config NAME: {name}")

    # Find experiment directory
    exp_dir = find_experiment_dir(name, exp_base)
    if exp_dir is None:
        return

    print(f"Using experiment: {exp_dir}")

    # Find best checkpoint (min-* prefix)
    best_ckpt = find_best_checkpoint(exp_dir)

    if best_ckpt is None:
        print("Error: No suitable checkpoint found in the experiment directory.")
        return

    print(f"Found best checkpoint (min-*):")
    print(f"  {best_ckpt}")

    # Check TRAIN.STAGE to determine if we should update PRETRAINED_VAE
    stage = extract_stage_from_config(config_path)
    print(f"TRAIN.STAGE: {stage}")

    # Read config file
    with open(config_path, 'r') as f:
        content = f.read()

    original_content = content

    # Update PRETRAINED_VAE only if STAGE is 'vae'
    if update_vae and stage == "vae":
        # Match PRETRAINED_VAE with optional quotes and any value
        pattern = r'(PRETRAINED_VAE:\s*)(["\']?[^"\'\n]*["\']?)?'
        replacement = f'\\1"{best_ckpt}"'
        content = re.sub(pattern, replacement, content)
        print(f"\nUpdating PRETRAINED_VAE to: {best_ckpt}")
    elif update_vae and stage != "vae":
        print(f"\nSkipping PRETRAINED_VAE update (STAGE is '{stage}', not 'vae')")

    # Update TEST.CHECKPOINTS
    if update_test:
        # Match CHECKPOINTS under TEST section
        pattern = r'(CHECKPOINTS:\s*)(["\']?[^"\'\n]*["\']?)?'
        replacement = f'\\1"{best_ckpt}"'
        content = re.sub(pattern, replacement, content)
        print(f"Updating TEST.CHECKPOINTS to: {best_ckpt}")

    if dry_run:
        print("\n[DRY RUN] Changes that would be made:")
        print("-" * 50)
        # Show diff-like output
        orig_lines = original_content.split('\n')
        new_lines = content.split('\n')
        for i, (orig, new) in enumerate(zip(orig_lines, new_lines), 1):
            if orig != new:
                print(f"Line {i}:")
                print(f"  - {orig}")
                print(f"  + {new}")
    else:
        # Write updated config
        with open(config_path, 'w') as f:
            f.write(content)
        print(f"\nConfig file updated: {config_path}")


def list_experiments(base_dir: str = "experiments/mgpt") -> None:
    """List available experiment directories."""
    if not os.path.exists(base_dir):
        print(f"Experiments directory not found: {base_dir}")
        return

    print(f"Available experiments in {base_dir}:")
    for exp in sorted(os.listdir(base_dir)):
        exp_path = os.path.join(base_dir, exp)
        if os.path.isdir(exp_path):
            ckpt_dir = os.path.join(exp_path, "checkpoints")
            if os.path.exists(ckpt_dir):
                ckpts = list(Path(ckpt_dir).glob("*.ckpt"))
                min_ckpts = list(Path(ckpt_dir).glob("min-*.ckpt"))
                print(f"  {exp}/ ({len(ckpts)} checkpoints, {len(min_ckpts)} min-*)")
            else:
                print(f"  {exp}/ (no checkpoints)")


def main():
    parser = argparse.ArgumentParser(
        description="Update config YAML files with the best checkpoint path"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to the config YAML file to update"
    )
    parser.add_argument(
        "--exp_base", "-b",
        type=str,
        default="experiments/mgpt",
        help="Base directory for experiments (default: experiments/mgpt)"
    )
    parser.add_argument(
        "--update_vae",
        action="store_true",
        default=True,
        help="Update PRETRAINED_VAE field (default: True)"
    )
    parser.add_argument(
        "--no_vae",
        action="store_true",
        help="Do not update PRETRAINED_VAE field"
    )
    parser.add_argument(
        "--update_test",
        action="store_true",
        default=True,
        help="Update TEST.CHECKPOINTS field (default: True)"
    )
    parser.add_argument(
        "--no_test",
        action="store_true",
        help="Do not update TEST.CHECKPOINTS field"
    )
    parser.add_argument(
        "--dry_run", "-d",
        action="store_true",
        help="Print changes without modifying the file"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available experiment directories"
    )

    args = parser.parse_args()

    if args.list:
        list_experiments(args.exp_base)
        return

    if not args.config:
        parser.print_help()
        print("\nExample usage:")
        print("  python update_config_checkpoints.py --config configs/deto_h2s_rvq_3_youtube.yaml")
        print("\nTo list available experiments:")
        print("  python update_config_checkpoints.py --list")
        return

    update_vae = args.update_vae and not args.no_vae
    update_test = args.update_test and not args.no_test

    update_config(
        config_path=args.config,
        exp_base=args.exp_base,
        update_vae=update_vae,
        update_test=update_test,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
