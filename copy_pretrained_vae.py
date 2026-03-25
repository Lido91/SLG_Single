#!/usr/bin/env python3
"""
Script to copy PRETRAINED_VAE path from one config YAML to another.

This is useful when you have trained a VAE (stage: vae) and want to use
that checkpoint in a language model training config (stage: lm).

Usage:
    python copy_pretrained_vae.py --from configs/deto_h2s_rvq_3_youtube.yaml --to configs/soke_h2s_stage2.yaml

Options:
    --from: Source config YAML file (with PRETRAINED_VAE path)
    --to: Destination config YAML file to update
    --dry_run: Print changes without modifying the file (default: False)
"""

import argparse
import os
import re
from typing import Optional


def extract_pretrained_vae(config_path: str) -> Optional[str]:
    """
    Extract the PRETRAINED_VAE field from a config YAML file.

    Args:
        config_path: Path to the config YAML file

    Returns:
        The PRETRAINED_VAE path, or None if not found or empty
    """
    with open(config_path, 'r') as f:
        content = f.read()

    # Match PRETRAINED_VAE: value (with or without quotes)
    match = re.search(r'PRETRAINED_VAE:\s*["\']?([^"\'\n]+)["\']?', content)
    if match:
        value = match.group(1).strip()
        if value:  # Return only if not empty
            return value
    return None


def update_pretrained_vae(config_path: str, vae_path: str, dry_run: bool = False) -> bool:
    """
    Update the PRETRAINED_VAE field in a config YAML file.

    Args:
        config_path: Path to the config YAML file to update
        vae_path: The PRETRAINED_VAE path to set
        dry_run: If True, print changes without modifying file

    Returns:
        True if successful, False otherwise
    """
    with open(config_path, 'r') as f:
        content = f.read()

    original_content = content

    # Match PRETRAINED_VAE with optional quotes and any value
    pattern = r'(PRETRAINED_VAE:\s*)(["\']?[^"\'\n]*["\']?)?'
    replacement = f'\\1"{vae_path}"'
    content = re.sub(pattern, replacement, content)

    if content == original_content:
        print("Warning: No changes made. PRETRAINED_VAE field may not exist in destination.")
        return False

    if dry_run:
        print("\n[DRY RUN] Changes that would be made:")
        print("-" * 50)
        orig_lines = original_content.split('\n')
        new_lines = content.split('\n')
        for i, (orig, new) in enumerate(zip(orig_lines, new_lines), 1):
            if orig != new:
                print(f"Line {i}:")
                print(f"  - {orig}")
                print(f"  + {new}")
    else:
        with open(config_path, 'w') as f:
            f.write(content)
        print(f"Config file updated: {config_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Copy PRETRAINED_VAE path from one config to another"
    )
    parser.add_argument(
        "--from", "-f",
        dest="source",
        type=str,
        required=True,
        help="Source config YAML file (with PRETRAINED_VAE path)"
    )
    parser.add_argument(
        "--to", "-t",
        dest="destination",
        type=str,
        required=True,
        help="Destination config YAML file to update"
    )
    parser.add_argument(
        "--dry_run", "-d",
        action="store_true",
        help="Print changes without modifying the file"
    )

    args = parser.parse_args()

    # Check source file exists
    if not os.path.exists(args.source):
        print(f"Error: Source config not found: {args.source}")
        return

    # Check destination file exists
    if not os.path.exists(args.destination):
        print(f"Error: Destination config not found: {args.destination}")
        return

    # Extract PRETRAINED_VAE from source
    vae_path = extract_pretrained_vae(args.source)
    if vae_path is None:
        print(f"Error: No PRETRAINED_VAE path found in source config: {args.source}")
        return

    print(f"Source config: {args.source}")
    print(f"Destination config: {args.destination}")
    print(f"PRETRAINED_VAE path: {vae_path}")

    # Update destination config
    update_pretrained_vae(args.destination, vae_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
