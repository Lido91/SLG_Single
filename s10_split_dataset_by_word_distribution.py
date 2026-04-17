#!/usr/bin/env python3
"""
Split YouTube ASL dataset into train/val/test splits with balanced word distribution.

This script splits the dataset while considering:
1. Word/vocabulary distribution across splits
2. Video-level grouping to prevent data leakage
3. Stratified sampling to ensure balanced representation

Usage:
    python s9_split_dataset_by_word_distribution.py --input_csv youtube_asl_how2sign_format.csv \
                                                   --output_dir ./dataset_splits \
                                                   --train_ratio 0.8 \
                                                   --val_ratio 0.1 \
                                                   --test_ratio 0.1
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import re
import shutil
from tqdm import tqdm


def extract_video_id(sentence_name):
    """
    Extract video ID from SENTENCE_NAME.

    Example: '4eNt91uV02o-007' -> '4eNt91uV02o'

    Args:
        sentence_name (str): SENTENCE_NAME from dataset

    Returns:
        str: Video ID
    """
    # Split by '-' and take all parts except the last (which is the sequence number)
    parts = sentence_name.rsplit('-', 1)
    return parts[0] if len(parts) > 1 else sentence_name


def tokenize_sentence(sentence):
    """
    Tokenize sentence into words (simple whitespace + punctuation splitting).

    Args:
        sentence (str): Sentence text

    Returns:
        list: List of lowercase words
    """
    if not isinstance(sentence, str):
        sentence = str(sentence)
    # Remove punctuation and convert to lowercase
    sentence = re.sub(r'[^\w\s]', ' ', sentence.lower())
    # Split and filter empty strings
    words = [w.strip() for w in sentence.split() if w.strip()]
    return words


def build_word_statistics(df):
    """
    Build word frequency statistics and word-to-sentence mapping.

    Args:
        df (pd.DataFrame): Dataset dataframe

    Returns:
        tuple: (word_freq, sentence_words, video_words)
            - word_freq: Counter of word frequencies
            - sentence_words: dict mapping sentence_name to set of words
            - video_words: dict mapping video_id to set of words
    """
    word_freq = Counter()
    sentence_words = {}
    video_words = defaultdict(set)

    for idx, row in df.iterrows():
        sentence_name = row['SENTENCE_NAME']
        sentence = row['SENTENCE']
        video_id = extract_video_id(sentence_name)

        # Tokenize
        words = tokenize_sentence(sentence)
        word_set = set(words)

        # Update statistics
        word_freq.update(words)
        sentence_words[sentence_name] = word_set
        video_words[video_id].update(word_set)

    return word_freq, sentence_words, video_words


def calculate_word_distribution(df, sentence_words):
    """
    Calculate word distribution for a given dataframe.

    Args:
        df (pd.DataFrame): Dataset dataframe
        sentence_words (dict): Mapping of sentence_name to words

    Returns:
        Counter: Word frequency distribution
    """
    word_dist = Counter()
    for sentence_name in df['SENTENCE_NAME']:
        if sentence_name in sentence_words:
            word_dist.update(sentence_words[sentence_name])
    return word_dist


def split_videos_by_word_distribution(df, sentence_words, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split videos into train/val/test while balancing word distribution.

    Uses an iterative approach to assign videos to splits such that
    word distributions are as balanced as possible.

    Args:
        df (pd.DataFrame): Dataset dataframe
        sentence_words (dict): Mapping of sentence_name to words
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set
        test_ratio (float): Proportion for test set
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    np.random.seed(seed)

    # Group by video
    df['VIDEO_ID'] = df['SENTENCE_NAME'].apply(extract_video_id)
    video_groups = df.groupby('VIDEO_ID')

    # Get video IDs and their sizes
    video_info = []
    for video_id, group in video_groups:
        video_info.append({
            'video_id': video_id,
            'count': len(group),
            'words': set()
        })
        # Collect all words from this video
        for sentence_name in group['SENTENCE_NAME']:
            if sentence_name in sentence_words:
                video_info[-1]['words'].update(sentence_words[sentence_name])

    # Shuffle videos
    np.random.shuffle(video_info)

    # Calculate target sizes
    total_samples = len(df)
    train_target = int(total_samples * train_ratio)
    val_target = int(total_samples * val_ratio)
    test_target = total_samples - train_target - val_target

    # Initialize splits
    train_videos = []
    val_videos = []
    test_videos = []

    train_count = 0
    val_count = 0
    test_count = 0

    # Greedy assignment: assign each video to the split that needs it most
    # and has the least coverage of the video's words
    train_words = Counter()
    val_words = Counter()
    test_words = Counter()

    for video in video_info:
        video_id = video['video_id']
        video_size = video['count']
        video_word_set = video['words']

        # Calculate how much each split needs this video's words
        # (prefer splits with lower coverage of these words)
        train_need = sum(1 for w in video_word_set if train_words[w] == 0) if train_count < train_target else -1e9
        val_need = sum(1 for w in video_word_set if val_words[w] == 0) if val_count < val_target else -1e9
        test_need = sum(1 for w in video_word_set if test_words[w] == 0) if test_count < test_target else -1e9

        # Calculate remaining capacity
        train_space = train_target - train_count
        val_space = val_target - val_count
        test_space = test_target - test_count

        # Decide which split to assign to
        if train_space > 0 and (train_need >= val_need and train_need >= test_need):
            train_videos.append(video_id)
            train_count += video_size
            train_words.update(video_word_set)
        elif val_space > 0 and (val_need >= test_need):
            val_videos.append(video_id)
            val_count += video_size
            val_words.update(video_word_set)
        elif test_space > 0:
            test_videos.append(video_id)
            test_count += video_size
            test_words.update(video_word_set)
        else:
            # If all splits are full, assign to the one with most space
            if train_space >= val_space and train_space >= test_space:
                train_videos.append(video_id)
                train_count += video_size
                train_words.update(video_word_set)
            elif val_space >= test_space:
                val_videos.append(video_id)
                val_count += video_size
                val_words.update(video_word_set)
            else:
                test_videos.append(video_id)
                test_count += video_size
                test_words.update(video_word_set)

    # Split dataframes
    train_df = df[df['VIDEO_ID'].isin(train_videos)].copy()
    val_df = df[df['VIDEO_ID'].isin(val_videos)].copy()
    test_df = df[df['VIDEO_ID'].isin(test_videos)].copy()

    # Remove VIDEO_ID column (was added temporarily)
    train_df = train_df.drop(columns=['VIDEO_ID'])
    val_df = val_df.drop(columns=['VIDEO_ID'])
    test_df = test_df.drop(columns=['VIDEO_ID'])

    return train_df, val_df, test_df


def print_split_statistics(train_df, val_df, test_df, sentence_words):
    """
    Print statistics about the splits.

    Args:
        train_df (pd.DataFrame): Training set
        val_df (pd.DataFrame): Validation set
        test_df (pd.DataFrame): Test set
        sentence_words (dict): Mapping of sentence_name to words
    """
    print("\n" + "=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)

    # Sample counts
    print("\n📊 Sample Distribution:")
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"  Train: {len(train_df):5d} samples ({len(train_df)/total*100:5.2f}%)")
    print(f"  Val:   {len(val_df):5d} samples ({len(val_df)/total*100:5.2f}%)")
    print(f"  Test:  {len(test_df):5d} samples ({len(test_df)/total*100:5.2f}%)")
    print(f"  Total: {total:5d} samples")

    # Video distribution
    print("\n🎥 Video Distribution:")
    train_videos = train_df['SENTENCE_NAME'].apply(extract_video_id).nunique()
    val_videos = val_df['SENTENCE_NAME'].apply(extract_video_id).nunique()
    test_videos = test_df['SENTENCE_NAME'].apply(extract_video_id).nunique()
    total_videos = train_videos + val_videos + test_videos
    print(f"  Train: {train_videos:5d} videos ({train_videos/total_videos*100:5.2f}%)")
    print(f"  Val:   {val_videos:5d} videos ({val_videos/total_videos*100:5.2f}%)")
    print(f"  Test:  {test_videos:5d} videos ({test_videos/total_videos*100:5.2f}%)")
    print(f"  Total: {total_videos:5d} videos")

    # Word distribution
    train_words = calculate_word_distribution(train_df, sentence_words)
    val_words = calculate_word_distribution(val_df, sentence_words)
    test_words = calculate_word_distribution(test_df, sentence_words)

    print("\n📖 Vocabulary Distribution:")
    print(f"  Train: {len(train_words):5d} unique words")
    print(f"  Val:   {len(val_words):5d} unique words")
    print(f"  Test:  {len(test_words):5d} unique words")

    # Word overlap
    all_words = set(train_words.keys()) | set(val_words.keys()) | set(test_words.keys())
    train_only = set(train_words.keys()) - set(val_words.keys()) - set(test_words.keys())
    val_only = set(val_words.keys()) - set(train_words.keys()) - set(test_words.keys())
    test_only = set(test_words.keys()) - set(train_words.keys()) - set(val_words.keys())

    print(f"\n🔄 Vocabulary Overlap:")
    print(f"  Total unique words: {len(all_words)}")
    print(f"  Words only in train: {len(train_only)} ({len(train_only)/len(all_words)*100:.2f}%)")
    print(f"  Words only in val: {len(val_only)} ({len(val_only)/len(all_words)*100:.2f}%)")
    print(f"  Words only in test: {len(test_only)} ({len(test_only)/len(all_words)*100:.2f}%)")
    print(f"  Words in all splits: {len(set(train_words.keys()) & set(val_words.keys()) & set(test_words.keys()))}")

    # Duration statistics
    print("\n⏱️  Duration Statistics:")
    print(f"  Train: {train_df['DURATION'].sum()/60:.2f} minutes (avg: {train_df['DURATION'].mean():.2f}s)")
    print(f"  Val:   {val_df['DURATION'].sum()/60:.2f} minutes (avg: {val_df['DURATION'].mean():.2f}s)")
    print(f"  Test:  {test_df['DURATION'].sum()/60:.2f} minutes (avg: {test_df['DURATION'].mean():.2f}s)")


def copy_data_files(df, source_dir, dest_dir, split_name):
    """
    Copy data files (folders) from source directory to destination based on SENTENCE_NAME.

    The source directory contains folders named by SENTENCE_NAME (e.g., _-4o0zcmo2I-038).
    These folders will be copied to the destination directory.

    Args:
        df (pd.DataFrame): DataFrame containing SENTENCE_NAME column
        source_dir (Path): Source directory containing data folders
        dest_dir (Path): Destination directory for this split
        split_name (str): Name of the split (train/val/test) for logging

    Returns:
        tuple: (copied_count, skipped_count, missing_count)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0
    missing_count = 0
    missing_files = []

    sentence_names = df['SENTENCE_NAME'].unique()

    for sentence_name in tqdm(sentence_names, desc=f"Copying {split_name} files"):
        src_path = source_dir / sentence_name
        dst_path = dest_dir / sentence_name

        if not src_path.exists():
            missing_count += 1
            missing_files.append(sentence_name)
            continue

        if dst_path.exists():
            skipped_count += 1
            continue

        # Copy the folder (or file)
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        copied_count += 1

    if missing_files and len(missing_files) <= 10:
        print(f"  Missing files: {missing_files}")
    elif missing_files:
        print(f"  First 10 missing files: {missing_files[:10]}")

    return copied_count, skipped_count, missing_count


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train/val/test with balanced word distribution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 80/10/10 split
  python split_dataset_by_word_distribution.py --input_csv youtube_asl_how2sign_format.csv

  # Custom split ratios
  python split_dataset_by_word_distribution.py --input_csv youtube_asl_how2sign_format.csv \\
                                                 --train_ratio 0.7 \\
                                                 --val_ratio 0.15 \\
                                                 --test_ratio 0.15
        """
    )

    parser.add_argument('--input_csv', type=str, default='youtube_asl_how2sign_format.csv',
                        help='Path to input CSV file in How2Sign format')
    parser.add_argument('--output_dir', type=str, default='/data/hwu/slg_data/Youtube3D',
                        help='Base output directory for split CSV files')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proportion for training set (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Proportion for validation set (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Proportion for test set (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--source_data_dir', type=str, default=None,
                        help='Source directory containing data folders named by SENTENCE_NAME '
                             '(e.g., /data/hwu/youtube3d/img_0/clip_fps24_img_0_out/smplx_params). '
                             'Used for filtering missing files and optionally copying data.')
    parser.add_argument('--dest_data_subdir', type=str, default='poses',
                        help='Subdirectory name for copied data in each split (default: poses)')
    parser.add_argument('--copy_files', action='store_true',
                        help='Copy data files to destination. If not set, only CSVs are created. '
                             'Requires --source_data_dir to be set.')

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    print("=" * 70)
    print("Dataset Splitting with Word Distribution Balancing")
    print("=" * 70)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: {args.train_ratio:.1%} / {args.val_ratio:.1%} / {args.test_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    if args.source_data_dir:
        print(f"Source data directory: {args.source_data_dir}")
        print(f"Destination subdirectory: {args.dest_data_subdir}")
        print(f"Copy files: {args.copy_files}")
    print("=" * 70)

    # Create base output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples")

    # Filter out entries with missing files if source_data_dir is provided
    if args.source_data_dir:
        print("\n[2/5] Filtering entries with missing files...")
        source_dir = Path(args.source_data_dir)
        if not source_dir.exists():
            print(f"Warning: Source directory does not exist: {source_dir}")
        else:
            original_count = len(df)
            existing_mask = df['SENTENCE_NAME'].apply(lambda x: (source_dir / x).exists())
            df = df[existing_mask].reset_index(drop=True)
            filtered_count = original_count - len(df)
            print(f"Filtered out {filtered_count} entries with missing files")
            print(f"Remaining {len(df)} samples with existing files")
    else:
        print("\n[2/5] Skipping file existence check (no --source_data_dir provided)")

    # Build word statistics
    print("\n[2/4] Building word statistics...")
    word_freq, sentence_words, video_words = build_word_statistics(df)
    print(f"Found {len(word_freq)} unique words")
    print(f"Found {len(video_words)} unique videos")

    # Split dataset
    print("\n[3/4] Splitting dataset with word distribution balancing...")
    train_df, val_df, test_df = split_videos_by_word_distribution(
        df, sentence_words,
        args.train_ratio, args.val_ratio, args.test_ratio,
        args.seed
    )

    # Save splits
    print("\n[4/4] Saving split files...")
    # Create subdirectories for each split
    train_dir = Path(args.output_dir) / 'train' / 're_aligned'
    val_dir = Path(args.output_dir) / 'val' / 're_aligned'
    test_dir = Path(args.output_dir) / 'test' / 're_aligned'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_path = train_dir / 'youtube_asl_train.csv'
    val_path = val_dir / 'youtube_asl_val.csv'
    test_path = test_dir / 'youtube_asl_test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  Saved train split: {train_path}")
    print(f"  Saved val split: {val_path}")
    print(f"  Saved test split: {test_path}")

    # Print statistics
    print_split_statistics(train_df, val_df, test_df, sentence_words)

    # Copy data files if source directory is provided
    if args.source_data_dir:
        print("\n" + "=" * 70)
        print("COPYING DATA FILES")
        print("=" * 70)

        source_dir = Path(args.source_data_dir)
        if not source_dir.exists():
            print(f"Warning: Source directory does not exist: {source_dir}")
        else:
            # Create destination directories for each split
            train_data_dir = Path(args.output_dir) / 'train' / args.dest_data_subdir
            val_data_dir = Path(args.output_dir) / 'val' / args.dest_data_subdir
            test_data_dir = Path(args.output_dir) / 'test' / args.dest_data_subdir

            print(f"\nCopying train data to {train_data_dir}...")
            train_copied, train_skipped, train_missing = copy_data_files(
                train_df, source_dir, train_data_dir, 'train'
            )
            print(f"  Copied: {train_copied}, Skipped (exists): {train_skipped}, Missing: {train_missing}")

            print(f"\nCopying val data to {val_data_dir}...")
            val_copied, val_skipped, val_missing = copy_data_files(
                val_df, source_dir, val_data_dir, 'val'
            )
            print(f"  Copied: {val_copied}, Skipped (exists): {val_skipped}, Missing: {val_missing}")

            print(f"\nCopying test data to {test_data_dir}...")
            test_copied, test_skipped, test_missing = copy_data_files(
                test_df, source_dir, test_data_dir, 'test'
            )
            print(f"  Copied: {test_copied}, Skipped (exists): {test_skipped}, Missing: {test_missing}")

            print(f"\nTotal files copied: {train_copied + val_copied + test_copied}")
            print(f"Total files skipped: {train_skipped + val_skipped + test_skipped}")
            print(f"Total files missing: {train_missing + val_missing + test_missing}")

    print("\n" + "=" * 70)
    print("Dataset splitting complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
