import os
import pandas as pd

# ============ UPDATE THESE PATHS ============
youtube3d_root = "/data/hwu/slg_data/Youtube3D"  # Update this
split_list = ["train","val","test"]  # or "val", "test"
# ============================================

overall_problematic = []
overall_missing_person0 = []
overall_dirs_no_person0 = []
overall_total_files_to_delete = 0

for split in split_list:
    print(f"current split: ------------------------------------- {split} -------------------------------------")
    # CSV path (adjust suffix if using filtered/gloss versions)
    csv_path = os.path.join(youtube3d_root, split, 're_aligned', f'youtube_asl_{split}.csv')
    poses_dir = os.path.join(youtube3d_root, split, 'poses')

    # Load CSV
    df = pd.read_csv(csv_path)
    df['DURATION'] = df['END_REALIGNED'] - df['START_REALIGNED']
    df = df[df['DURATION'] < 30].reset_index(drop=True)

    # print(f"Total samples after filtering: {len(df)}")

    problematic_samples = []
    missing_person0_samples = []
    dirs_with_no_person0 = []

    for idx in range(len(df)):
        name = df['SENTENCE_NAME'].iloc[idx]
        fps = df['fps'].iloc[idx] if 'fps' in df.columns else 24

        base_dir = os.path.join(poses_dir, name)

        if not os.path.exists(base_dir):
            # print(f"[MISSING] idx={idx}, name={name}: directory not found")
            continue

        frame_files = sorted([f for f in os.listdir(base_dir) if f.endswith('.pkl')])
        num_frames = len(frame_files)

        # Check if all pkl files have person_0 in filename
        files_with_person0 = [f for f in frame_files if 'person_0' in f]
        files_missing_person0 = [f for f in frame_files if 'person_0' not in f]

        # Check if directory has no person_0 files at all
        if len(files_with_person0) == 0 and num_frames > 0:
            dirs_with_no_person0.append({
                'idx': idx,
                'name': name,
                'total_files': num_frames
            })
            # print(f"[NO PERSON_0 DIR] idx={idx}, name={name}: {num_frames} files, none have person_0")

        if files_missing_person0:
            missing_person0_samples.append({
                'idx': idx,
                'name': name,
                'missing_files': files_missing_person0,
                'total_files': num_frames
            })
            # print(f"[NO PERSON_0] idx={idx}, name={name}: {len(files_missing_person0)}/{num_frames} files missing person_0")

        if fps > 24:
            target_count = int(24 * num_frames / fps)
            if target_count == 0:
                problematic_samples.append({
                    'idx': idx,
                    'name': name,
                    'num_frames': num_frames,
                    'fps': fps,
                    'target_count': target_count
                })
                # print(f"[ZERO DIV] idx={idx}, name={name}, frames={num_frames}, fps={fps}")

    print(f"\n{'='*50}")
    print(f"Total problematic samples (division by zero): {len(problematic_samples)}")
    for s in problematic_samples:
        print(f"  - {s['name']}: {s['num_frames']} frames @ {s['fps']} fps")

    print(f"\n{'='*50}")
    print(f"Total samples missing person_0: {len(missing_person0_samples)}")
    total_files_to_delete = 0
    for s in missing_person0_samples:
        total_files_to_delete += len(s['missing_files'])
        print(f"  - {s['name']}: {len(s['missing_files'])}/{s['total_files']} files missing person_0")
        for f in s['missing_files']:
            print(f"      {f}")
    print(f"\nTotal files that would be deleted: {total_files_to_delete}")

    print(f"\n{'='*50}")
    print(f"Total directories with NO person_0 files: {len(dirs_with_no_person0)}")
    for d in dirs_with_no_person0:
        print(f"  - {d['name']}: {d['total_files']} files")

    # Accumulate for overall summary
    overall_problematic.extend([(split, s) for s in problematic_samples])
    overall_missing_person0.extend([(split, s) for s in missing_person0_samples])
    overall_dirs_no_person0.extend([(split, d) for d in dirs_with_no_person0])
    overall_total_files_to_delete += total_files_to_delete

# Overall summary
print(f"\n{'='*70}")
print(f"{'='*70}")
print(f"OVERALL SUMMARY ACROSS ALL SPLITS")
print(f"{'='*70}")
print(f"Total problematic samples (division by zero): {len(overall_problematic)}")
print(f"Total samples with missing person_0 files: {len(overall_missing_person0)}")
print(f"Total directories with NO person_0 files: {len(overall_dirs_no_person0)}")
print(f"Total files that would be deleted: {overall_total_files_to_delete}")
