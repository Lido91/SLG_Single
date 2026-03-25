from pathlib import Path
import pandas as pd

# ========= CONFIG =========
root = Path("/data/hwu/slg_data/Youtube3D")
splits = ["train", "val", "test"]
# ==========================

for split in splits:
    csv_path = root / split / "re_aligned" / f"youtube_asl_{split}.csv"
    poses_dir = root / split / "poses"

    print(f"\nProcessing {split}")

    df = pd.read_csv(csv_path)

    if "SENTENCE_NAME" not in df.columns:
        raise ValueError(f"SENTENCE_NAME missing in {csv_path}")

    # keep only rows whose pose dir exists
    mask = df["SENTENCE_NAME"].astype(str).apply(
        lambda n: (poses_dir / n).is_dir()
    )

    kept = df[mask].reset_index(drop=True)
    removed = len(df) - len(kept)

    # 🔥 overwrite original CSV
    kept.to_csv(csv_path, index=False)

    print(f"Original rows : {len(df)}")
    print(f"Kept rows     : {len(kept)}")
    print(f"Removed rows  : {removed}")
    print(f"Overwritten  : {csv_path}")
