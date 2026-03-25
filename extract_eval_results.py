"""Extract Per-Batch Average and Global Retrieval tables from contrastive eval logs."""

import re
from pathlib import Path
import pandas as pd

# ── CONFIG: point this to your eval_results folder ──
EVAL_DIR = Path("experiments/contrastive_codex/NewContrastive_TransformerProj_YTB/eval_results")

DIRECTIONS = [
    "Speech -> Text", "Text -> Speech",
    "Speech -> Motion", "Motion -> Speech",
    "Text -> Motion", "Motion -> Text",
]
METRICS = ["R@1", "R@5", "R@10", "MedR"]

DIRECTION_PATTERN = re.compile(
    r"(Speech -> Text|Text -> Speech|Speech -> Motion|Motion -> Speech|"
    r"Text -> Motion|Motion -> Text|Average)"
    r"\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)"
)


def parse_result_table(lines, start_idx):
    """Parse a retrieval result table from the Direction header line onward."""
    results = {}
    i = start_idx
    while i < len(lines) and not lines[i].strip().startswith("Direction"):
        i += 1
    if i >= len(lines):
        return results, i
    i += 1  # skip header
    if i < len(lines) and lines[i].strip().startswith("---"):
        i += 1  # skip separator

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("===") or line.startswith("Encoding"):
            break
        if line.startswith("---"):
            i += 1
            continue
        m = DIRECTION_PATTERN.search(line)
        if m:
            results[m.group(1)] = {
                "R@1": float(m.group(2)),
                "R@5": float(m.group(3)),
                "R@10": float(m.group(4)),
                "MedR": float(m.group(5)),
            }
        i += 1
    return results, i


def extract_from_log(log_path):
    """Extract per-batch average and global retrieval tables from a log file."""
    lines = Path(log_path).read_text().splitlines()

    per_batch = None
    global_ret = None

    for i, line in enumerate(lines):
        if "Per-Batch Average" in line:
            per_batch, _ = parse_result_table(lines, i)
        if "Global Retrieval" in line and ">>>" not in line:
            j = i
            while j < len(lines) and "Evaluating on" not in lines[j]:
                j += 1
            if j < len(lines):
                global_ret, _ = parse_result_table(lines, j)

    return per_batch, global_ret


def results_to_df(all_results, table_type):
    """Convert extracted results into a pandas DataFrame.
    Rows = checkpoints, columns = MultiIndex (Direction, Metric)."""
    rows = []
    index = []
    for ckpt_name, (per_batch, global_ret) in all_results.items():
        data = per_batch if table_type == "per_batch" else global_ret
        if data is None:
            continue
        row = {}
        for d in DIRECTIONS:
            if d in data:
                for m in METRICS:
                    row[(d, m)] = data[d][m]
        if "Average" in data:
            for m in METRICS:
                row[("Average", m)] = data["Average"][m]
        rows.append(row)
        index.append(ckpt_name)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, index=index)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Direction", "Metric"])
    df.index.name = "Checkpoint"
    # Sort columns by direction order
    col_order = DIRECTIONS + (["Average"] if ("Average", "R@1") in df.columns else [])
    df = df.reindex(columns=col_order, level="Direction")
    return df


# ── MAIN ──
log_files = sorted(EVAL_DIR.glob("*.log"))
print(f"Found {len(log_files)} log files in {EVAL_DIR}\n")

all_results = {}
for log_path in log_files:
    ckpt_name = log_path.stem.replace("_test", "")
    all_results[ckpt_name] = extract_from_log(log_path)

# Build DataFrames
df_batch = results_to_df(all_results, "per_batch")
df_global = results_to_df(all_results, "global")

print("=" * 70)
print(" Per-Batch Average (across all batches)")
print("=" * 70)
display(df_batch)

print()
print("=" * 70)
print(" Global Retrieval")
print("=" * 70)
display(df_global)
