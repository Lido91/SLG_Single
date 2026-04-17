"""
Pure DataLoader throughput benchmark — no model, no GPU.

Measures how fast the train dataloader can produce batches.
If sustained throughput is much lower than the first ~32 batches,
the bottleneck is disk/NFS I/O (prefetch buffer drained).

Usage:
    python benchmark_dataloader.py --cfg configs/deto_h2s_rvq_3_youtube_all.yaml
"""
import argparse
import sys
import time
from collections import deque
from statistics import mean, median

from omegaconf import OmegaConf

from mGPT.config import parse_args
from mGPT.data.build_data import build_data


def _stage(label):
    print(f"[stage] {label}", flush=True)


def main():
    _stage("parse_args")
    cfg = parse_args(phase="train")

    print(f"NUM_WORKERS: {cfg.TRAIN.NUM_WORKERS}, BATCH_SIZE: {cfg.TRAIN.BATCH_SIZE}", flush=True)
    print(f"DATASET: {cfg.DATASET.H2S.DATASET_NAME}, ROOT: {cfg.DATASET.H2S.get('YOUTUBE3D_ROOT') or cfg.DATASET.H2S.ROOT}", flush=True)

    _stage("build_data")
    dm = build_data(cfg)

    _stage("setup('fit')  ← loads train + val datasets (CSV, mean/std)")
    dm.setup("fit")

    _stage("train_dataloader()  ← builds DataLoader object (fast)")
    dl = dm.train_dataloader()

    try:
        print(f"train_dataset size: {len(dm.train_dataset)}", flush=True)
        print(f"train_dataloader steps/epoch: {len(dl)}", flush=True)
    except TypeError:
        pass

    # --- Pre-check: directly load sample 0 in main process (no workers) ---
    _stage("probe: dataset[0] in main process (no workers, no collate)")
    t0 = time.perf_counter()
    sample = dm.train_dataset[0]
    dt = time.perf_counter() - t0
    if sample is None:
        print(f"  dataset[0] -> None  in {dt*1000:.0f}ms (sample was rejected)", flush=True)
    else:
        # sample is the 13-tuple from __getitem__
        try:
            text = sample[0]
            poses = sample[1]
            m_length = sample[2]
            name = sample[3]
            print(f"  dataset[0] OK in {dt*1000:.0f}ms  name={name}  len={m_length}  poses.shape={tuple(poses.shape)}", flush=True)
        except Exception as e:
            print(f"  dataset[0] OK in {dt*1000:.0f}ms but tuple inspect failed: {e}", flush=True)

    _stage("probe: dataset[1..5] timing")
    for i in range(1, 6):
        t0 = time.perf_counter()
        _ = dm.train_dataset[i]
        dt = time.perf_counter() - t0
        print(f"  dataset[{i}] in {dt*1000:.0f}ms", flush=True)

    n_total = 200
    window = 20

    _stage("iter(dl)  ← spawns worker processes")
    it = iter(dl)

    _stage("next(it) first call  ← waits for workers to produce first batch (NFS reads)")
    times = []
    recent = deque(maxlen=window)

    t_all_start = time.perf_counter()
    for i in range(n_total):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            print(f"Epoch ended at step {i}")
            break
        dt = time.perf_counter() - t0
        times.append(dt)
        recent.append(dt)

        # Report at milestones + every step in a compact form
        if i < 32 or (i + 1) % 10 == 0:
            rolling_rate = len(recent) / sum(recent) if recent else 0
            cum_rate = (i + 1) / (time.perf_counter() - t_all_start)
            print(f"[{i+1:4d}/{n_total}] dt={dt*1000:7.1f}ms  "
                  f"rolling({len(recent)})={rolling_rate:5.2f} it/s  "
                  f"cum={cum_rate:5.2f} it/s")

    total_elapsed = time.perf_counter() - t_all_start

    # Summary: compare first-32 vs last-32 to expose prefetch-vs-steady-state gap
    first_n = min(32, len(times))
    first = times[:first_n]
    last = times[-first_n:] if len(times) >= 2 * first_n else times[-max(1, len(times)//2):]

    print("\n" + "=" * 60)
    print(f"Total: {len(times)} batches in {total_elapsed:.1f}s "
          f"({len(times)/total_elapsed:.2f} it/s avg)")
    print(f"First {len(first)} batches : mean={mean(first)*1000:7.1f}ms  median={median(first)*1000:7.1f}ms  "
          f"throughput={len(first)/sum(first):5.2f} it/s")
    print(f"Last  {len(last)} batches : mean={mean(last)*1000:7.1f}ms  median={median(last)*1000:7.1f}ms  "
          f"throughput={len(last)/sum(last):5.2f} it/s")
    print(f"Slowdown factor: {(mean(last)/mean(first)):.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
