"""
Precompute global z-score statistics (one mean/std pair) on the TRAIN split (v2 plan).

Train split (repeat split 6/2/2):
    dir00 actions (10 repetitions): rep 1-6
    directional actions (5 reps)  : rep 1-3

Statistics are computed on the raw samples (all time steps, before any time pooling),
over all receivers, all selected antennas and all subcarriers:
    esp        : ESP32, 3 rx x 1 antenna
    asus       : ASUS, 3 rx x 4 antennas
    asus_ant{k}: ASUS, 3 rx x antenna k only (k = 0..3)

Usage:
    python compute_norm_stats.py --data_root data/sshar_amp_interp --rooms room_02
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

PATTERN = re.compile(r"act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)")
RX = ["rx_00", "rx_01", "rx_02"]
LABELS = set(range(1, 9))


def is_train(direction, rep):
    return rep <= 6 if direction == 0 else rep <= 3


def accumulate(root, room, device):
    """Return per-antenna sums (float64): sum, sum of squares, count; and file counts."""
    base = root / room / device / RX[0]
    n_ant = None
    s = sq = cnt = None
    n_samples = 0
    for subject_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in sorted(subject_dir.glob("amp_*.npy")):
            m = PATTERN.search(f.name)
            if m is None:
                continue
            act, _, direction, rep = map(int, m.groups())
            if act not in LABELS or not is_train(direction, rep):
                continue
            for rx in RX:
                x = np.load(root / room / device / rx / subject_dir.name / f.name).astype(np.float64)
                if x.ndim != 3 or x.shape[1] != 56:
                    raise ValueError(f"unexpected shape {x.shape} in {rx}/{subject_dir.name}/{f.name}")
                if n_ant is None:
                    n_ant = x.shape[0]
                    s, sq, cnt = np.zeros(n_ant), np.zeros(n_ant), np.zeros(n_ant)
                s += x.sum(axis=(1, 2))
                sq += np.square(x).sum(axis=(1, 2))
                cnt += x.shape[1] * x.shape[2]
            n_samples += 1
    return s, sq, cnt, n_samples


def mean_std(s, sq, cnt):
    mean = s / cnt
    std = np.sqrt(sq / cnt - mean ** 2)
    return float(mean), float(std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/sshar_amp_interp")
    ap.add_argument("--rooms", nargs="+", default=["room_02"])
    ap.add_argument("--out", default="norm_stats/norm_stats_v2.json")
    args = ap.parse_args()
    root = Path(args.data_root)

    t0 = time.time()
    tot = {}
    for device in ("esp", "asus"):
        S = SQ = C = None
        n = 0
        for room in args.rooms:
            s, sq, cnt, ns = accumulate(root, room, device)
            S, SQ, C = (s, sq, cnt) if S is None else (S + s, SQ + sq, C + cnt)
            n += ns
        tot[device] = (S, SQ, C, n)
        print(f"{device}: {n} train samples, antennas={len(S)}  ({time.time() - t0:.0f}s)")

    stats = {}
    s, sq, c, n = tot["esp"]
    stats["esp"] = dict(zip(("mean", "std"), mean_std(s.sum(), sq.sum(), c.sum())), n_samples=n)
    s, sq, c, n = tot["asus"]
    stats["asus"] = dict(zip(("mean", "std"), mean_std(s.sum(), sq.sum(), c.sum())), n_samples=n)
    for k in range(len(s)):
        stats[f"asus_ant{k}"] = dict(zip(("mean", "std"), mean_std(s[k], sq[k], c[k])), n_samples=n)

    out = {
        "description": "Global z-score stats on TRAIN split (dir00 rep1-6, directional rep1-3), "
                       "raw samples before time pooling, 56 subcarriers, labels 1-8",
        "data_root": str(root),
        "rooms": args.rooms,
        "stats": stats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for k, v in stats.items():
        print(f"  {k:10s} mean={v['mean']:.4f}  std={v['std']:.4f}  n={v['n_samples']}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
