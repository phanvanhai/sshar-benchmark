DATASET_ROOT = r"D:\workspace\HUST\Personal\WiFi-Sensing\Code\CodeForPaper\benchmark\data\XRF55_8act_dataset"

"""
Tính sẵn mean/std (z-score toàn cục) của XRF55 trên TẬP TRAIN của từng fold, kế hoạch v3
(doc/experiment_plan_v3.md): 5-fold theo lần lặp.

XRF55 (bản 8 hành động): file <user>_<action>_<rep>.npy, 30 người x 8 hành động x 20 lần lặp,
mỗi file (270, 1000) = 3 rx x 3 ăng-ten x 30 subcarrier, 1000 điểm thời gian.

Nhóm lần lặp: {1-4} {5-8} {9-12} {13-16} {17-20}.
Fold k: test = nhóm k, val = nhóm k-1 (quay vòng), train = 3 nhóm còn lại.
Fold 5 = cách chia v2 (train 1-12) -> mean/std fold 5 phải khớp V2_NORM_STATS["xrf55"].
Kết quả: norm_stats_xrf55_v3.json (cùng thư mục với file này).

Chạy:
    python norm_stats/compute_norm_stats_xrf55_v3.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

LABELS = set(range(1, 9))
FOLDS = (1, 2, 3, 4, 5)
OUT_FILE = Path(__file__).resolve().parent / "norm_stats_xrf55_v3.json"


def repeat_group(rep):
    return (rep + 3) // 4


def train_groups(fold):
    val = (fold - 2) % 5 + 1
    return [g for g in FOLDS if g not in (fold, val)]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    acc = {g: [0.0, 0.0, 0, 0] for g in FOLDS}   # tổng, tổng bình phương, số phần tử, số mẫu
    for f in sorted(Path(DATASET_ROOT).glob("*.npy")):
        parts = f.stem.split("_")
        if len(parts) != 3:
            continue
        _, act, rep = map(int, parts)
        if act not in LABELS:
            continue
        g = repeat_group(rep)
        if g not in acc:
            raise ValueError(f"lần lặp ngoài 1..20: {f.name}")
        x = np.load(f).astype(np.float64)
        if x.shape != (270, 1000):
            raise ValueError(f"shape {x.shape} không đúng: {f.name}")
        acc[g][0] += x.sum()
        acc[g][1] += np.square(x).sum()
        acc[g][2] += x.size
        acc[g][3] += 1
    print(f"mẫu theo nhóm {[acc[g][3] for g in FOLDS]} ({time.time() - t0:.0f}s)")

    stats = {}
    for k in FOLDS:
        gs = train_groups(k)
        s = sum(acc[g][0] for g in gs)
        sq = sum(acc[g][1] for g in gs)
        cnt = sum(acc[g][2] for g in gs)
        mean = s / cnt
        stats[f"fold{k}"] = {"xrf55": {"mean": float(mean), "std": float(np.sqrt(sq / cnt - mean ** 2)),
                                       "n_samples": sum(acc[g][3] for g in gs)}}
        v = stats[f"fold{k}"]["xrf55"]
        print(f"  fold{k}  mean={v['mean']:.4f}  std={v['std']:.4f}  n={v['n_samples']}")

    OUT_FILE.write_text(json.dumps({
        "dataset_root": DATASET_ROOT,
        "split": "v3 5-fold over repetitions; groups {1-4}..{17-20}; "
                 "fold k: test=group k, val=group k-1 (cyclic), train=other 3 groups (labels 1-8, all users)",
        "note": "global z-score stats on raw samples (270 x 1000); fold5 == v2 split",
        "stats": stats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Đã lưu", OUT_FILE)


if __name__ == "__main__":
    main()
