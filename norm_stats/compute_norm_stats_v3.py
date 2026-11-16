DATASET_ROOTS = [
    r"D:\workspace\HUST\Personal\WiFi-Sensing\Code\CodeForPaper\benchmark\data\sshar_amp_interp\room_01",
    r"D:\workspace\HUST\Personal\WiFi-Sensing\Code\CodeForPaper\benchmark\data\sshar_amp_interp\room_02",
]

"""
Tính sẵn mean/std (z-score toàn cục) trên TẬP TRAIN của từng fold, kế hoạch v3
(doc/experiment_plan_v3.md): 5-fold theo lần lặp.

Nhóm lần lặp:
    hành động không hướng (dir00, 10 lần lặp): {1,2} {3,4} {5,6} {7,8} {9,10}
    hành động có hướng (dir01/02, 5 lần lặp) : {1} {2} {3} {4} {5}
Fold k: test = nhóm k, val = nhóm k-1 (quay vòng), train = 3 nhóm còn lại.
Fold 5 = cách chia v2 -> mean/std fold 5 phải khớp V2_NORM_STATS (datasets.py).

Đọc mỗi file một lần, cộng dồn theo (nhóm, ăng-ten), rồi ghép lại cho từng fold.
Tính trên dữ liệu gốc (đủ 1000 điểm thời gian), gộp mọi rx, subcarrier, 8 lớp:
    esp  : ESP32, 3 rx x 1 ăng-ten
    asus : ASUS, 3 rx x 4 ăng-ten

Kết quả: norm_stats_v3_<tên phòng>.json cho mỗi phòng trong DATASET_ROOTS (cùng thư mục file này).
Cấu trúc dữ liệu: <phòng>/<device>/<rx>/<subject>/amp_actXX_posXX_dirXX_repXX.npy

Chạy:
    python norm_stats/compute_norm_stats_v3.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PATTERN = re.compile(r"act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)")
RX = ["rx_00", "rx_01", "rx_02"]
LABELS = set(range(1, 9))
FOLDS = (1, 2, 3, 4, 5)
OUT_DIR = Path(__file__).resolve().parent


def repeat_group(direction, rep):
    return (rep + 1) // 2 if direction == 0 else rep


def train_groups(fold):
    val = (fold - 2) % 5 + 1
    return [g for g in FOLDS if g not in (fold, val)]


def accumulate(room_dir, device):
    """Cộng dồn (float64) theo nhóm lần lặp: {nhóm: [tổng, tổng bình phương, số phần tử, số mẫu]}."""
    base = room_dir / device / RX[0]
    acc = {g: [0.0, 0.0, 0, 0] for g in FOLDS}
    for subject_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in sorted(subject_dir.glob("amp_*.npy")):
            m = PATTERN.search(f.name)
            if m is None:
                continue
            act, _, direction, rep = map(int, m.groups())
            if act not in LABELS:
                continue
            g = repeat_group(direction, rep)
            if g not in acc:
                raise ValueError(f"lần lặp ngoài 1..5 nhóm: {device}/{subject_dir.name}/{f.name}")
            for rx in RX:
                x = np.load(room_dir / device / rx / subject_dir.name / f.name).astype(np.float64)
                if x.ndim != 3 or x.shape[1] != 56:
                    raise ValueError(f"shape {x.shape} không đúng: {device}/{rx}/{subject_dir.name}/{f.name}")
                acc[g][0] += x.sum()
                acc[g][1] += np.square(x).sum()
                acc[g][2] += x.size
            acc[g][3] += 1
    return acc


def fold_stats(acc, fold):
    gs = train_groups(fold)
    s = sum(acc[g][0] for g in gs)
    sq = sum(acc[g][1] for g in gs)
    cnt = sum(acc[g][2] for g in gs)
    n = sum(acc[g][3] for g in gs)
    mean = s / cnt
    return {"mean": float(mean), "std": float(np.sqrt(sq / cnt - mean ** 2)), "n_samples": n}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    for root in DATASET_ROOTS:
        room_dir = Path(root)
        room = room_dir.name
        stats = {f"fold{k}": {} for k in FOLDS}
        for device in ("esp", "asus"):
            acc = accumulate(room_dir, device)
            print(f"{room} {device}: mẫu theo nhóm {[acc[g][3] for g in FOLDS]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            for k in FOLDS:
                stats[f"fold{k}"][device] = fold_stats(acc, k)

        out = OUT_DIR / f"norm_stats_v3_{room}.json"
        out.write_text(json.dumps({
            "room": room,
            "dataset_root": root,
            "split": "v3 5-fold over repetitions; groups dir00 {1,2}..{9,10}, directional {1}..{5}; "
                     "fold k: test=group k, val=group k-1 (cyclic), train=other 3 groups (labels 1-8)",
            "note": "global z-score stats on raw samples (1000 time steps), 56 subcarriers; "
                    "fold5 == v2 split",
            "stats": stats,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        for k in FOLDS:
            line = "  ".join(f"{d}: mean={v['mean']:.4f} std={v['std']:.4f} n={v['n_samples']}"
                             for d, v in stats[f"fold{k}"].items())
            print(f"  {room} fold{k}  {line}")
        print("Đã lưu", out)


if __name__ == "__main__":
    main()
