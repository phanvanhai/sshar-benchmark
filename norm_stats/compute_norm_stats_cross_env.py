SRC_ROOM_ROOT = r"D:\workspace\HUST\Personal\WiFi-Sensing\Code\CodeForPaper\benchmark\data\sshar_amp_interp\room_01"
SRC_XRF55_ROOT = r"D:\workspace\HUST\Personal\WiFi-Sensing\Code\CodeForPaper\benchmark\data\XRF55_8act_dataset"

"""
Tính sẵn mean/std (z-score toàn cục) cho thí nghiệm cross-environment room 1 -> room 2
(doc/experiment_plan_cross_env.md).

Chỉ dùng phòng NGUỒN (room 1), tập train của từng fold:
    nhóm lần lặp như v3: dir00 {1,2}..{9,10}, có hướng {1}..{5}, XRF55 {1-4}..{17-20}
    fold k: val = nhóm k, train = 4 nhóm còn lại (8/2)
Phòng đích (room 2) không tham gia; mean/std này dùng chung cho train, val và test.

    esp   : room 1, ESP32, 3 rx x 1 ăng-ten
    asus  : room 1, ASUS, 3 rx x 4 ăng-ten
    xrf55 : XRF55 room1 (270 x 1000)

Kết quả: norm_stats_cross_env.json (cùng thư mục với file này).

Chạy:
    python norm_stats/compute_norm_stats_cross_env.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_norm_stats_v3 import accumulate          # cộng dồn theo nhóm lần lặp (room)
from compute_norm_stats_xrf55_v3 import repeat_group as xrf55_group

FOLDS = (1, 2, 3, 4, 5)
LABELS = set(range(1, 9))
OUT_FILE = Path(__file__).resolve().parent / "norm_stats_cross_env.json"


def accumulate_xrf55(root):
    acc = {g: [0.0, 0.0, 0, 0] for g in FOLDS}
    for f in sorted(Path(root).glob("*.npy")):
        parts = f.stem.split("_")
        if len(parts) != 3:
            continue
        _, act, rep = map(int, parts)
        if act not in LABELS:
            continue
        x = np.load(f).astype(np.float64)
        if x.shape != (270, 1000):
            raise ValueError(f"shape {x.shape} không đúng: {f.name}")
        g = xrf55_group(rep)
        acc[g][0] += x.sum()
        acc[g][1] += np.square(x).sum()
        acc[g][2] += x.size
        acc[g][3] += 1
    return acc


def fold_stats(acc, fold):
    gs = [g for g in FOLDS if g != fold]
    s = sum(acc[g][0] for g in gs)
    sq = sum(acc[g][1] for g in gs)
    cnt = sum(acc[g][2] for g in gs)
    mean = s / cnt
    return {"mean": float(mean), "std": float(np.sqrt(sq / cnt - mean ** 2)),
            "n_samples": sum(acc[g][3] for g in gs)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    accs = {}
    for device in ("esp", "asus"):
        accs[device] = accumulate(Path(SRC_ROOM_ROOT), device)
        print(f"room_01 {device}: mẫu theo nhóm {[accs[device][g][3] for g in FOLDS]} "
              f"({time.time() - t0:.0f}s)", flush=True)
    accs["xrf55"] = accumulate_xrf55(SRC_XRF55_ROOT)
    print(f"xrf55 room1: mẫu theo nhóm {[accs['xrf55'][g][3] for g in FOLDS]} ({time.time() - t0:.0f}s)")

    stats = {f"fold{k}": {d: fold_stats(a, k) for d, a in accs.items()} for k in FOLDS}
    OUT_FILE.write_text(json.dumps({
        "src_room_root": SRC_ROOM_ROOT,
        "src_xrf55_root": SRC_XRF55_ROOT,
        "split": "cross-env source room 1: fold k: val = repetition group k, train = other 4 groups "
                 "(labels 1-8); groups as v3",
        "note": "global z-score stats on raw samples (1000 time steps), applied to train, val and "
                "the whole target room",
        "stats": stats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    for k in FOLDS:
        line = "  ".join(f"{d}: mean={v['mean']:.4f} std={v['std']:.4f} n={v['n_samples']}"
                         for d, v in stats[f"fold{k}"].items())
        print(f"  fold{k}  {line}")
    print("Đã lưu", OUT_FILE)


if __name__ == "__main__":
    main()
