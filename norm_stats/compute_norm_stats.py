DATASET_ROOT = r"D:\workspace\HUST\Personal\WiFi-Sensing\Code\CodeForPaper\benchmark\data\sshar_amp_interp\room_02"

"""
Tính sẵn các cặp mean/std (z-score toàn cục) trên TẬP TRAIN theo kế hoạch v2
(doc/experiment_plan_v2.md).

Tập train (chia theo lần lặp 6/2/2):
    hành động không hướng (dir00, 10 lần lặp): rep 1-6
    hành động có hướng (dir01/02, 5 lần lặp) : rep 1-3

Tính trên dữ liệu gốc (đủ 1000 điểm thời gian), gộp mọi rx, mọi subcarrier, 8 lớp:
    esp        : ESP32, 3 rx x 1 ăng-ten           -> thí nghiệm 1, test của thí nghiệm 3
    asus       : ASUS, 3 rx x 4 ăng-ten            -> thí nghiệm 2
    asus_ant0  : ASUS, 3 rx x ăng-ten 0            -> train/val của thí nghiệm 3
    asus_ant1..3: dự phòng

Cấu trúc dữ liệu: DATASET_ROOT/<device>/<rx>/<subject>/amp_actXX_posXX_dirXX_repXX.npy
Kết quả: norm_stats_v2.json (cùng thư mục với file này).

Chạy:
    python norm_stats/compute_norm_stats.py
"""
import json
import re
import time
from pathlib import Path

import numpy as np

PATTERN = re.compile(r"act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)")
RX = ["rx_00", "rx_01", "rx_02"]
LABELS = set(range(1, 9))
OUT_FILE = Path(__file__).resolve().parent / "norm_stats_v2.json"


def is_train(direction, rep):
    return rep <= 6 if direction == 0 else rep <= 3


def accumulate(room_dir, device):
    """Cộng dồn theo từng ăng-ten (float64): tổng, tổng bình phương, số phần tử."""
    base = room_dir / device / RX[0]
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
                x = np.load(room_dir / device / rx / subject_dir.name / f.name).astype(np.float64)
                if x.ndim != 3 or x.shape[1] != 56:
                    raise ValueError(f"shape {x.shape} không đúng: {device}/{rx}/{subject_dir.name}/{f.name}")
                if s is None:
                    s, sq, cnt = np.zeros(x.shape[0]), np.zeros(x.shape[0]), np.zeros(x.shape[0])
                s += x.sum(axis=(1, 2))
                sq += np.square(x).sum(axis=(1, 2))
                cnt += x.shape[1] * x.shape[2]
            n_samples += 1
    return s, sq, cnt, n_samples


def mean_std(s, sq, cnt):
    mean = s / cnt
    return float(mean), float(np.sqrt(sq / cnt - mean ** 2))


def main():
    room_dir = Path(DATASET_ROOT)
    t0 = time.time()
    stats = {}

    s, sq, c, n = accumulate(room_dir, "esp")
    print(f"esp : {n} mẫu train ({time.time() - t0:.0f}s)", flush=True)
    stats["esp"] = dict(zip(("mean", "std"), mean_std(s.sum(), sq.sum(), c.sum())), n_samples=n)

    s, sq, c, n = accumulate(room_dir, "asus")
    print(f"asus: {n} mẫu train ({time.time() - t0:.0f}s)", flush=True)
    stats["asus"] = dict(zip(("mean", "std"), mean_std(s.sum(), sq.sum(), c.sum())), n_samples=n)
    for k in range(len(s)):
        stats[f"asus_ant{k}"] = dict(zip(("mean", "std"), mean_std(s[k], sq[k], c[k])), n_samples=n)

    OUT_FILE.write_text(json.dumps({
        "dataset_root": DATASET_ROOT,
        "split": "train: dir00 rep1-6, directional rep1-3 (labels 1-8)",
        "note": "global z-score stats on raw samples (1000 time steps), 56 subcarriers",
        "stats": stats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    for k, v in stats.items():
        print(f"  {k:10s} mean={v['mean']:.4f}  std={v['std']:.4f}  n={v['n_samples']}")
    print("Đã lưu", OUT_FILE)


if __name__ == "__main__":
    main()
