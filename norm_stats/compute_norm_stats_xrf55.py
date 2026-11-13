DATASET_ROOT = r"D:\workspace\HUST\Personal\WiFi-Sensing\Code\CodeForPaper\benchmark\data\XRF55_8act_dataset"

"""
Tính sẵn cặp mean/std (z-score toàn cục) của XRF55 trên TẬP TRAIN theo kế hoạch v2.

XRF55 (bản 8 hành động): file <user>_<action>_<rep>.npy, 30 người x 8 hành động x 20 lần lặp,
mỗi file (270, 1000) = 3 rx x 3 ăng-ten x 30 subcarrier, 1000 điểm thời gian.

Chia theo lần lặp 6/2/2: train rep 1-12, val rep 13-16, test rep 17-20.
Tính trên dữ liệu gốc (đủ 1000 điểm thời gian), gộp mọi kênh.
Kết quả: norm_stats_xrf55_v2.json (cùng thư mục với file này).

Chạy:
    python norm_stats/compute_norm_stats_xrf55.py
"""
import json
import time
from pathlib import Path

import numpy as np

LABELS = set(range(1, 9))
TRAIN_MAX_REP = 12
OUT_FILE = Path(__file__).resolve().parent / "norm_stats_xrf55_v2.json"


def main():
    t0 = time.time()
    s = sq = cnt = 0.0
    n_samples = 0
    for f in sorted(Path(DATASET_ROOT).glob("*.npy")):
        parts = f.stem.split("_")
        if len(parts) != 3:
            continue
        _, act, rep = map(int, parts)
        if act not in LABELS or rep > TRAIN_MAX_REP:
            continue
        x = np.load(f).astype(np.float64)
        if x.shape != (270, 1000):
            raise ValueError(f"shape {x.shape} không đúng: {f.name}")
        s += x.sum()
        sq += np.square(x).sum()
        cnt += x.size
        n_samples += 1

    mean = s / cnt
    std = float(np.sqrt(sq / cnt - mean ** 2))
    OUT_FILE.write_text(json.dumps({
        "dataset_root": DATASET_ROOT,
        "split": f"train: rep 1-{TRAIN_MAX_REP} (labels 1-8, all users)",
        "note": "global z-score stats on raw samples (270 x 1000)",
        "stats": {"xrf55": {"mean": float(mean), "std": std, "n_samples": n_samples}},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"xrf55 mean={mean:.4f}  std={std:.4f}  n={n_samples}  ({time.time() - t0:.0f}s)")
    print("Đã lưu", OUT_FILE)


if __name__ == "__main__":
    main()
