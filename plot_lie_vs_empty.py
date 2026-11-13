"""
Plot ESP32 CSI amplitude of one subcarrier: Fall (act04), Lie still (act05), Empty room (act08).

One figure per subject: 3 rows (Fall, Lie still, Empty room) x 3 columns (rx_00, rx_01, rx_02,
one antenna each on ESP32). Raw amplitude is drawn in grey; a zero-phase Butterworth
low-pass filtered version is drawn on top in colour to make slow fluctuations visible.
Fall dirXX repYY is recorded immediately before Lie still dirXX repYY (same sequence).

--compare dir (default): Fall / Lie still = dir01 vs dir02 at the first repetition (rep01);
                         Empty room has only dir00 -> rep01 vs rep10.
--compare rep          : first vs last repetition (Fall / Lie still dir01 rep01/rep05,
                         Empty room rep01/rep10).

Usage:
    python plot_lie_vs_empty.py --pos 1
    python plot_lie_vs_empty.py --pos 3 --sub 20 --cutoff 1.0
    python plot_lie_vs_empty.py --pos 1 --compare rep
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

FS = 200.0  # Hz
RX = ["rx_00", "rx_01", "rx_02"]
COLORS = ("tab:blue", "tab:orange")
RAW_GREYS = ("0.55", "0.80")

# class label, act, list of (dir, rep) to overlay
CLASSES = {
    "dir": [
        ("Fall (act04)", 4, [(1, 1), (2, 1)]),
        ("Lie still (act05)", 5, [(1, 1), (2, 1)]),
        ("Empty room (act08)", 8, [(0, 1), (0, 10)]),
    ],
    "rep": [
        ("Fall (act04)", 4, [(1, 1), (1, 5)]),
        ("Lie still (act05)", 5, [(1, 1), (1, 5)]),
        ("Empty room (act08)", 8, [(0, 1), (0, 10)]),
    ],
}


def load(root, rx, subject, act, pos, direction, rep):
    f = root / rx / subject / f"amp_act{act:02d}_pos{pos:02d}_dir{direction:02d}_rep{rep:02d}.npy"
    if not f.exists():
        return None
    return np.load(f).astype(np.float32)  # (1, 56, T)


def lowpass(y, cutoff, order=4):
    b, a = butter(order, cutoff / (FS / 2), btype="low")
    return filtfilt(b, a, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/sshar_amp_interp")
    ap.add_argument("--room", default="room_02")
    ap.add_argument("--sub", type=int, default=10, help="subcarrier index 0..55")
    ap.add_argument("--pos", type=int, default=1)
    ap.add_argument("--compare", choices=["dir", "rep"], default="dir")
    ap.add_argument("--cutoff", type=float, default=2.0, help="low-pass cutoff (Hz)")
    ap.add_argument("--out", default="images")
    args = ap.parse_args()

    root = Path(args.data_root) / args.room / "esp"
    out = Path(args.out) / (f"fall_lie_empty_{args.room}_sub{args.sub:02d}_pos{args.pos:02d}"
                            f"_{args.compare}_lp{args.cutoff:g}Hz")
    out.mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.name for p in (root / RX[0]).iterdir() if p.is_dir())

    for subject in subjects:
        rows = CLASSES[args.compare]
        fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.2 * len(rows)), sharex=True,
                                 constrained_layout=True)
        for r, (name, act, variants) in enumerate(rows):
            for c, rx in enumerate(RX):
                ax = axes[r, c]
                for k, (d, rep) in enumerate(variants):
                    x = load(root, rx, subject, act, args.pos, d, rep)
                    if x is None:
                        continue
                    y = x[0, args.sub]
                    t = np.arange(y.size) / FS
                    yf = lowpass(y, args.cutoff)
                    ax.plot(t, y, color=RAW_GREYS[k], lw=0.6, zorder=1)
                    ax.plot(t, yf, color=COLORS[k], lw=1.6, zorder=2,
                            label=f"dir{d:02d} rep{rep:02d}  mean={y.mean():.1f} "
                                  f"std raw={y.std():.2f} / LP={yf.std():.2f}")
                ax.set_title(f"{name} | {rx}", fontsize=10)
                ax.legend(fontsize=7.5, loc="upper right")
                ax.grid(alpha=0.3)
                if c == 0:
                    ax.set_ylabel("Amplitude")
                if r == len(rows) - 1:
                    ax.set_xlabel("Time (s)")
        fig.suptitle(f"ESP32 {args.room} {subject} | subcarrier {args.sub} | pos {args.pos:02d} | "
                     f"raw (grey) + low-pass {args.cutoff:g} Hz (colour)", fontsize=12)
        fig.savefig(out / f"{subject}.png", dpi=120)
        plt.close(fig)
        print("saved", out / f"{subject}.png")


if __name__ == "__main__":
    main()
