"""
Plot consecutive windows of one action sequence on a single time axis (ESP32, one subcarrier).

Windows with the same subject/pos/dir/rep but different actions follow each other in time
inside one sequence (e.g. pickup -> fall -> lie still), with no pause in between.

One row per receiver (rx_00, rx_01, rx_02); raw amplitude in grey, zero-phase low-pass in
colour, a dashed line at each window boundary.

Usage:
    python plot_sequence.py --subject subject_11 --pos 1 --dir 1 --rep 5 --acts 4 5
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

FS = 200.0
RX = ["rx_00", "rx_01", "rx_02"]
ACT_NAMES = {1: "sitdown", 2: "standup", 3: "pickup", 4: "fall",
             5: "lie still", 6: "walk", 7: "run", 8: "empty room"}
COLORS = ["tab:red", "tab:blue", "tab:green", "tab:purple"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/sshar_amp_interp/room_02/esp")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--dir", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--acts", type=int, nargs="+", required=True, help="actions in time order, e.g. 4 5")
    ap.add_argument("--sub", type=int, default=10, help="subcarrier index 0..55")
    ap.add_argument("--cutoff", type=float, default=2.0, help="low-pass cutoff (Hz)")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="images")
    args = ap.parse_args()

    root = Path(args.data_root)
    b, a = butter(4, args.cutoff / (FS / 2), "low")
    names = [f"amp_act{act:02d}_pos{args.pos:02d}_dir{args.dir:02d}_rep{args.rep:02d}.npy" for act in args.acts]

    fig, axes = plt.subplots(len(RX), 1, figsize=(14, 2.8 * len(RX)), sharex=True, constrained_layout=True)
    for r, rx in enumerate(RX):
        ax = axes[r]
        t0 = 0.0
        for k, (act, name) in enumerate(zip(args.acts, names)):
            y = np.load(root / rx / args.subject / name).astype(np.float64)[0, args.sub]
            t = t0 + np.arange(y.size) / FS
            ax.plot(t, y, color="0.7", lw=0.6, zorder=1)
            ax.plot(t, filtfilt(b, a, y), color=COLORS[k % len(COLORS)], lw=1.6, zorder=2,
                    label=f"act{act:02d} {ACT_NAMES.get(act, '')}  mean={y.mean():.1f} std={y.std():.2f}")
            if k > 0:
                ax.axvline(t0, color="k", ls="--", lw=1)
            t0 = t[-1] + 1 / FS
        ax.set_title(rx, fontsize=10)
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"ESP32 {args.subject} pos{args.pos:02d} dir{args.dir:02d} rep{args.rep:02d} | "
                 f"subcarrier {args.sub} | raw (grey) + low-pass {args.cutoff:g} Hz | {args.title}", fontsize=11)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / (f"seq_{args.subject}_pos{args.pos:02d}_dir{args.dir:02d}_rep{args.rep:02d}_act"
               + "-".join(f"{a:02d}" for a in args.acts) + f"_sub{args.sub:02d}.png")
    fig.savefig(f, dpi=120)
    print("saved", f)


if __name__ == "__main__":
    main()
