"""
Plot v2 accuracy (repeat split 6/2/2, best_val checkpoint).

--figure same  (default): same-platform runs
--figure cross          : cross-platform runs ESP32 <-> ASUS antenna 0
                          (colour = room, solid + square = ASUS0 -> ESP32, dashed + circle = ESP32 -> ASUS0,
                           grey dotted line = chance level 12.5 %)

x axis : model (CNN-GRU, BiMamba)
lines  : one per dataset/platform
         colour = dataset (Room 1, Room 2, XRF55)
         style  = platform (ESP32: solid + circle, ASUS: dashed + square, XRF55: solid + triangle)

Reads results/v2/<run>/best_val/metrics.json (or confusion_matrix.csv for --actions 6),
writes images/acc_same_platform.{pdf,png} (--actions 6: acc_same_platform_6act.{pdf,png}).

Usage:
    python plot_results_v2.py [--results results/v2] [--out images]
    python plot_results_v2.py --actions 6 --ymin 86
    python plot_results_v2.py --figure cross --formats png
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("cnn_gru", "CNN-GRU"), ("bimamba", "BiMamba")]

# (run prefix, legend label, colour, linestyle, marker)
SERIES_SAME = [
    ("room_01_esp_esp",     "Room 1 – ESP32", "#2a78d6", "-",  "o"),
    ("room_01_asus_asus",   "Room 1 – ASUS",  "#2a78d6", "--", "s"),
    ("room_02_esp_esp",     "Room 2 – ESP32", "#eb6834", "-",  "o"),
    ("room_02_asus_asus",   "Room 2 – ASUS",  "#eb6834", "--", "s"),
    ("xrf55",               "XRF55",          "#1baf7a", "-",  "^"),
]

SERIES_CROSS = [
    ("room_01_asus0_esp",   "Room 1: ASUS0 → ESP32", "#2a78d6", "-",  "s"),
    ("room_01_esp_asus0",   "Room 1: ESP32 → ASUS0", "#2a78d6", "--", "o"),
    ("room_02_asus0_esp",   "Room 2: ASUS0 → ESP32", "#eb6834", "-",  "s"),
    ("room_02_esp_asus0",   "Room 2: ESP32 → ASUS0", "#eb6834", "--", "o"),
]
CHANCE = 100 / 8

TEXT = "#2b2b29"
MUTED = "#6b6a63"
GRID = "#e4e3de"


# --actions 6: drop the two extra classes (room: lie still / empty room; XRF55: play ukulele / play drum)
EXTRA_CLASSES = {"lie still", "empty room", "play ukulele", "play drum"}


def load_acc(results, prefix, model, actions=8):
    """
    actions=8: test accuracy from metrics.json.
    actions=6: accuracy over test samples whose TRUE label is one of the 6 actions; the model
               still predicts among all 8 classes, so a prediction into an extra class is wrong.
    """
    run = results / f"{prefix}_{model}" / "best_val"
    if not run.exists() and prefix.startswith("room_0"):
        # tolerate run folders renamed by hand as room1_ / room2_
        run = results / f"room{prefix[6]}_{prefix[8:]}_{model}" / "best_val"
    if actions == 8:
        f = run / "metrics.json"
        if not f.exists():
            raise FileNotFoundError(f)
        return 100 * json.loads(f.read_text(encoding="utf-8"))["accuracy"]
    f = run / "confusion_matrix.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    rows = list(csv.reader(f.open(encoding="utf-8")))
    names = rows[0][1:]
    cm = [[int(v) for v in r[1:]] for r in rows[1:]]
    keep = [i for i, n in enumerate(names) if n not in EXTRA_CLASSES]
    if len(keep) != 6:
        raise ValueError(f"{f}: expected 6 actions after dropping {EXTRA_CLASSES}, got {len(keep)}")
    correct = sum(cm[i][i] for i in keep)
    total = sum(sum(cm[i]) for i in keep)
    return 100 * correct / total


def spread(values, min_gap):
    """Label y positions: keep order, push apart until neighbours are >= min_gap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = {i: values[i] for i in order}
    for _ in range(50):
        moved = False
        for a, b in zip(order, order[1:]):
            gap = pos[b] - pos[a]
            if gap < min_gap:
                d = (min_gap - gap) / 2
                pos[a] -= d
                pos[b] += d
                moved = True
        if not moved:
            break
    return [pos[i] for i in range(len(values))]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/v2")
    ap.add_argument("--out", default="images")
    ap.add_argument("--figure", default="same", choices=["same", "cross"])
    ap.add_argument("--ymin", type=float, default=None, help="default 88 (same) / 0 (cross)")
    ap.add_argument("--ymax", type=float, default=None, help="default 100 (same) / 30 (cross)")
    ap.add_argument("--formats", default="pdf,png", help="comma-separated, e.g. png")
    ap.add_argument("--actions", type=int, default=8, choices=[8, 6],
                    help="8: all classes; 6: only samples of the 6 actions (see load_acc)")
    args = ap.parse_args()
    cross = args.figure == "cross"
    SERIES = SERIES_CROSS if cross else SERIES_SAME
    if args.ymin is None:
        args.ymin = 0 if cross else 88
    if args.ymax is None:
        args.ymax = 30 if cross else 100

    results = Path(args.results)
    acc = {p: [load_acc(results, p, m, args.actions) for m, _ in MODELS] for p, *_ in SERIES}
    for (p, label, *_ ) in SERIES:
        print(f"{label:16s} " + "  ".join(f"{name}={a:.2f}" for (_, name), a in zip(MODELS, acc[p])))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    xs = list(range(len(MODELS)))
    if cross:
        ax.axhline(CHANCE, color=MUTED, ls=":", lw=1.2, zorder=2)
        ax.text(0.5, CHANCE - 0.4, f"chance level ({CHANCE:g} %)", ha="center", va="top",
                fontsize=8, color=MUTED)

    for p, label, color, ls, marker in SERIES:
        ax.plot(xs, acc[p], color=color, ls=ls, lw=2, marker=marker, ms=7,
                mec="white", mew=1.2, label=label, zorder=3)

    # value labels: left of CNN-GRU points, right of BiMamba points, de-overlapped
    min_gap = 0.042 * (args.ymax - args.ymin)
    for k, x in enumerate(xs):
        vals = [acc[p][k] for p, *_ in SERIES]
        ys = spread(vals, min_gap)
        side = -1 if k == 0 else 1
        for (p, _, color, *_), v, y in zip(SERIES, vals, ys):
            tx = x + side * 0.16
            ax.plot([x + side * 0.05, tx - side * 0.01], [v, y], color=GRID, lw=0.8, zorder=1)
            ax.text(tx, y, f"{v:.2f}", va="center", ha="right" if side < 0 else "left",
                    fontsize=8, color=TEXT, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

    ax.set_xticks(xs, [name for _, name in MODELS])
    ax.tick_params(axis="x", labelsize=10, colors=TEXT, length=0)
    ax.set_xlim(-0.55, len(MODELS) - 1 + 0.55)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_ylabel("Accuracy (%)" if args.actions == 8 else "Accuracy on 6 actions (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2 if cross else 3, frameon=False,
              fontsize=8, handlelength=2.6, columnspacing=1.2, labelcolor=TEXT)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "acc_cross_platform" if cross else "acc_same_platform"
    suffix = "" if args.actions == 8 else "_6act"
    for ext in [e.strip() for e in args.formats.split(",") if e.strip()]:
        f = out / f"{name}{suffix}.{ext}"
        fig.savefig(f, dpi=300)
        print("saved", f)


if __name__ == "__main__":
    main()
