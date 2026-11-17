"""
Plot v3 accuracy (5-fold over repetitions, best_val checkpoint): mean over folds with ±std bars.

Same layout as plot_results_v2.py --figure same:
    x axis : model (CNN-GRU, BiMamba)
    lines  : one per dataset/platform; colour = dataset (Room 1, Room 2, XRF55),
             style = platform (ESP32: solid + circle, ASUS: dashed + square, XRF55: solid + triangle)
    series are shifted slightly along x so that the error bars do not overlap.

Reads results/v3/<config>_<model>_fold<k>/<selection>/ (see aggregate_v3.py),
writes images/acc_v3_same_platform.{pdf,png} (--actions 6: acc_v3_same_platform_6act.{pdf,png}).

Usage:
    python plot_results_v3.py [--results results/v3] [--out images]
    python plot_results_v3.py --actions 6 --ymin 84
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_v3 import MODELS, collect, mean_std
from plot_results_v2 import SERIES_SAME, TEXT, MUTED, GRID, spread

SERIES = SERIES_SAME          # run prefixes: room_0X_esp_esp, room_0X_asus_asus, xrf55
DODGE = 0.05


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/v3")
    ap.add_argument("--out", default="images")
    ap.add_argument("--selection", default="best_val")
    ap.add_argument("--actions", type=int, default=8, choices=[8, 6])
    ap.add_argument("--ymin", type=float, default=86)
    ap.add_argument("--ymax", type=float, default=100)
    ap.add_argument("--formats", default="pdf,png")
    args = ap.parse_args()
    key = "acc" if args.actions == 8 else "acc6"
    results = Path(args.results)

    stats = {}
    for p, label, *_ in SERIES:
        stats[p] = []
        for m, mname in MODELS:
            runs = collect(results, p, m, args.selection)
            mu, sd = mean_std([runs[k][key] for k in sorted(runs)])
            if mu is None:
                raise FileNotFoundError(f"no fold found for {p}_{m}_fold*/{args.selection}")
            stats[p].append((mu, sd, len(runs)))
        print(f"{label:16s} " + "  ".join(f"{mname}={mu:.2f}±{sd:.2f} (n={n})"
                                         for (_, mname), (mu, sd, n) in zip(MODELS, stats[p])))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    xs = list(range(len(MODELS)))
    n = len(SERIES)
    for i, (p, label, color, ls, marker) in enumerate(SERIES):
        dx = (i - (n - 1) / 2) * DODGE
        xi = [x + dx for x in xs]
        mus = [s[0] for s in stats[p]]
        sds = [s[1] for s in stats[p]]
        ax.errorbar(xi, mus, yerr=sds, color=color, ls=ls, lw=2, marker=marker, ms=7,
                    mec="white", mew=1.2, capsize=3, elinewidth=1.2, label=label, zorder=3)

    # value labels (mean): left of CNN-GRU, right of BiMamba, de-overlapped
    min_gap = 0.042 * (args.ymax - args.ymin)
    for k, x in enumerate(xs):
        vals = [stats[p][k][0] for p, *_ in SERIES]
        ys = spread(vals, min_gap)
        side = -1 if k == 0 else 1
        edge = x + side * ((n - 1) / 2 * DODGE + 0.03)
        for i, (v, y) in enumerate(zip(vals, ys)):
            px = x + (i - (n - 1) / 2) * DODGE + side * 0.03      # just outside this series' marker
            tx = edge + side * 0.13
            ax.plot([px, tx - side * 0.01], [v, y], color=GRID, lw=0.8, zorder=1)
            ax.text(tx, y, f"{v:.2f}", va="center", ha="right" if side < 0 else "left",
                    fontsize=8, color=TEXT, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))

    ax.set_xticks(xs, [name for _, name in MODELS])
    ax.tick_params(axis="x", labelsize=10, colors=TEXT, length=0)
    ax.set_xlim(-0.6, len(MODELS) - 1 + 0.6)
    ax.set_ylim(args.ymin, args.ymax)
    ax.set_ylabel("Accuracy (%)" if args.actions == 8 else "Accuracy on 6 actions (%)")
    ax.text(0.01, 0.99, "mean ± std over 5 folds", transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False,
              fontsize=8, handlelength=2.6, columnspacing=1.2, labelcolor=TEXT)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.actions == 8 else "_6act"
    for ext in [e.strip() for e in args.formats.split(",") if e.strip()]:
        f = out / f"acc_v3_same_platform{suffix}.{ext}"
        fig.savefig(f, dpi=300)
        print("saved", f)


if __name__ == "__main__":
    main()
