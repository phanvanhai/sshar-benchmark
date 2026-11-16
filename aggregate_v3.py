"""
Aggregate v3 results (5-fold over repetitions, doc/experiment_plan_v3.md).

Reads results/v3/<config>_<model>_fold<k>/<selection>/{metrics.json, confusion_matrix.csv}
    config = room_01_esp_esp, room_01_asus_asus, room_02_esp_esp, room_02_asus_asus, xrf55

Per (config, model, selection) it reports mean +- std over the available folds of
    acc, precision_macro, recall_macro, f1_macro      (8 classes, from metrics.json)
    acc6                                              (6 actions, from the confusion matrix:
                                                       samples whose TRUE label is one of the 6 actions)
std = sample standard deviation (ddof=1) over folds.

Writes
    results/v3/summary_v3.csv                      one row per (config, model, selection), with per-fold values
    results/v3/cm_sum/<config>_<model>_<sel>.csv   confusion matrix summed over folds

Usage:
    python aggregate_v3.py [--results results/v3] [--selection best_val,last_epoch]
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

FOLDS = (1, 2, 3, 4, 5)
MODELS = [("cnn_gru", "CNN-GRU"), ("bimamba", "BiMamba")]
CONFIGS = [
    ("room_01_esp_esp",   "Room 1 – ESP32"),
    ("room_01_asus_asus", "Room 1 – ASUS"),
    ("room_02_esp_esp",   "Room 2 – ESP32"),
    ("room_02_asus_asus", "Room 2 – ASUS"),
    ("xrf55",             "XRF55"),
]
METRICS = ("acc", "precision_macro", "recall_macro", "f1_macro", "acc6")
# --6 actions: drop the two extra classes (room: lie still / empty room; XRF55: play ukulele / play drum)
EXTRA_CLASSES = {"lie still", "empty room", "play ukulele", "play drum"}


def read_cm(f):
    rows = list(csv.reader(f.open(encoding="utf-8")))
    return rows[0][1:], [[int(v) for v in r[1:]] for r in rows[1:]]


def acc6_from_cm(names, cm):
    keep = [i for i, n in enumerate(names) if n not in EXTRA_CLASSES]
    if len(keep) != 6:
        raise ValueError(f"expected 6 actions after dropping {EXTRA_CLASSES}, got {len(keep)}")
    return 100 * sum(cm[i][i] for i in keep) / sum(sum(cm[i]) for i in keep)


def load_run(run_dir):
    """Metrics (%) + confusion matrix of one checkpoint folder, or None if missing."""
    mf, cf = run_dir / "metrics.json", run_dir / "confusion_matrix.csv"
    if not (mf.exists() and cf.exists()):
        return None
    m = json.loads(mf.read_text(encoding="utf-8"))
    names, cm = read_cm(cf)
    return {
        "acc": 100 * m["accuracy"],
        "precision_macro": 100 * m["precision_macro"],
        "recall_macro": 100 * m["recall_macro"],
        "f1_macro": 100 * m["f1_macro"],
        "acc6": acc6_from_cm(names, cm),
        "names": names,
        "cm": cm,
    }


def collect(results, config, model, selection):
    """{fold: run dict} for the folds that exist."""
    out = {}
    for k in FOLDS:
        r = load_run(results / f"{config}_{model}_fold{k}" / selection)
        if r is not None:
            out[k] = r
    return out


def mean_std(values):
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def summarize(results, config, model, selection):
    runs = collect(results, config, model, selection)
    row = {"config": config, "model": model, "selection": selection,
           "folds": "".join(str(k) for k in sorted(runs)), "n_folds": len(runs)}
    for key in METRICS:
        vals = [runs[k][key] for k in sorted(runs)]
        row[f"{key}_mean"], row[f"{key}_std"] = mean_std(vals)
        for k in FOLDS:
            row[f"{key}_fold{k}"] = runs[k][key] if k in runs else None
    cm_sum = None
    if runs:
        names = runs[min(runs)]["names"]
        cm_sum = [[sum(runs[k]["cm"][i][j] for k in runs) for j in range(len(names))]
                  for i in range(len(names))]
        cm_sum = (names, cm_sum)
    return row, cm_sum


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/v3")
    ap.add_argument("--selection", default="best_val,last_epoch")
    args = ap.parse_args()
    results = Path(args.results)
    selections = [s.strip() for s in args.selection.split(",") if s.strip()]

    rows = []
    cm_dir = results / "cm_sum"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for sel in selections:
        print(f"\n=== {sel}: mean ± std over folds (%) ===")
        print(f"{'config':16s} {'model':8s} {'folds':6s} " + " ".join(f"{k:>16s}" for k in METRICS))
        for config, label in CONFIGS:
            for model, mname in MODELS:
                row, cm = summarize(results, config, model, sel)
                rows.append(row)
                cells = []
                for key in METRICS:
                    m, s = row[f"{key}_mean"], row[f"{key}_std"]
                    cells.append(f"{'-':>16s}" if m is None else f"{m:9.2f} ± {s:4.2f}")
                print(f"{label:16s} {mname:8s} {row['folds'] or '-':6s} " + " ".join(cells))
                if cm is not None:
                    names, mat = cm
                    with (cm_dir / f"{config}_{model}_{sel}.csv").open("w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow([""] + names)
                        for n, r in zip(names, mat):
                            w.writerow([n] + r)

    out = results / "summary_v3.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("\nsaved", out)
    print("saved", cm_dir)


if __name__ == "__main__":
    main()
