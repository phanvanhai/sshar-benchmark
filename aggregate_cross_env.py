"""
Aggregate cross-environment results room 1 -> room 2 (doc/experiment_plan_cross_env.md).

Reads results/cross_env/r1_to_r2_<device>_<model>_fold<k>/<selection>/{metrics.json, predictions.csv}
    device = esp, asus, xrf55
predictions.csv has a "subject" column; metrics.json lists "shared_subjects" (in both rooms).

For every (device, model, selection) and subject group
    all    : every test sample of room 2
    shared : subjects that are also in room 1
    new    : subjects only in room 2
reports mean +- std (ddof=1) over the folds of
    acc, precision_macro, recall_macro, f1_macro (8 classes) and acc6 (true label in the 6 actions).

Writes results/cross_env/summary_cross_env.csv and prints the tables.

Usage:
    python aggregate_cross_env.py [--results results/cross_env] [--selection best_val,last_epoch]
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

FOLDS = (1, 2, 3, 4, 5)
MODELS = [("cnn_gru", "CNN-GRU"), ("bimamba", "BiMamba")]
DEVICES = [("esp", "ESP32"), ("asus", "ASUS"), ("xrf55", "XRF55")]
GROUPS = ("all", "shared", "new")
METRICS = ("acc", "precision_macro", "recall_macro", "f1_macro", "acc6")
EXTRA_CLASSES = {"lie still", "empty room", "play ukulele", "play drum"}


def load_fold(run_dir):
    mf, pf = run_dir / "metrics.json", run_dir / "predictions.csv"
    if not (mf.exists() and pf.exists()):
        return None
    m = json.loads(mf.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(pf.open(encoding="utf-8")))
    if not rows or "subject" not in rows[0]:
        raise ValueError(f"{pf}: no 'subject' column")
    return m, rows


def group_metrics(rows, class_names, keep):
    rows = [r for r in rows if keep(r["subject"])]
    if not rows:
        return None
    y = [int(r["true"]) for r in rows]
    p = [int(r["pred"]) for r in rows]
    labels = list(range(len(class_names)))
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, average="macro", zero_division=0)
    six = [i for i, n in enumerate(class_names) if n not in EXTRA_CLASSES]
    y6 = [(a, b) for a, b in zip(y, p) if a in six]
    return {
        "acc": 100 * accuracy_score(y, p),
        "precision_macro": 100 * pr,
        "recall_macro": 100 * rc,
        "f1_macro": 100 * f1,
        "acc6": 100 * sum(a == b for a, b in y6) / len(y6) if y6 else None,
        "n": len(rows),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/cross_env")
    ap.add_argument("--selection", default="best_val,last_epoch")
    args = ap.parse_args()
    results = Path(args.results)
    selections = [s.strip() for s in args.selection.split(",") if s.strip()]

    out_rows = []
    for sel in selections:
        print(f"\n=== {sel}: room 1 -> room 2, mean ± std over folds (%) ===")
        print(f"{'device':6s} {'model':8s} {'group':7s} {'folds':6s} {'n':>5s} "
              + " ".join(f"{k:>15s}" for k in METRICS))
        for dev, dname in DEVICES:
            for model, mname in MODELS:
                per_fold = {}
                for k in FOLDS:
                    r = load_fold(results / f"r1_to_r2_{dev}_{model}_fold{k}" / sel)
                    if r is not None:
                        per_fold[k] = r
                for g in GROUPS:
                    vals = {}
                    for k, (m, rows) in per_fold.items():
                        shared = set(m.get("shared_subjects", []))
                        keep = {"all": lambda s: True,
                                "shared": lambda s, sh=shared: s in sh,
                                "new": lambda s, sh=shared: s not in sh}[g]
                        gm = group_metrics(rows, m["class_names"], keep)
                        if gm is not None:
                            vals[k] = gm
                    row = {"device": dev, "model": model, "selection": sel, "group": g,
                           "folds": "".join(str(k) for k in sorted(vals)),
                           "n_test": vals[min(vals)]["n"] if vals else 0}
                    cells = []
                    for key in METRICS:
                        xs = [vals[k][key] for k in sorted(vals) if vals[k][key] is not None]
                        mu = statistics.mean(xs) if xs else None
                        sd = (statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else None
                        row[f"{key}_mean"], row[f"{key}_std"] = mu, sd
                        for k in FOLDS:
                            row[f"{key}_fold{k}"] = vals[k][key] if k in vals else None
                        cells.append(f"{'-':>15s}" if mu is None else f"{mu:8.2f} ± {sd:4.2f}")
                    out_rows.append(row)
                    if vals:
                        print(f"{dname:6s} {mname:8s} {g:7s} {row['folds']:6s} {row['n_test']:5d} "
                              + " ".join(cells))

    if not out_rows:
        print("no results found in", results)
        return
    out = results / "summary_cross_env.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print("\nsaved", out)


if __name__ == "__main__":
    main()
