"""
main.py

WiFi HAR Benchmark
"""

import argparse
import os
import random
import time

import numpy as np

import torch

from datasets import load_dataset
from models import get_model
from train import train
from metrics import (
    evaluate_model,
    measure_model_complexity,
    save_model_complexity,
)

# ============================================================
# Global seed
# ============================================================
SEED = 42

def set_seed(seed=SEED):
    """
    Set global random seed for reproducibility.

    Covers
    ------
    - Python `random`
    - NumPy
    - PyTorch (CPU + CUDA)
    - cuDNN deterministic mode
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Argument Parser
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="WiFi HAR Benchmark"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "ut_har",
            "xrf55",
            "sshar_esp",
            "sshar_asus",
        ],
        help="Dataset name",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "mlp",
            "cnn5",
            "bilstm",
            "vit",
            "resnet18",
            "resnet1d",
            "cnn_gru",
            "bimamba",
        ],
        help="Model name",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        nargs="+",
        required=True,
        help="Dataset root directory. SSHAR: one or several roots, each containing "
             "room_XX folders (e.g. the room-1 and room-2 Kaggle inputs).",
    )

    parser.add_argument(
        "--rooms",
        type=str,
        default="all",
        help="SSHAR only: rooms to use. 'all' (every room found under --data_root), "
             "'1', '2', '1,2' or folder names such as 'room_01,room_02'.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["repeat", "user"],
        help="SSHAR only: 'repeat' (first repetitions train) or 'user' "
             "(USER_CASES[case_split] subjects train). Default: SPLIT_BY_REPEAT in datasets.py.",
    )

    parser.add_argument(
        "--norm",
        type=str,
        default="global",
        choices=["global", "sample", "sample_all"],
        help="SSHAR / XRF55 z-score: 'global' (one mean/std from the TRAIN split), "
             "'sample' (each sample, each receiver normalized by its own mean/std) or "
             "'sample_all' (each sample with one mean/std over all receivers).",
    )

    parser.add_argument(
        "--result_root",
        type=str,
        default="results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=[
            "none",
            "steplr",
            "multisteplr",
            "cosineannealinglr",
            "reducelronplateau",
            "onecyclelr",
        ],
        help="Optional learning rate scheduler",
    )

    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=30,
        help="Step size for StepLR",
    )

    parser.add_argument(
        "--scheduler-milestones",
        type=str,
        default="",
        help="Comma-separated milestones for MultiStepLR",
    )

    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Gamma for scheduler decay",
    )

    parser.add_argument(
        "--scheduler-t-max",
        type=int,
        default=50,
        help="T_max for CosineAnnealingLR",
    )

    parser.add_argument(
        "--scheduler-eta-min",
        type=float,
        default=0.0,
        help="Eta min for CosineAnnealingLR",
    )

    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=10,
        help="Patience for ReduceLROnPlateau",
    )

    parser.add_argument(
        "--scheduler-threshold",
        type=float,
        default=1e-4,
        help="Threshold for ReduceLROnPlateau",
    )

    parser.add_argument(
        "--scheduler-max-lr",
        type=float,
        default=None,
        help="Max LR for OneCycleLR",
    )

    parser.add_argument(
        "--scheduler-steps-per-epoch",
        type=int,
        default=None,
        help="Steps per epoch for OneCycleLR",
    )

    parser.add_argument(
        "--scheduler-total-steps",
        type=int,
        default=None,
        help="Total steps for OneCycleLR",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=[
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
            "--case_split",
            type=int,
            default=0,
            choices=[0, 1, 2, 3, 4],
            help="case split from 0 to 4"
        )

    args = parser.parse_args()

    if not args.dataset.startswith("sshar") and len(args.data_root) > 1:
        parser.error("several --data_root values are only supported for sshar_* datasets")
    if not args.dataset.startswith("sshar"):
        args.data_root = args.data_root[0]

    args.room_list = parse_rooms(args.rooms)
    return args


def parse_rooms(rooms):
    """'all' -> None; '1,2' / 'room_01,room_02' -> ['room_01', 'room_02']."""
    if rooms is None or rooms.strip().lower() == "all":
        return None
    out = []
    for r in rooms.split(","):
        r = r.strip()
        if not r:
            continue
        out.append(f"room_{int(r):02d}" if r.isdigit() else r)
    return sorted(set(out))


def run_tag(args, train_dataset):
    """
    Suffix for the result folder so per-room / pooled / split runs do not overwrite
    each other, built from what the dataset actually loaded.
    e.g. sshar_esp_cnn_gru_room01+02_repeat, sshar_asus_bimamba_room02_user_case1,
         sshar_esp_cnn_gru_room01+02_repeat_normsample (per-sample, per-RX z-score)
    """
    norm_tag = "" if args.norm == "global" else f"_norm{args.norm.replace('_', '')}"
    if not args.dataset.startswith("sshar"):
        return norm_tag
    rooms = "+".join(r.replace("room_", "") for r in train_dataset.rooms)
    tag = f"_room{rooms}_{train_dataset.split_mode}"
    if train_dataset.split_mode == "user":
        tag += f"_case{args.case_split}"
    return tag + norm_tag


def get_default_scheduler_cfg(model_name):
    model_name = model_name.lower()
    if model_name == "resnet1d":
        return {
            "name": "multisteplr",
            "args": {
                "milestones": [40, 80, 120, 160],
                "gamma": 0.5,
            },
        }
    if model_name == "resnet18":
        return {
            "name": "steplr",
            "args": {
                "step_size": 30,
                "gamma": 0.1,
            },
        }
    return None


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    set_seed(SEED)
    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("WiFi HAR Benchmark")
    print("=" * 60)
    print("Dataset :", args.dataset)
    print("Model   :", args.model)
    print("Device  :", device)
    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    dataset_kwargs = {}
    dataset_kwargs["case_split"] = args.case_split
    if args.dataset in ("sshar_esp", "sshar_asus", "xrf55"):
        dataset_kwargs["norm_type"] = {
            "global": "zscore",
            "sample": "zscore_sample",          # per sample, per RX
            "sample_all": "zscore_sample_all",  # per sample, all RX together
        }[args.norm]
    elif args.norm != "global":
        raise ValueError(f"--norm {args.norm} is only supported for sshar_* and xrf55")
    if args.dataset.startswith("sshar"):
        dataset_kwargs["rooms"] = args.room_list
        dataset_kwargs["split_mode"] = args.split

    if args.model == "bimamba":
        dataset_kwargs["target_time"] = 250

    if args.model == "cnn_gru":
        train_loader, test_loader, info = load_dataset(
            name=args.dataset,
            root_dir=args.data_root,
            batch_size=args.batch_size,
            shape_option="3d",
            intel_shape=True,
            **dataset_kwargs,
        )
    else:
        train_loader, test_loader, info = load_dataset(
            name=args.dataset,
            root_dir=args.data_root,
            batch_size=args.batch_size,
            intel_shape=True,
            **dataset_kwargs,
        )

    input_shape = info["input_shape"]
    num_classes = info["num_classes"]
    train_size = info["train_size"]
    test_size = info["test_size"]

    print("Input Shape :", input_shape)
    print("Classes     :", num_classes)
    print("Train Size  :", train_size)
    print("Test Size   :", test_size)
    print()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = get_model(
        model_name=args.model,
        input_shape=input_shape,
        num_classes=num_classes,
        dataset=args.dataset,
    )
    model = model.to(device)

    scheduler_cfg = get_default_scheduler_cfg(args.model)
    if args.scheduler and args.scheduler.lower() != "none":
        scheduler_cfg = {
            "name": args.scheduler,
            "args": {
                "step_size": args.scheduler_step_size,
                "gamma": args.scheduler_gamma,
                "milestones": args.scheduler_milestones,
                "t_max": args.scheduler_t_max,
                "eta_min": args.scheduler_eta_min,
                "patience": args.scheduler_patience,
                "threshold": args.scheduler_threshold,
                "max_lr": args.scheduler_max_lr,
                "steps_per_epoch": args.scheduler_steps_per_epoch,
                "total_steps": args.scheduler_total_steps,
            },
        }

    # --------------------------------------------------------
    # Result folder
    # --------------------------------------------------------
    save_dir = os.path.join(
        args.result_root,
        f"{args.dataset}_{args.model}{run_tag(args, train_loader.dataset)}",
    )
    print("Save dir    :", save_dir)
    os.makedirs(
        save_dir,
        exist_ok=True,
    )
    # --------------------------------------------------------
    # Count and save model complexity before training
    # --------------------------------------------------------
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    complexity = measure_model_complexity(
        model=model,
        input_shape=input_shape,
    )

    def format_count(value):
        if value is None:
            return "N/A"
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.3f}G"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.3f}M"
        if value >= 1_000:
            return f"{value / 1_000:.3f}K"
        return str(value)

    print("Model Complexity")
    print(f"Total Params     : {format_count(total_params)}")
    print(f"Trainable Params : {format_count(trainable_params)}")
    print(f"MACs / sample    : {format_count(complexity['macs_per_sample'])}")
    print(f"FLOPs / sample   : {format_count(complexity['flops_per_sample'])}")

    save_model_complexity(
        save_dir=save_dir,
        dataset=args.dataset,
        model=args.model,
        total_params=total_params,
        trainable_params=trainable_params,
        complexity=complexity,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    start_time = time.time()
    history, results = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=save_dir,
        scheduler_cfg=scheduler_cfg,
    )
    training_time = time.time() - start_time
    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    # One full result folder per checkpoint-selection rule:
    #   save_dir/last_epoch/  -> reported numbers (no selection on test)
    #   save_dir/best_test/   -> oracle (test-selected), analysis only
    for selection, r in results.items():
        print(f"\n>>> {selection} (epoch {r['epoch']}) -> {r['dir']}")
        evaluate_model(
            history=history,
            y_true=r["y_true"],
            y_pred=r["y_pred"],
            save_dir=r["dir"],
            dataset=args.dataset,
            model=args.model,
            best_epoch=r["epoch"],
            train_size=train_size,
            test_size=test_size,
            training_time=training_time,
            total_params=total_params,
            trainable_params=trainable_params,
            complexity=complexity,
            selection=selection,
            total_epochs=args.epochs,
        )

    print()
    print("=" * 60)
    print("Finished")
    print("Result Folder")
    print(save_dir)
    print("=" * 60)

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    main()

# python main.py --data_root=".\data\" --dataset="ut_har" --model="mlp" --case_split=0