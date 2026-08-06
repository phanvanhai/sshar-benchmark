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
from metrics import evaluate_model

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
            "sshar_esp",
            "sshar_nexmon",
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
            "cnn_gru",
            "bimamba",
        ],
        help="Model name",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root directory",
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
        "--device",
        type=str,
        default="cuda",
        choices=[
            "cpu",
            "cuda",
        ],
    )

    return parser.parse_args()


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
    train_loader, test_loader, info = load_dataset(
        name=args.dataset,
        root_dir=args.data_root,
        batch_size=args.batch_size,
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
    )

    # --------------------------------------------------------
    # Result folder
    # --------------------------------------------------------
    save_dir = os.path.join(
        args.result_root,
        f"{args.dataset}_{args.model}",
    )
    os.makedirs(
        save_dir,
        exist_ok=True,
    )
    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    start_time = time.time()
    history, y_true, y_pred, best_epoch, best_acc = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=save_dir,
    )
    training_time = time.time() - start_time
    # --------------------------------------------------------
    # Count parameters
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
    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    evaluate_model(
        history=history,
        y_true=y_true,
        y_pred=y_pred,
        save_dir=save_dir,
        dataset=args.dataset,
        model=args.model,
        best_epoch=best_epoch,
        train_size=train_size,
        test_size=test_size,
        training_time=training_time,
        total_params=total_params,
        trainable_params=trainable_params,
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

# python main.py --dataset="ut_har" --model="mlp" --data_root=".\data\" --device="cpu"