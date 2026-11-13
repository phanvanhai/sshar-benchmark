"""
train.py

Simple training utilities
"""

import inspect
import os
import time
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score

# Sub-folders of save_dir, one per checkpoint-selection rule.
#   last_epoch : model after the final epoch (no selection; reported protocol)
#   best_test  : epoch with the highest test accuracy (oracle, for analysis only)
SELECTIONS = ("last_epoch", "best_test")


def create_scheduler(optimizer, scheduler_cfg=None):
    """Create a learning-rate scheduler from a config dict."""
    if scheduler_cfg is None:
        return None

    scheduler_name = scheduler_cfg.get("name")
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None

    scheduler_args = scheduler_cfg.get("args", {}) or {}
    scheduler_args = {
        key: value
        for key, value in scheduler_args.items()
        if value is not None
    }

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "steplr":
        return lr_scheduler.StepLR(optimizer, **scheduler_args)

    if scheduler_name == "multisteplr":
        milestones = scheduler_args.get("milestones")
        if isinstance(milestones, str):
            scheduler_args["milestones"] = [
                int(x)
                for x in milestones.split(",")
                if x.strip()
            ]
        return lr_scheduler.MultiStepLR(optimizer, **scheduler_args)

    if scheduler_name == "cosineannealinglr":
        return lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)

    if scheduler_name == "reducelronplateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    if scheduler_name == "onecyclelr":
        return lr_scheduler.OneCycleLR(optimizer, **scheduler_args)

    raise ValueError(
        f"Unsupported scheduler: {scheduler_name}."
    )


# ============================================================
# Train one epoch
# ============================================================
def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """
    Train one epoch.

    Returns
    -------
    train_loss
    train_acc
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / len(loader)
    train_acc = correct / total

    return train_loss, train_acc

# ============================================================
# Evaluate
# ============================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    """
    Evaluate model.

    Returns
    -------
    loss
    accuracy
    y_true
    y_pred
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        running_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        y_true.extend(
            y.cpu().numpy()
        )

        y_pred.extend(
            pred.cpu().numpy()
        )

    test_loss = running_loss / len(loader)
    test_acc = correct / total

    return (
        test_loss,
        test_acc,
        y_true,
        y_pred,
    )

# ============================================================
# Train
# ============================================================

def train(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    device,
    save_dir,
    scheduler_cfg=None,
):
    """
    Train model.

    Parameters
    ----------
    scheduler_cfg : dict or None
        Optional scheduler configuration dict with keys:
        - name: scheduler type
        - args: scheduler arguments

    Saves
    -----
    save_dir/history.csv
    save_dir/last_epoch/model.pth   model after the final epoch (reported)
    save_dir/best_test/model.pth    epoch with highest test accuracy (oracle)

    Returns
    -------
    history : DataFrame
    results : {"last_epoch"|"best_test": {"epoch", "y_true", "y_pred", "dir"}}
    """

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    scheduler = create_scheduler(optimizer, scheduler_cfg)

    history = []

    ckpt_dirs = {name: os.path.join(save_dir, name) for name in SELECTIONS}
    for d in ckpt_dirs.values():
        os.makedirs(d, exist_ok=True)

    best_acc = -1.0
    best = None  # {"epoch", "y_true", "y_pred"} at the best test accuracy

    start_time = time.time()
    # --------------------------------------------------------
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc, y_true, y_pred = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        test_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1_macro": test_f1,
        })

        print(
            f"[{epoch:03d}/{epochs}] "
            f"Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.4f}  "
            f"Test Loss: {test_loss:.4f}  "
            f"Test Acc: {test_acc:.4f}  "
            f"Test F1: {test_f1:.4f}"
        )

        # ----------------------------------------------------
        # best_test (oracle): highest test accuracy so far
        # ----------------------------------------------------
        if test_acc > best_acc:
            best_acc = test_acc
            best = {"epoch": epoch, "y_true": list(y_true), "y_pred": list(y_pred)}
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dirs["best_test"], "model.pth"),
            )

        if scheduler is not None:
            if isinstance(
                scheduler,
                lr_scheduler.ReduceLROnPlateau,
            ):
                scheduler.step(test_loss)
            else:
                scheduler.step()

    # --------------------------------------------------------
    # last_epoch: model after the final epoch
    # --------------------------------------------------------
    torch.save(
        model.state_dict(),
        os.path.join(ckpt_dirs["last_epoch"], "model.pth"),
    )
    last = {"epoch": epochs, "y_true": list(y_true), "y_pred": list(y_pred)}

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------
    elapsed = time.time() - start_time

    print("\nTraining Finished")
    print(f"Last Epoch : {epochs}  Acc: {history[-1]['test_acc']:.4f}  (reported)")
    print(f"Best Epoch : {best['epoch']}  Acc: {best_acc:.4f}  (oracle, test-selected)")
    print(f"Time       : {elapsed:.1f} sec")

    # --------------------------------------------------------
    # Save history (shared by both selections)
    # --------------------------------------------------------
    history = pd.DataFrame(history)
    history.to_csv(
        os.path.join(
            save_dir,
            "history.csv",
        ),
        index=False,
    )

    # results[selection] = {"epoch", "y_true", "y_pred", "dir"}
    results = {"last_epoch": last, "best_test": best}
    for name, r in results.items():
        r["dir"] = ckpt_dirs[name]

    return history, results

# ============================================================
# V2: train / val / test (doc/experiment_plan_v2.md)
# ============================================================
# Sub-folders of save_dir, one per checkpoint:
#   best_val   : epoch with the highest validation accuracy
#   last_epoch : model after the final epoch
SELECTIONS_V2 = ("best_val", "last_epoch")


def train_v2(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    lr,
    device,
    save_dir,
):
    """
    Train with Adam + cross-entropy (no scheduler), select checkpoints on VAL only,
    then evaluate the TEST set once per checkpoint after training.

    Saves
    -----
    save_dir/history.csv              train/val loss, accuracy, macro-F1 per epoch
    save_dir/best_val/model.pth
    save_dir/last_epoch/model.pth

    Returns
    -------
    history : DataFrame
    results : {"best_val"|"last_epoch": {"epoch", "val_acc", "y_true", "y_pred", "dir"}}
    elapsed : training time (s)
    """
    import copy

    os.makedirs(save_dir, exist_ok=True)
    ckpt_dirs = {name: os.path.join(save_dir, name) for name in SELECTIONS_V2}
    for d in ckpt_dirs.values():
        os.makedirs(d, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_acc = -1.0
    best_epoch = None
    best_state = None

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model, loader=train_loader, criterion=criterion,
            optimizer=optimizer, device=device,
        )
        val_loss, val_acc, vy_true, vy_pred = evaluate(
            model=model, loader=val_loader, criterion=criterion, device=device,
        )
        val_f1 = f1_score(vy_true, vy_pred, average="macro", zero_division=0)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1,
        })
        print(
            f"[{epoch:03d}/{epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  Val F1: {val_f1:.4f}"
        )

        # best_val: first epoch reaching the highest validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(ckpt_dirs["best_val"], "model.pth"))

    elapsed = time.time() - start_time
    last_state = copy.deepcopy(model.state_dict())
    torch.save(last_state, os.path.join(ckpt_dirs["last_epoch"], "model.pth"))

    history = pd.DataFrame(history)
    history.to_csv(os.path.join(save_dir, "history.csv"), index=False)

    # --------------------------------------------------------
    # Test: evaluated once per checkpoint, after training
    # --------------------------------------------------------
    results = {}
    for name, state, epoch, vacc in (
        ("best_val", best_state, best_epoch, best_val_acc),
        ("last_epoch", last_state, epochs, float(history["val_acc"].iloc[-1])),
    ):
        model.load_state_dict(state)
        _, test_acc, y_true, y_pred = evaluate(
            model=model, loader=test_loader, criterion=criterion, device=device,
        )
        results[name] = {
            "epoch": epoch,
            "val_acc": vacc,
            "y_true": list(y_true),
            "y_pred": list(y_pred),
            "dir": ckpt_dirs[name],
        }
        print(f"{name:10s}: epoch {epoch:3d}  val acc {vacc:.4f}  test acc {test_acc:.4f}")

    print(f"Training time: {elapsed:.1f} sec")
    return history, results, elapsed
