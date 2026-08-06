"""
datasets.py

Unified dataset loader for WiFi HAR Benchmark

Supported datasets
------------------
1. UT_HAR
2. SSHAR_ESP
3. SSHAR_Nexmon

Author:
"""

from pathlib import Path
import os
import glob
import re

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# ============================================================
# Default configuration
# ============================================================

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_PIN_MEMORY = True


# ============================================================
# Utility
# ============================================================

def scan_folders(root: Path, prefix: str):
    """
    Scan folders with given prefix.

    Example
    -------
    scan_folders(data_root, "room")
        -> ["room_01","room_02"]

    scan_folders(room_path, "subject")
        -> ["subject_01",...]
    """

    if not root.exists():
        return []

    folders = []

    for item in sorted(root.iterdir()):

        if item.is_dir() and item.name.startswith(prefix):
            folders.append(item.name)

    return folders


# ============================================================
#                    UT_HAR
# ============================================================

class UTHARDataset(Dataset):
    """
    UT_HAR Dataset

    Folder structure

    UT_HAR
    ├── data
    │     train_data.csv
    │     val_data.csv
    │     test_data.csv
    │
    └── label
          train_label.csv
          val_label.csv
          test_label.csv

    Parameters
    ----------
    root_dir

    split
        train
        val
        test

    normalize

    norm_type
        minmax
        zscore

    mean,std
        only used for zscore
    """

    def __init__(
        self,
        root_dir,
        split="train",
        normalize=True,
        norm_type="minmax",
        mean=None,
        std=None,
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.normalize = normalize
        self.norm_type = norm_type
        self.mean = mean
        self.std = std

        self.X = None
        self.y = None

        self.load_data()

    # --------------------------------------------------------

    def load_data(self):
        data_file = (
            self.root_dir /
            "UT_HAR" /
            "data" /
            f"X_{self.split}.csv"
        )

        label_file = (
            self.root_dir /
            "UT_HAR" /
            "label" /
            f"y_{self.split}.csv"
        )
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        if not label_file.exists():
            raise FileNotFoundError(label_file)

        with open(data_file, "rb") as f:
            x = np.load(f)

        # Raw data shape: (N, 250, 90) = (N, time, subcarrier)
        # Transpose to (N, 90, 250) = (N, subcarrier, time)
        # to match model input format (C, T)
        x = x.transpose(0, 2, 1)

        with open(label_file, "rb") as f:
            y = np.load(f)

        # one-hot -> class index
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)

        self.X = x.astype(np.float32)
        self.y = y.astype(np.int64)

    # --------------------------------------------------------
    def __len__(self):
        return len(self.X)
    
    # --------------------------------------------------------
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.normalize:
            if self.norm_type == "minmax":
                xmin = x.min()
                xmax = x.max()
                x = (
                    x - xmin
                ) / (
                    xmax - xmin + 1e-8
                )
            elif self.norm_type == "zscore":
                if self.mean is None or self.std is None:
                    raise ValueError(
                        "mean/std required for zscore"
                    )
                x = (
                    x - self.mean
                ) / (
                    self.std + 1e-8
                )
            else:
                raise ValueError(
                    f"Unknown norm_type {self.norm_type}"
                )

        x = torch.from_numpy(x).float()
        y = int(self.y[idx])

        return x, y


# ============================================================
# compute mean/std for UT_HAR
# ============================================================
def compute_ut_har_mean_std(root_dir):
    """
    Compute global mean/std
    using TRAIN split only.

    Only used when norm_type='zscore'
    """

    dataset = UTHARDataset(
        root_dir=root_dir,
        split="train",
        normalize=False,
    )

    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    for x, _ in dataset:
        x = x.numpy()
        total_sum += x.sum()
        total_sq += np.square(x).sum()
        total_count += x.size

    mean = total_sum / total_count

    std = np.sqrt(
        total_sq / total_count - mean ** 2
    )

    print(f"UT_HAR mean = {mean:.6f}")
    print(f"UT_HAR std  = {std:.6f}")

    return mean, std


# ============================================================
#               Base SSHAR Dataset
# ============================================================
class _SSHARDatasetBase(Dataset):
    """
    Internal SSHAR dataset.

    Do not use directly.

    Use

        SSHAR_ESP_Dataset

    or

        SSHAR_Nexmon_Dataset
    """

    pattern = re.compile(
        r"act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)"
    )
    # --------------------------------------------------------

    def __init__(
        self,
        root_dir,
        device,
        signal="amp",
        split="train",
        normalize=True,
        norm_type="zscore",
        mean=None,
        std=None,
        rooms=None,
        subjects=None,
        rx_list=None,
    ):

        self.root_dir = Path(root_dir)
        self.device = device
        self.signal = signal
        self.split = split
        self.normalize = normalize
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        # ----------------------------------------
        if rooms is None:
            rooms = scan_folders(
                self.root_dir,
                "room",
            )

        self.rooms = rooms

        # ----------------------------------------
        if rx_list is None:
            rx_list = [
                "rx_00",
                "rx_01",
                "rx_02",
            ]
        self.rx_list = rx_list

        # ----------------------------------------
        if subjects is None:
            subjects = []
            for room in self.rooms:
                base = (
                    self.root_dir /
                    room /
                    self.device /
                    self.rx_list[0]
                )

                subjects.extend(
                    scan_folders(
                        base,
                        "subject",
                    )
                )

            subjects = sorted(
                list(
                    set(subjects)
                )
            )

        self.subjects = subjects

        # ----------------------------------------

        self.samples = []
        self.build_index()

    # --------------------------------------------------------
    def build_index(self):

        """
        Build sample index.

        Only scan RX00.

        Store all RX file paths.

        samples[i] =

        {
            "label":0,
            "files":[
                rx00,
                rx01,
                rx02
            ]
        }
        """
        self.samples.clear()

        for room in self.rooms:
            for subject in self.subjects:
                folder = (
                    self.root_dir /
                    room /
                    self.device /
                    self.rx_list[0] /
                    subject
                )

                if not folder.exists():
                    continue

                for file in sorted(
                    folder.iterdir()
                ):
                    if not file.name.startswith(
                        self.signal
                    ):
                        continue

                    m = self.pattern.search(
                        file.name
                    )

                    if m is None:
                        continue

                    act = int(m.group(1))
                    pos = int(m.group(2))
                    direction = int(m.group(3))
                    rep = int(m.group(4))
                    train = (
                        rep <= 8
                        if direction == 0
                        else rep <= 4
                    )

                    if (
                        self.split == "train"
                        and not train
                    ):
                        continue

                    if (
                        self.split == "test"
                        and train
                    ):
                        continue

                    rx_files = []

                    for rx in self.rx_list:
                        rx_files.append(
                            self.root_dir
                            / room
                            / self.device
                            / rx
                            / subject
                            / file.name
                        )

                    self.samples.append(
                        {
                            "label": act - 1,
                            "room": room,
                            "subject": subject,
                            "files": rx_files,
                        }
                    )

    # --------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]
        rx_data = []

        for file in sample["files"]:
            x = np.load(file).astype(np.float32)
            rx_data.append(x)

        # --------------------------------------------------
        # Stack RX
        #
        # ESP:
        #   (3,1,56,1000)
        #
        # Nexmon:
        #   (3,4,56,1000)
        # --------------------------------------------------
        x = np.stack(rx_data)

        # --------------------------------------------------
        # Merge RX + antenna
        #
        # ESP:
        #   (168,1000)
        #
        # Nexmon:
        #   (672,1000)
        # --------------------------------------------------
        x = x.reshape(-1, x.shape[-1])

        # --------------------------------------------------
        # Normalization
        # --------------------------------------------------
        if self.normalize:
            if self.norm_type == "zscore":
                if self.mean is None or self.std is None:
                    raise ValueError(
                        "mean/std required for zscore"
                    )
                x = (
                    x - self.mean
                ) / (
                    self.std + 1e-8
                )
            elif self.norm_type == "minmax":
                xmin = x.min()
                xmax = x.max()
                x = (
                    x - xmin
                ) / (
                    xmax - xmin + 1e-8
                )
            else:
                raise ValueError(
                    f"Unknown norm_type {self.norm_type}"
                )

        x = torch.from_numpy(x).float()
        # --------------------------------------------------
        # Temporal pooling
        #
        # 1000 -> 250
        # --------------------------------------------------
        x = F.adaptive_max_pool1d(
            x,
            output_size=250,
        )

        y = sample["label"]
        return x, y


# ============================================================
# SSHAR ESP
# ============================================================

class SSHAR_ESP_Dataset(_SSHARDatasetBase):
    def __init__(
        self,
        root_dir,
        signal="amp",
        split="train",
        normalize=True,
        norm_type="zscore",
        mean=None,
        std=None,
        rooms=None,
        subjects=None,
        rx_list=None,
    ):
        super().__init__(
            root_dir=root_dir,
            device="esp",
            signal=signal,
            split=split,
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
            rooms=rooms,
            subjects=subjects,
            rx_list=rx_list,
        )

# ============================================================
# SSHAR Nexmon
# ============================================================
class SSHAR_Nexmon_Dataset(_SSHARDatasetBase):
    def __init__(
        self,
        root_dir,
        signal="amp",
        split="train",
        normalize=True,
        norm_type="zscore",
        mean=None,
        std=None,
        rooms=None,
        subjects=None,
        rx_list=None,
    ):
        super().__init__(
            root_dir=root_dir,
            device="nexmon",
            signal=signal,
            split=split,
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
            rooms=rooms,
            subjects=subjects,
            rx_list=rx_list,
        )

# ============================================================
# Compute SSHAR mean/std
# ============================================================
def compute_sshar_mean_std(dataset):
    """
    Parameters
    ----------
    dataset

        SSHAR_ESP_Dataset
        or
        SSHAR_Nexmon_Dataset

        normalize=False
    """

    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    for x, _ in dataset:
        x = x.numpy()
        total_sum += x.sum()
        total_sq += np.square(x).sum()
        total_count += x.size

    mean = total_sum / total_count
    std = np.sqrt(
        total_sq / total_count - mean ** 2
    )

    print(f"SSHAR mean = {mean:.6f}")
    print(f"SSHAR std  = {std:.6f}")

    return mean, std


# ============================================================
# DataLoader
# ============================================================

def create_dataloader(
    train_dataset,
    test_dataset,
    batch_size=DEFAULT_BATCH_SIZE,
    num_workers=DEFAULT_NUM_WORKERS,
    pin_memory=DEFAULT_PIN_MEMORY,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


# ============================================================
# Unified Loader
# ============================================================
def load_dataset(
    name,
    root_dir,
    batch_size=DEFAULT_BATCH_SIZE,
    normalize=True,
    norm_type=None,
):

    name = name.lower()
    # =======================================================
    # UT_HAR
    # =======================================================
    if name == "ut_har":
        if norm_type is None:
            norm_type = "minmax"

        mean = std = None
        if normalize and norm_type == "zscore":
            mean, std = compute_ut_har_mean_std(
                root_dir
            )

        train_dataset = UTHARDataset(
            root_dir=root_dir,
            split="train",
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
        )

        test_dataset = UTHARDataset(
            root_dir=root_dir,
            split="test",
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
        )

    # =======================================================
    # SSHAR ESP
    # =======================================================
    elif name == "sshar_esp":
        if norm_type is None:
            norm_type = "zscore"

        mean = std = None
        if normalize and norm_type == "zscore":
            tmp = SSHAR_ESP_Dataset(
                root_dir=root_dir,
                split="train",
                normalize=False,
            )
            mean, std = compute_sshar_mean_std(tmp)

        train_dataset = SSHAR_ESP_Dataset(
            root_dir=root_dir,
            split="train",
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
        )

        test_dataset = SSHAR_ESP_Dataset(
            root_dir=root_dir,
            split="test",
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
        )

    # =======================================================
    # SSHAR Nexmon
    # =======================================================
    elif name == "sshar_nexmon":
        if norm_type is None:
            norm_type = "zscore"

        mean = std = None
        if normalize and norm_type == "zscore":
            tmp = SSHAR_Nexmon_Dataset(
                root_dir=root_dir,
                split="train",
                normalize=False,
            )
            mean, std = compute_sshar_mean_std(tmp)

        train_dataset = SSHAR_Nexmon_Dataset(
            root_dir=root_dir,
            split="train",
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
        )

        test_dataset = SSHAR_Nexmon_Dataset(
            root_dir=root_dir,
            split="test",
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
        )
    else:

        raise ValueError(
            f"Unknown dataset: {name}"
        )

    # =======================================================

    train_loader, test_loader = create_dataloader(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
    )

    x, y = train_dataset[0]

    info = {
        "dataset": name,
        "input_shape": tuple(x.shape),
        "num_classes": len(
            set(train_dataset.y)
        ) if isinstance(train_dataset, UTHARDataset)
        else len(
            set(s["label"] for s in train_dataset.samples)
        ),
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
    }

    return (
        train_loader,
        test_loader,
        info,
    )

