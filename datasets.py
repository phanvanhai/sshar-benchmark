"""
datasets.py

Unified dataset loader for WiFi HAR Benchmark

Supported datasets
------------------
1. UT_HAR
2. SSHAR_ESP
3. SSHAR_asus
4. XRF55
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


# Default split mode when none is passed explicitly (main.py --split overrides it).
#   1 -> "repeat": first repetitions train, last repetitions test
#   0 -> "user"  : USER_CASES["sshar"][case_split] subjects train, the others test
SPLIT_BY_REPEAT  = 0
SPLIT_MODES = ("repeat", "user")


def default_split_mode():
    return "repeat" if SPLIT_BY_REPEAT else "user"


# Normalization types (norm_type):
#   "zscore"        : global z-score, one mean/std from the pooled TRAIN split
#   "zscore_sample" : per-sample, per-receiver z-score: each RX of each sample uses its
#                     own mean/std over all its antennas, subcarriers and time steps
#   "zscore_sample_all" : per-sample z-score with ONE mean/std over the whole sample
#                     (all RX together)
#   "minmax"        : per-sample min-max to [0, 1]
# No dataset statistics are used by the per-sample types.
# All normalizations are applied BEFORE time pooling (as in the official BiMamba code).
NORM_TYPES = ("zscore", "zscore_sample", "zscore_sample_all", "minmax")

# XRF55 rows (270) = 3 receivers x 3 antennas x 30 subcarriers, receiver-major.
XRF55_NUM_RX = 3


def zscore_per_sample(x, eps=1e-8):
    """Z-score one sample (numpy array) with its own mean/std over all elements."""
    return (x - x.mean()) / (x.std() + eps)


def zscore_per_sample_rx(x, n_rx, eps=1e-8):
    """
    Z-score one sample separately for each receiver.

    x must be receiver-major along its first axis (e.g. (rx*ant*sub, T) or
    (rx*ant, sub, T)), so that x.reshape(n_rx, -1) gives one row per receiver.
    """
    shape = x.shape
    if shape[0] % n_rx != 0:
        raise ValueError(f"first axis {shape[0]} not divisible by n_rx={n_rx}")
    g = x.reshape(n_rx, -1)
    g = (g - g.mean(axis=1, keepdims=True)) / (g.std(axis=1, keepdims=True) + eps)
    return g.reshape(shape)

# ============================================================
# Default configuration
# ============================================================

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_PIN_MEMORY = True

# Keep only the requested classes in the benchmark.
# Labels are remapped to contiguous indices 0..5 for PyTorch.
DEFAULT_SELECTED_LABELS = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_LABEL_MAP = {
    label: idx for idx, label in enumerate(DEFAULT_SELECTED_LABELS)
}

INTEL_30_SUBCARRIERS = [
    -28, -26, -24, -22, -20, -18, -16, -14,
    -12, -10, -8, -6, -4, -2,
    -1, 1, 3, 5, 7, 9, 11, 13,
    15, 17, 19, 21, 23, 25, 27, 28
]

# Map signed subcarrier -> index 0..55
INTEL_30_INDICES = [
    sc + 28 if sc < 0 else sc + 27
    for sc in INTEL_30_SUBCARRIERS
]

USER_CASES = {
    "xrf55": [
        list(range(1, 25)),
        [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 21, 23, 24, 25, 27, 28, 29, 30],
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29],
        [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29],
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 29, 30]
    ],
    "sshar": [
        [1, 2, 3, 4, 9, 10, 11, 12],
        [1, 3, 4, 9, 11, 12, 13, 14],
        [1, 2, 4, 9, 10, 12, 13, 14],
        [2, 3, 4, 10, 11, 12, 13, 14],
        [1, 2, 4, 9, 10, 11, 13, 14],
    ]
}
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


def resolve_room_dirs(root_dirs, rooms=None):
    """
    Collect room folders from one or several dataset roots.

    root_dirs : str | Path | list of them. Each root contains room_XX folders.
    rooms     : None (all rooms found) or list of room folder names, e.g. ["room_01", "room_02"].

    Returns a list of (room_name, room_path), sorted by room name.
    """
    if isinstance(root_dirs, (str, Path)):
        root_dirs = [root_dirs]

    found = {}
    for root in root_dirs:
        root = Path(root)
        for name in scan_folders(root, "room"):
            if name in found:
                raise ValueError(
                    f"Room '{name}' found in several roots: {found[name]} and {root / name}"
                )
            found[name] = root / name

    if not found:
        raise ValueError(f"No room_* folder found under: {[str(r) for r in root_dirs]}")

    if rooms is None:
        rooms = sorted(found)
    missing = [r for r in rooms if r not in found]
    if missing:
        raise ValueError(f"Rooms {missing} not found. Available: {sorted(found)}")

    return [(r, found[r]) for r in sorted(rooms)]


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

        SSHAR_ASUS_Dataset
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
        case_split = 0,
        shape_option="2d",
        normalize=True,
        norm_type="zscore",
        mean=None,
        std=None,
        target_time=None,
        rooms=None,
        subjects=None,
        rx_list=None,
        intel_shape=False,
        allowed_labels=None,
        split_mode=None,
    ):

        self.root_dir = root_dir
        self.device = device
        self.signal = signal
        self.split = split
        self.case_split = case_split
        self.shape_option = shape_option
        self.normalize = normalize
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.target_time = target_time
        self.intel_shape = intel_shape
        self.allowed_labels = tuple(DEFAULT_SELECTED_LABELS) if allowed_labels is None else tuple(allowed_labels)
        self.label_map = {label: idx for idx, label in enumerate(self.allowed_labels)}
        self.split_mode = default_split_mode() if split_mode is None else split_mode
        if self.split_mode not in SPLIT_MODES:
            raise ValueError(f"Unknown split_mode {self.split_mode}, expected {SPLIT_MODES}")
        # ----------------------------------------
        # list of (room_name, room_path); roots may be several folders
        self.room_dirs = resolve_room_dirs(root_dir, rooms)
        self.rooms = [name for name, _ in self.room_dirs]

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
            for _, room_path in self.room_dirs:
                base = (
                    room_path /
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
        for room, room_path in self.room_dirs:
            for subject in self.subjects:
                folder = (
                    room_path /
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

                    if act not in self.label_map:
                        continue

                    
                    if self.split_mode == "repeat":
                        # Chia theo so lan lap
                        is_train = (rep <= 8 if direction == 0 else rep <= 4)
                    else:
                        # Chia theo nguoi (subject id is global across rooms)
                        user_num = int(subject.split("_")[1])
                        is_train = user_num in USER_CASES["sshar"][self.case_split]

                    if (self.split == "train" and not is_train):
                        continue
                    if (self.split == "test" and is_train ):
                        continue

                    rx_files = []

                    for rx in self.rx_list:
                        rx_files.append(
                            room_path
                            / self.device
                            / rx
                            / subject
                            / file.name
                        )

                    self.samples.append(
                        {
                            "label": self.label_map[act],
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
        #   (3, 1, 56, T)
        #
        # ASUS:
        #   (3, 4, 56, T)
        # --------------------------------------------------
        x = np.stack(rx_data)

        # ============================================================
        # Intel shape
        # ============================================================
        if self.intel_shape:
            # Select antennas
            if self.device == "esp":
                # ESP32: 1 antenna
                antenna_indices = [0]
            elif self.device == "asus":
                # ASUS: keep antenna 0, 1, 3
                antenna_indices = [0,1,2,3]
            else:
                raise ValueError(
                    f"Unknown device: {self.device}"
                )
            x = x[:, antenna_indices, :, :]
            # Select 30 Intel-like subcarriers
            # x = x[:, :, INTEL_30_INDICES, :]

        # --------------------------------------------------
        # Shape
        # --------------------------------------------------
        if self.shape_option == "2d":
            # (rx, antenna, sub, time)
            # -> (rx * antenna * sub, time)
            x = x.reshape(
                -1,
                x.shape[-1]
            )
        elif self.shape_option == "3d":
            # (rx, antenna, sub, time)
            # -> (rx * antenna, sub, time)
            rx, antenna, sub, time_len = x.shape
            x = x.reshape(
                rx * antenna,
                sub,
                time_len
            )
        else:
            raise ValueError(
                f"Unknown shape_option: {self.shape_option}"
            )

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
            elif self.norm_type == "zscore_sample":
                # x is RX-major after the reshape above
                x = zscore_per_sample_rx(x, n_rx=len(sample["files"]))
            elif self.norm_type == "zscore_sample_all":
                x = zscore_per_sample(x)
            else:
                raise ValueError(
                    f"Unknown norm_type {self.norm_type}"
                )

        x = torch.from_numpy(x).float()

        if self.target_time is not None and x.ndim in (2, 3) and x.shape[-1] != self.target_time:
            x = F.adaptive_max_pool1d(x, output_size=self.target_time)

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
        case_split = 0,
        shape_option="2d",
        normalize=True,
        norm_type="zscore",
        mean=None,
        std=None,
        target_time=None,
        rooms=None,
        subjects=None,
        rx_list=None,
        intel_shape=False,
        allowed_labels=None,
        split_mode=None,
    ):
        super().__init__(
            root_dir=root_dir,
            device="esp",
            signal=signal,
            split=split,
            case_split = case_split,
            shape_option=shape_option,
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
            target_time=target_time,
            rooms=rooms,
            subjects=subjects,
            rx_list=rx_list,
            intel_shape=intel_shape,
            allowed_labels=allowed_labels,
            split_mode=split_mode,
        )

# ============================================================
# SSHAR asus
# ============================================================
class SSHAR_ASUS_Dataset(_SSHARDatasetBase):
    def __init__(
        self,
        root_dir,
        signal="amp",
        split="train",
        case_split = 0,
        shape_option="2d",
        normalize=True,
        norm_type="zscore",
        mean=None,
        std=None,
        target_time=None,
        rooms=None,
        subjects=None,
        rx_list=None,
        intel_shape=False,
        allowed_labels=None,
        split_mode=None,
    ):
        super().__init__(
            root_dir=root_dir,
            device="asus",
            signal=signal,
            split=split,
            case_split = case_split,
            shape_option=shape_option,
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
            target_time=target_time,
            rooms=rooms,
            subjects=subjects,
            rx_list=rx_list,
            intel_shape=intel_shape,
            allowed_labels=allowed_labels,
            split_mode=split_mode,
        )

# ============================================================
# SSHAR split summary
# ============================================================
def print_sshar_split_summary(train_dataset, test_dataset):
    """Print rooms / split mode and per-room sample + subject counts."""
    print(f"SSHAR device={train_dataset.device} split_mode={train_dataset.split_mode} "
          f"rooms={train_dataset.rooms}")
    for room, room_path in train_dataset.room_dirs:
        for tag, ds in (("train", train_dataset), ("test", test_dataset)):
            subs = sorted({x["subject"] for x in ds.samples if x["room"] == room})
            n = sum(1 for x in ds.samples if x["room"] == room)
            ids = [int(x.split("_")[1]) for x in subs]
            print(f"  {room} ({room_path}) {tag:5s}: {n:5d} samples, subjects {ids}")


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
        SSHAR_ASUS_Dataset

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
# XRF55 Dataset (Subset)
# ============================================================
class XRF55Dataset(Dataset):
    """
    XRF55 Dataset (Subset for 8 actions and all users)

    Format requirements:
    - Files are in .npy format.
    - Filenames follow the pattern a_b_c.npy
      a = User ID (2 digits)
      b = Action ID (2 digits) -> will be zero-indexed for PyTorch
      c = Repetition number

    Split rule is determined by a maximum repetition number for training.
    """
    def __init__(
        self,
        root_dir,
        split="train",
        case_split=0,
        shape_option="2d",
        num_sub=30,  # XRF55 typically uses Intel 5300 with 30 subcarriers per antenna
        normalize=True,
        norm_type="minmax",
        mean=None,
        std=None,
        target_time=None,
        allowed_labels=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.case_split = case_split
        self.shape_option = shape_option
        self.num_sub = num_sub
        self.normalize = normalize
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.target_time = target_time
        self.allowed_labels = tuple(DEFAULT_SELECTED_LABELS) if allowed_labels is None else tuple(allowed_labels)
        self.label_map = {label: idx for idx, label in enumerate(self.allowed_labels)}

        self.samples = []
        self.build_index()

    def build_index(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        for file_path in self.root_dir.glob("*.npy"):
            name_parts = file_path.stem.split('_')
            
            # Ensure it matches a_b_c format
            if len(name_parts) == 3:
                user_id, action_id, rep_id = name_parts
                rep_num = int(rep_id)
                
                action_id_int = int(action_id)
                if action_id_int not in self.label_map:
                    continue

                if SPLIT_BY_REPEAT:
                    # Chia theo so lan lap                
                    is_train = rep_num <= 16
                else:
                    # Chia theo nguoi
                    user_num = int(user_id)
                    is_train = user_num in USER_CASES["xrf55"][self.case_split]
                

                if self.split == "train" and not is_train:
                    continue
                if self.split == "test" and is_train:
                    continue

                # Remap selected labels to contiguous indices 0..5.
                label = self.label_map[action_id_int]

                self.samples.append({
                    "path": str(file_path),
                    "label": label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = np.load(sample["path"]).astype(np.float32)
        y = sample["label"]
        
        # Reshape logically based on shape_option
        if self.shape_option == "3d":
            # Assuming raw input is 2D: (C, Time) where C = dev_anten * num_sub
            if len(x.shape) == 2:
                c, t = x.shape
                assert c % self.num_sub == 0
                dev_anten = c // self.num_sub
                x = x.reshape(dev_anten, self.num_sub, t)
        elif self.shape_option == "2d":
            # If input is already 3D: (dev_anten, sub, Time), flatten it back to 2D
            if len(x.shape) == 3:
                d, s, t = x.shape
                x = x.reshape(d * s, t)

        if self.normalize:
            if self.norm_type == "minmax":
                xmin = x.min()
                xmax = x.max()
                x = (x - xmin) / (xmax - xmin + 1e-8)
            elif self.norm_type == "zscore":
                if self.mean is None or self.std is None:
                    raise ValueError("mean/std required for zscore")
                x = (x - self.mean) / (self.std + 1e-8)
            elif self.norm_type == "zscore_sample":
                x = zscore_per_sample_rx(x, n_rx=XRF55_NUM_RX)
            elif self.norm_type == "zscore_sample_all":
                x = zscore_per_sample(x)
            else:
                raise ValueError(f"Unknown norm_type {self.norm_type}")

        x = torch.from_numpy(x).float()

        if self.target_time is not None and x.ndim in (2, 3) and x.shape[-1] != self.target_time:
            x = F.adaptive_max_pool1d(x, output_size=self.target_time)

        return x, y


# ============================================================
# Compute XRF55 mean/std
# ============================================================
def compute_xrf55_mean_std(root_dir, case_split, shape_option="2d", num_sub=30, target_time=None):
    """
    Compute global mean/std for XRF55 using TRAIN split only.
    """
    dataset = XRF55Dataset(
        root_dir=root_dir,
        split="train",
        case_split=case_split,
        shape_option=shape_option,
        num_sub=num_sub,
        normalize=False,
        target_time=target_time,
    )

    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    for x, _ in dataset:
        x = x.numpy()
        total_sum += x.sum()
        total_sq += np.square(x).sum()
        total_count += x.size

    if total_count == 0:
        return 0.0, 1.0

    mean = total_sum / total_count
    std = np.sqrt(total_sq / total_count - mean ** 2)

    print(f"XRF55 mean = {mean:.6f}")
    print(f"XRF55 std  = {std:.6f}")

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
    **kwargs
):
    """
    Added kwargs to support passing specific split definitions 
    like case_split for XRF55, shape_option for CNN models, and target_time for bimamba.
    """

    name = name.lower()
    shape_option = kwargs.get("shape_option", "2d")
    num_sub = kwargs.get("num_sub", 30) # Used for XRF55
    target_time = kwargs.get("target_time", None)
    intel_shape = kwargs.get("intel_shape", False)
    allowed_labels = kwargs.get("allowed_labels", DEFAULT_SELECTED_LABELS)
    case_split = kwargs.get("case_split", 0)
    print(f"Split case = {case_split}")

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
    # SSHAR ESP / asus
    # =======================================================
    elif name in ("sshar_esp", "sshar_asus"):
        if norm_type is None:
            norm_type = "zscore"

        cls = SSHAR_ESP_Dataset if name == "sshar_esp" else SSHAR_ASUS_Dataset
        common = dict(
            root_dir=root_dir,
            case_split=case_split,
            shape_option=shape_option,
            target_time=target_time,
            intel_shape=intel_shape,
            rooms=kwargs.get("rooms", None),
            split_mode=kwargs.get("split_mode", None),
        )

        mean = std = None
        if normalize and norm_type == "zscore":
            # statistics from the TRAIN split only (all selected rooms pooled)
            tmp = cls(split="train", normalize=False, **common)
            mean, std = compute_sshar_mean_std(tmp)

        train_dataset = cls(
            split="train", normalize=normalize, norm_type=norm_type,
            mean=mean, std=std, allowed_labels=allowed_labels, **common,
        )
        test_dataset = cls(
            split="test", normalize=normalize, norm_type=norm_type,
            mean=mean, std=std, allowed_labels=allowed_labels, **common,
        )
        print_sshar_split_summary(train_dataset, test_dataset)

    # =======================================================
    # XRF55
    # =======================================================
    elif name == "xrf55":
        if norm_type is None:
            norm_type = "zscore"

        mean = std = None
        if normalize and norm_type == "zscore":
            mean, std = compute_xrf55_mean_std(root_dir, case_split, shape_option, num_sub, target_time)

        train_dataset = XRF55Dataset(
            root_dir=root_dir,
            split="train",
            case_split=case_split,
            shape_option=shape_option,
            num_sub=num_sub,
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
            target_time=target_time,
            allowed_labels=allowed_labels,
        )

        test_dataset = XRF55Dataset(
            root_dir=root_dir,
            split="test",
            case_split=case_split,
            shape_option=shape_option,
            num_sub=num_sub,
            normalize=normalize,
            norm_type=norm_type,
            mean=mean,
            std=std,
            target_time=target_time,
            allowed_labels=allowed_labels,
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

    # Handling info extraction
    if len(train_dataset) > 0:
        x, y = train_dataset[0]
        input_shape = tuple(x.shape)
    else:
        input_shape = ()
        
    num_classes = len(set(train_dataset.y)) if isinstance(train_dataset, UTHARDataset) else len(set(s["label"] for s in train_dataset.samples))

    info = {
        "dataset": name,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
    }

    return (
        train_loader,
        test_loader,
        info,
    )

# ============================================================
# ============================================================
# V2 experiments (doc/experiment_plan_v2.md)
# ============================================================
# ============================================================
# Room 2 only; repeat split 6/2/2; global z-score with hardcoded TRAIN statistics
# (Appendix A, computed by norm_stats/compute_norm_stats.py).

V2_ROOM = "room_02"
V2_RX = ("rx_00", "rx_01", "rx_02")
V2_SPLITS = ("train", "val", "test")

# Label order act01..act08 -> index 0..7 (confusion matrices use this order)
V2_LABEL_NAMES = (
    "sitdown", "standup", "pickup", "fall",
    "lie still", "walk", "run", "empty room",
)
# XRF55 subset: act05 / act08 are "play ukulele" / "play drum"
V2_LABEL_NAMES_XRF55 = (
    "sitdown", "standup", "pickup", "fall",
    "play ukulele", "walk", "run", "play drum",
)

# Appendix A: mean/std on the TRAIN split (raw samples, 1000 time steps)
V2_NORM_STATS = {
    "esp":       {"mean": 27.3713,   "std": 7.3189},    # room 2 ESP, 3 rx x 1 antenna x 56 sub
    "asus":      {"mean": 2620.5715, "std": 640.4433},  # room 2 ASUS, 3 rx x 4 antennas x 56 sub
    "asus_ant0": {"mean": 2591.4997, "std": 695.2622},  # room 2 ASUS, 3 rx x antenna 0 x 56 sub
    "xrf55":     {"mean": 9.2577,    "std": 4.8905},    # XRF55, 3 rx x 3 antennas x 30 sub
}

# One "domain" = (device, antennas to keep, normalization statistics key)
#   antennas=None keeps every antenna of the device; device "xrf55" = XRF55 dataset.
V2_EXPERIMENTS = {
    # 1: ESP32 -> ESP32
    "esp_esp":   {"train": ("esp", None, "esp"),       "test": ("esp", None, "esp"),
                  "class_names": V2_LABEL_NAMES},
    # 2: ASUS (4 antennas) -> ASUS
    "asus_asus": {"train": ("asus", None, "asus"),     "test": ("asus", None, "asus"),
                  "class_names": V2_LABEL_NAMES},
    # 3: ASUS (antenna 0) -> ESP32  (train + val on ASUS antenna 0, test on ESP)
    "asus0_esp": {"train": ("asus", [0], "asus_ant0"), "test": ("esp", None, "esp"),
                  "class_names": V2_LABEL_NAMES},
    # 4: XRF55 -> XRF55  (--data_root = folder holding the XRF55 .npy files)
    "xrf55":     {"train": ("xrf55", None, "xrf55"),   "test": ("xrf55", None, "xrf55"),
                  "class_names": V2_LABEL_NAMES_XRF55},
    # 5: ESP32 -> ASUS (antenna 0)  (train + val on ESP, test on ASUS antenna 0)
    "esp_asus0": {"train": ("esp", None, "esp"),       "test": ("asus", [0], "asus_ant0"),
                  "class_names": V2_LABEL_NAMES},
}

# XRF55: 20 repetitions per (user, action) -> 6/2/2 = rep 1-12 / 13-16 / 17-20
XRF55_NUM_SUB = 30


def xrf55_split_of(rep):
    return "train" if rep <= 12 else ("val" if rep <= 16 else "test")


class XRF55V2Dataset(Dataset):
    """
    XRF55 (8 actions, 30 users), one split of the v2 repeat split.
    Files <user>_<action>_<rep>.npy of shape (270, 1000) = (3 rx x 3 antennas x 30 sub, T).

    __getitem__ returns a z-scored sample:
        shape_option="3d": (9, 30, T)   (CNN-GRU)
        shape_option="2d": (270, T)     (BiMamba)
    With target_time, the time axis is max-pooled AFTER normalization (BiMamba).
    """

    device = "xrf55"
    antennas = None

    def __init__(self, root_dir, split, mean, std, shape_option="2d", target_time=None):
        if split not in V2_SPLITS:
            raise ValueError(f"split must be one of {V2_SPLITS}")
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"XRF55 folder not found: {self.root_dir}")
        self.split = split
        self.mean = float(mean)
        self.std = float(std)
        self.shape_option = shape_option
        self.target_time = target_time
        self.samples = []
        for f in sorted(self.root_dir.glob("*.npy")):
            parts = f.stem.split("_")
            if len(parts) != 3:
                continue
            user, act, rep = map(int, parts)
            if act not in DEFAULT_LABEL_MAP or xrf55_split_of(rep) != split:
                continue
            self.samples.append({"label": DEFAULT_LABEL_MAP[act], "subject": user, "path": f})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = np.load(sample["path"]).astype(np.float32)          # (270, T)
        x = (x - self.mean) / self.std
        if self.shape_option == "3d":
            x = x.reshape(x.shape[0] // XRF55_NUM_SUB, XRF55_NUM_SUB, x.shape[-1])
        elif self.shape_option != "2d":
            raise ValueError(f"Unknown shape_option: {self.shape_option}")
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        if self.target_time is not None and x.shape[-1] != self.target_time:
            x = F.adaptive_max_pool1d(x, output_size=self.target_time)
        return x, sample["label"]


def v2_split_of(direction, rep):
    """Repeat split 6/2/2: dir00 (10 reps) 1-6/7-8/9-10, directional (5 reps) 1-3/4/5."""
    if direction == 0:
        return "train" if rep <= 6 else ("val" if rep <= 8 else "test")
    return "train" if rep <= 3 else ("val" if rep == 4 else "test")


class SSHARV2Dataset(Dataset):
    """
    One device of room 2, one split of the v2 repeat split.

    __getitem__ returns a z-scored sample:
        shape_option="3d": (rx * ant, 56, T)       (CNN-GRU)
        shape_option="2d": (rx * ant * 56, T)      (BiMamba)
    With target_time, the time axis is max-pooled AFTER normalization (BiMamba).
    """

    pattern = re.compile(r"act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)")

    def __init__(self, root_dir, device, split, mean, std, antennas=None,
                 shape_option="2d", target_time=None):
        if split not in V2_SPLITS:
            raise ValueError(f"split must be one of {V2_SPLITS}")
        room_dirs = dict(resolve_room_dirs(root_dir, [V2_ROOM]))
        self.room_path = room_dirs[V2_ROOM]
        self.device = device
        self.split = split
        self.mean = float(mean)
        self.std = float(std)
        self.antennas = None if antennas is None else list(antennas)
        self.shape_option = shape_option
        self.target_time = target_time
        self.samples = []
        self._build_index()

    def _build_index(self):
        base = self.room_path / self.device / V2_RX[0]
        for subject in scan_folders(base, "subject"):
            for f in sorted((base / subject).iterdir()):
                if not f.name.startswith("amp"):
                    continue
                m = self.pattern.search(f.name)
                if m is None:
                    continue
                act, _, direction, rep = map(int, m.groups())
                if act not in DEFAULT_LABEL_MAP or v2_split_of(direction, rep) != self.split:
                    continue
                self.samples.append({
                    "label": DEFAULT_LABEL_MAP[act],
                    "subject": subject,
                    "files": [self.room_path / self.device / rx / subject / f.name for rx in V2_RX],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = np.stack([np.load(f).astype(np.float32) for f in sample["files"]])  # (rx, ant, 56, T)
        if self.antennas is not None:
            x = x[:, self.antennas]
        x = (x - self.mean) / self.std
        rx, ant, sub, t = x.shape
        if self.shape_option == "3d":
            x = x.reshape(rx * ant, sub, t)
        elif self.shape_option == "2d":
            x = x.reshape(rx * ant * sub, t)
        else:
            raise ValueError(f"Unknown shape_option: {self.shape_option}")
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        if self.target_time is not None and x.shape[-1] != self.target_time:
            x = F.adaptive_max_pool1d(x, output_size=self.target_time)
        return x, sample["label"]


def load_experiment_v2(exp, root_dir, batch_size=64, shape_option="2d", target_time=None,
                       num_workers=DEFAULT_NUM_WORKERS, pin_memory=DEFAULT_PIN_MEMORY):
    """
    Build train / val / test loaders for one v2 experiment.

    Train and val come from the training domain, test from the test domain;
    each domain is normalized with its own hardcoded statistics (V2_NORM_STATS).
    """
    if exp not in V2_EXPERIMENTS:
        raise ValueError(f"Unknown exp {exp}, expected {list(V2_EXPERIMENTS)}")
    cfg = V2_EXPERIMENTS[exp]

    def make(domain, split):
        device, antennas, stats_key = domain
        stats = V2_NORM_STATS[stats_key]
        if device == "xrf55":
            return XRF55V2Dataset(root_dir, split, stats["mean"], stats["std"],
                                  shape_option=shape_option, target_time=target_time)
        return SSHARV2Dataset(root_dir, device, split, stats["mean"], stats["std"],
                              antennas=antennas, shape_option=shape_option, target_time=target_time)

    datasets = {
        "train": make(cfg["train"], "train"),
        "val": make(cfg["train"], "val"),
        "test": make(cfg["test"], "test"),
    }
    loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=num_workers, pin_memory=pin_memory)
        for split, ds in datasets.items()
    }

    x0, _ = datasets["train"][0]
    xt, _ = datasets["test"][0]
    if tuple(x0.shape) != tuple(xt.shape):
        raise ValueError(f"train/test input shapes differ: {tuple(x0.shape)} vs {tuple(xt.shape)}")

    info = {
        "exp": exp,
        "train_domain": cfg["train"],
        "test_domain": cfg["test"],
        "input_shape": tuple(x0.shape),
        "num_classes": len(cfg["class_names"]),
        "class_names": list(cfg["class_names"]),
        "train_size": len(datasets["train"]),
        "val_size": len(datasets["val"]),
        "test_size": len(datasets["test"]),
    }
    for split, ds in datasets.items():
        n_lab = len({s["label"] for s in ds.samples})
        print(f"  {split:5s}: device={ds.device:4s} antennas={ds.antennas} "
              f"mean={ds.mean} std={ds.std} samples={len(ds)} labels={n_lab}")
    return loaders["train"], loaders["val"], loaders["test"], info
