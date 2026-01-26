import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset


class TCTimeWindowDataset(Dataset):
    """Windowed dataset: maps T-step_in frames to the next frame."""

    def __init__(self, data, indices, step_in):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.step_in = step_in
        self.num_windows = data.shape[1] - step_in

    def __len__(self):
        return self.indices.size * self.num_windows

    def __getitem__(self, idx):
        sample_pos = idx // self.num_windows
        t = idx % self.num_windows
        sample_idx = self.indices[sample_pos]

        x = self.data[sample_idx, t : t + self.step_in, ...]  # [step_in, H, W, C]
        y = self.data[sample_idx, t + self.step_in, ...]      # [H, W, C]

        x = np.transpose(x, (0, 3, 1, 2))  # [step_in, C, H, W]
        y = np.transpose(y, (2, 0, 1))     # [C, H, W]

        fields = torch.from_numpy(x).float()
        target = torch.from_numpy(y).float()
        return {"fields": fields, "target_fields": target}


def load_split_arrays(temp_dir):
    tmp_path = pathlib.Path(temp_dir)
    train_path = tmp_path / "train.npy"
    val_path = tmp_path / "val.npy"
    test_path = tmp_path / "test.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing split files in {tmp_path}")
    train = np.load(train_path, mmap_mode="r")
    val = np.load(val_path, mmap_mode="r") if val_path.exists() else None
    test = np.load(test_path, mmap_mode="r")
    return train, val, test
