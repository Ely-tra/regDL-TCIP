from __future__ import annotations

import pathlib
import re

import numpy as np

from .base import BaseCanonicalizer, CanonicalizedData


def _parse_exp_string(token: str) -> list[int]:
    return [int(x) for x in re.findall(r"10|[1-9]", token)]


def _resolve_exp_lists(args) -> tuple[list[int], list[int], list[int]]:
    if getattr(args, "train_exp", None) or getattr(args, "val_exp", None) or getattr(args, "test_exp", None):
        return (
            list(args.train_exp or []),
            list(args.val_exp or []),
            list(args.test_exp or []),
        )
    parts = str(args.exp_split).split("+")
    if len(parts) != 3:
        raise ValueError("exp_split must be in the form train+val+test")
    return (
        _parse_exp_string(parts[0]),
        _parse_exp_string(parts[1]),
        _parse_exp_string(parts[2]),
    )


class WrfExperimentCanonicalizer(BaseCanonicalizer):
    name = "wrf_experiments"

    def supported_splitters(self):
        return ("wrf_experiment",)

    def default_splitter(self, args) -> str:
        return "wrf_experiment"

    def load(self, args) -> CanonicalizedData:
        train_exp, val_exp, test_exp = _resolve_exp_lists(args)
        all_exp = train_exp + val_exp + test_exp
        if not all_exp:
            raise ValueError("No experiments provided for data_mode=1")

        data_list = []
        for exp_num in all_exp:
            fname = f"x_{args.x_resolution}_{args.imsize}x{args.imsize}_exp_02km_m{exp_num:02d}.npy"
            fpath = pathlib.Path(args.wrf_dir) / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing WRF file: {fpath}")
            arr = np.load(fpath, mmap_mode="r")
            if arr.ndim != 4:
                raise ValueError(f"Expected [T, C, H, W] for {fpath}, got {arr.shape}")
            arr = np.transpose(arr, (0, 2, 3, 1))
            data_list.append(arr)

        min_t = min(a.shape[0] for a in data_list)
        data_list = [a[:min_t] for a in data_list]
        data = np.stack(data_list, axis=0)
        meta = {"train_exp": train_exp, "val_exp": val_exp, "test_exp": test_exp}
        return CanonicalizedData(data=data, meta=meta)
