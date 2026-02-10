from __future__ import annotations

import numpy as np

from .base import BaseCanonicalizer, CanonicalizedData


class NpySingleCanonicalizer(BaseCanonicalizer):
    name = "npy_single"

    def supported_splitters(self):
        return ("ordered", "segmented", "wrf_experiment")

    def default_splitter(self, args) -> str:
        return "segmented" if getattr(args, "num_segments", 0) > 0 else "ordered"

    def load(self, args) -> CanonicalizedData:
        data = np.load(args.data_path, mmap_mode="r")
        if data.ndim != 5:
            raise ValueError(f"Expected [N, T, H, W, C], got {data.shape}")
        meta: dict[str, object] = {}
        exp_ids_path = getattr(args, "exp_ids_path", "")
        if exp_ids_path:
            exp_ids = np.load(exp_ids_path)
            if exp_ids.ndim != 1:
                raise ValueError(f"Expected 1D exp_ids array, got shape {exp_ids.shape}")
            if exp_ids.shape[0] != data.shape[0]:
                raise ValueError(
                    f"exp_ids length {exp_ids.shape[0]} does not match sample count {data.shape[0]}"
                )
            meta["exp_ids"] = exp_ids.astype(str)
        return CanonicalizedData(data=data, meta=meta)
