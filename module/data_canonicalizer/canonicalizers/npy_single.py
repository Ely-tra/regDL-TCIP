from __future__ import annotations

import numpy as np

from .base import BaseCanonicalizer, CanonicalizedData


class NpySingleCanonicalizer(BaseCanonicalizer):
    name = "npy_single"

    def supported_splitters(self):
        return ("ordered", "segmented")

    def default_splitter(self, args) -> str:
        return "segmented" if getattr(args, "num_segments", 0) > 0 else "ordered"

    def load(self, args) -> CanonicalizedData:
        data = np.load(args.data_path, mmap_mode="r")
        if data.ndim != 5:
            raise ValueError(f"Expected [N, T, H, W, C], got {data.shape}")
        return CanonicalizedData(data=data, meta={})
