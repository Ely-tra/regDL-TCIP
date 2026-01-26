from __future__ import annotations

import numpy as np

from .base import BaseSplitter, SplitResult


class WrfExperimentSplitter(BaseSplitter):
    name = "wrf_experiment"

    def split(self, data: np.ndarray, args, meta) -> SplitResult:
        train_exp = list(meta.get("train_exp", []))
        val_exp = list(meta.get("val_exp", []))
        test_exp = list(meta.get("test_exp", []))
        all_exp = train_exp + val_exp + test_exp
        if not all_exp:
            raise ValueError("No experiments provided for data_mode=1")

        if data.shape[0] != len(all_exp):
            raise ValueError("WRF data count does not match experiment list length")

        train_idx = np.arange(0, len(train_exp), dtype=np.int64)
        val_idx = np.arange(len(train_exp), len(train_exp) + len(val_exp), dtype=np.int64)
        test_idx = np.arange(len(train_exp) + len(val_exp), data.shape[0], dtype=np.int64)
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
