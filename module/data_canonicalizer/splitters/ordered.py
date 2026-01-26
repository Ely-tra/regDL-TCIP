from __future__ import annotations

import numpy as np

from .base import BaseSplitter, SplitResult


class OrderedSplitter(BaseSplitter):
    name = "ordered"

    def split(self, data: np.ndarray, args, meta) -> SplitResult:
        N = data.shape[0]
        train_frac = args.train_frac
        val_frac = args.val_frac
        test_frac = args.test_frac

        if not (0 < train_frac < 1) or not (0 < val_frac < 1):
            raise ValueError("train_frac and val_frac must be in (0, 1).")
        if test_frac is None:
            test_frac = 1.0 - train_frac - val_frac
        if not (0 < test_frac < 1):
            raise ValueError("test_frac must be in (0, 1).")
        if train_frac + val_frac + test_frac > 1:
            raise ValueError("train_frac + val_frac + test_frac must be <= 1.")

        n_train = int(round(N * train_frac))
        n_val = int(round(N * val_frac))

        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= N:
            n_val = max(1, N - n_train - 1)

        train_idx = np.arange(0, n_train, dtype=np.int64)
        val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
        test_idx = np.arange(n_train + n_val, N, dtype=np.int64)

        if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
            raise ValueError(
                f"Bad split sizes: train={train_idx.size}, val={val_idx.size}, test={test_idx.size}"
            )
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
