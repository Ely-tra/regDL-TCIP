from __future__ import annotations

import numpy as np

from .base import BaseSplitter, SplitResult


class SegmentedSplitter(BaseSplitter):
    name = "segmented"

    def split(self, data: np.ndarray, args, meta) -> SplitResult:
        N = data.shape[0]
        train_frac = args.train_frac
        val_frac = args.val_frac
        test_frac = args.test_frac
        num_segments = args.num_segments
        seed = args.segment_seed

        if num_segments <= 0:
            raise ValueError("num_segments must be > 0 for segmented split.")
        if not (0 < train_frac < 1) or not (0 < val_frac < 1) or not (0 < test_frac < 1):
            raise ValueError("train_frac, val_frac, test_frac must be in (0, 1).")
        if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1 for segmented split.")

        total_val_test = val_frac + test_frac
        seg_len = total_val_test / num_segments
        test_len = test_frac / num_segments

        half_seg = seg_len / 2.0
        half_test = test_len / 2.0
        low = half_seg
        high = 1.0 - half_seg

        rng = np.random.default_rng(seed)
        centers: list[float] = []
        segments: list[tuple[float, float]] = []
        max_tries = 1000
        tries = 0
        while len(centers) < num_segments and tries < max_tries:
            center = float(rng.uniform(low, high))
            seg = (center - half_seg, center + half_seg)
            overlaps = any(not (seg[1] <= s[0] or seg[0] >= s[1]) for s in segments)
            if not overlaps:
                centers.append(center)
                segments.append(seg)
            tries += 1

        if len(centers) < num_segments:
            centers = [float((i + 0.5) / num_segments) for i in range(num_segments)]

        val_idx_list: list[np.ndarray] = []
        test_idx_list: list[np.ndarray] = []
        for center in centers:
            seg_start = center - half_seg
            seg_end = center + half_seg
            test_start = center - half_test
            test_end = center + half_test

            seg_start_i = int(round(seg_start * N))
            seg_end_i = int(round(seg_end * N))
            test_start_i = int(round(test_start * N))
            test_end_i = int(round(test_end * N))

            seg_start_i = max(0, min(N, seg_start_i))
            seg_end_i = max(0, min(N, seg_end_i))
            test_start_i = max(seg_start_i, min(N, test_start_i))
            test_end_i = max(test_start_i, min(seg_end_i, test_end_i))

            if seg_start_i < test_start_i:
                val_idx_list.append(np.arange(seg_start_i, test_start_i, dtype=np.int64))
            if test_start_i < test_end_i:
                test_idx_list.append(np.arange(test_start_i, test_end_i, dtype=np.int64))
            if test_end_i < seg_end_i:
                val_idx_list.append(np.arange(test_end_i, seg_end_i, dtype=np.int64))

        val_idx = np.unique(np.concatenate(val_idx_list)) if val_idx_list else np.array([], dtype=np.int64)
        test_idx = np.unique(np.concatenate(test_idx_list)) if test_idx_list else np.array([], dtype=np.int64)

        used = np.zeros(N, dtype=bool)
        used[val_idx] = True
        used[test_idx] = True
        train_idx = np.where(~used)[0].astype(np.int64)

        if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
            raise ValueError(
                f"Bad split sizes: train={train_idx.size}, val={val_idx.size}, test={test_idx.size}"
            )
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
