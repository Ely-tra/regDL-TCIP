from __future__ import annotations

import re

import numpy as np

from .base import BaseSplitter, SplitResult


def _parse_exp_string(token: str) -> list[int]:
    return [int(x) for x in re.findall(r"10|[1-9]", token)]


def _resolve_exp_lists(args) -> tuple[list[int], list[int], list[int]]:
    if getattr(args, "train_exp", None) or getattr(args, "val_exp", None) or getattr(args, "test_exp", None):
        return (
            list(args.train_exp or []),
            list(args.val_exp or []),
            list(args.test_exp or []),
        )
    parts = str(getattr(args, "exp_split", "")).split("+")
    if len(parts) != 3:
        raise ValueError("exp_split must be in the form train+val+test")
    return (
        _parse_exp_string(parts[0]),
        _parse_exp_string(parts[1]),
        _parse_exp_string(parts[2]),
    )


def _exp_num_from_token(token: str) -> int:
    text = str(token)
    if re.fullmatch(r"\d+", text):
        return int(text)
    for pattern in [r"exp_02km_m(\d{2})", r"_m(\d{2})", r"m(\d{1,2})"]:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not parse experiment number from exp_id token: {text!r}")


class WrfExperimentSplitter(BaseSplitter):
    name = "wrf_experiment"

    def split(self, data: np.ndarray, args, meta) -> SplitResult:
        train_exp, val_exp, test_exp = _resolve_exp_lists(args)
        all_exp = train_exp + val_exp + test_exp
        if not all_exp:
            raise ValueError("No experiments provided for data_mode=1")

        exp_ids = meta.get("exp_ids")
        if exp_ids is None:
            if data.shape[0] != len(all_exp):
                raise ValueError("WRF data count does not match experiment list length")

            train_idx = np.arange(0, len(train_exp), dtype=np.int64)
            val_idx = np.arange(len(train_exp), len(train_exp) + len(val_exp), dtype=np.int64)
            test_idx = np.arange(len(train_exp) + len(val_exp), data.shape[0], dtype=np.int64)
            return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

        exp_ids_arr = np.asarray(exp_ids).astype(str)
        if exp_ids_arr.ndim != 1:
            raise ValueError(f"exp_ids must be 1D, got shape {exp_ids_arr.shape}")
        if exp_ids_arr.shape[0] != data.shape[0]:
            raise ValueError(
                f"exp_ids length {exp_ids_arr.shape[0]} does not match sample count {data.shape[0]}"
            )

        exp_nums = np.array([_exp_num_from_token(tok) for tok in exp_ids_arr], dtype=np.int64)
        train_idx = np.where(np.isin(exp_nums, np.array(train_exp, dtype=np.int64)))[0].astype(np.int64)
        val_idx = np.where(np.isin(exp_nums, np.array(val_exp, dtype=np.int64)))[0].astype(np.int64)
        test_idx = np.where(np.isin(exp_nums, np.array(test_exp, dtype=np.int64)))[0].astype(np.int64)

        assigned_mask = np.isin(exp_nums, np.array(all_exp, dtype=np.int64))
        if not np.all(assigned_mask):
            unknown = sorted(set(exp_nums[~assigned_mask].tolist()))
            raise ValueError(
                f"Found samples with exp ids not covered by exp_split/train_exp/val_exp/test_exp: {unknown}"
            )
        return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
