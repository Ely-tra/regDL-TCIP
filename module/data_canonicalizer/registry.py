from __future__ import annotations

import pathlib

import numpy as np

from .canonicalizers.base import BaseCanonicalizer, CanonicalizedData
from .canonicalizers.npy_single import NpySingleCanonicalizer
from .canonicalizers.wrf_experiments import WrfExperimentCanonicalizer
from .splitters.base import BaseSplitter, SplitResult
from .splitters.ordered import OrderedSplitter
from .splitters.segmented import SegmentedSplitter
from .splitters.wrf_experiment import WrfExperimentSplitter

_CANONICALIZERS: dict[str, BaseCanonicalizer] = {
    "npy_single": NpySingleCanonicalizer(),
    "wrf_experiments": WrfExperimentCanonicalizer(),
}

_SPLITTERS: dict[str, BaseSplitter] = {
    "ordered": OrderedSplitter(),
    "segmented": SegmentedSplitter(),
    "wrf_experiment": WrfExperimentSplitter(),
}


def available_canonicalizers() -> tuple[str, ...]:
    return tuple(_CANONICALIZERS.keys())


def available_splitters() -> tuple[str, ...]:
    return tuple(_SPLITTERS.keys())


def _get_canonicalizer(name: str) -> BaseCanonicalizer:
    if name not in _CANONICALIZERS:
        raise ValueError(f"Unknown canonicalizer: {name}")
    return _CANONICALIZERS[name]


def _get_splitter(name: str) -> BaseSplitter:
    if name not in _SPLITTERS:
        raise ValueError(f"Unknown splitter: {name}")
    return _SPLITTERS[name]


def resolve_canonicalizer(args) -> BaseCanonicalizer:
    canonicalizer_name = getattr(args, "canonicalizer", None)
    if canonicalizer_name:
        return _get_canonicalizer(canonicalizer_name)

    data_mode = getattr(args, "data_mode", None)
    if data_mode == 0:
        canonicalizer_name = "npy_single"
    elif data_mode == 1:
        canonicalizer_name = "wrf_experiments"
    else:
        raise ValueError("data_mode must be provided when canonicalizer is not set")
    return _get_canonicalizer(canonicalizer_name)


def resolve_splitter(canonicalizer: BaseCanonicalizer, args) -> BaseSplitter:
    splitter_name = getattr(args, "splitter", None)
    if splitter_name:
        if splitter_name not in canonicalizer.supported_splitters():
            raise ValueError(
                f"Splitter {splitter_name} is not supported by canonicalizer {canonicalizer.name}"
            )
    else:
        splitter_name = canonicalizer.default_splitter(args)

    if splitter_name not in canonicalizer.supported_splitters():
        raise ValueError(f"Splitter {splitter_name} is not supported by canonicalizer {canonicalizer.name}")
    return _get_splitter(splitter_name)


def canonicalize_and_split(args):
    canonicalizer = resolve_canonicalizer(args)
    canonicalized = canonicalizer.load(args)
    splitter = resolve_splitter(canonicalizer, args)
    split_result = splitter.split(canonicalized.data, args, canonicalized.meta)
    return canonicalizer, splitter, canonicalized, split_result


def _write_split_meta(
    meta_path: pathlib.Path,
    args,
    data: np.ndarray,
    split_result: SplitResult,
    meta: dict[str, object],
    canonicalizer_name: str,
    splitter_name: str,
):
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"canonicalizer={canonicalizer_name}\n")
        f.write(f"splitter={splitter_name}\n")
        f.write(f"data_mode={getattr(args, 'data_mode', None)}\n")
        f.write(f"data_path={getattr(args, 'data_path', None)}\n")
        f.write(f"wrf_dir={getattr(args, 'wrf_dir', None)}\n")
        f.write(f"train_frac={getattr(args, 'train_frac', None)}\n")
        f.write(f"val_frac={getattr(args, 'val_frac', None)}\n")
        f.write(f"test_frac={getattr(args, 'test_frac', None)}\n")
        f.write(f"num_segments={getattr(args, 'num_segments', None)}\n")
        f.write(f"segment_seed={getattr(args, 'segment_seed', None)}\n")
        if "train_exp" in meta:
            f.write(f"train_exp={meta['train_exp']}\n")
        if "val_exp" in meta:
            f.write(f"val_exp={meta['val_exp']}\n")
        if "test_exp" in meta:
            f.write(f"test_exp={meta['test_exp']}\n")
        f.write(f"N={data.shape[0]}\n")
        f.write(f"train_size={split_result.train_idx.size}\n")
        f.write(f"val_size={split_result.val_idx.size}\n")
        f.write(f"test_size={split_result.test_idx.size}\n")


def save_splits(
    args,
    data: np.ndarray,
    split_result: SplitResult,
    meta: dict[str, object],
    canonicalizer_name: str,
    splitter_name: str,
):
    if not getattr(args, "temp_dir", None):
        raise ValueError("temp_dir is required; more functions will be added later.")

    tmp_path = pathlib.Path(args.temp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)

    np.save(tmp_path / "train.npy", np.asarray(data[split_result.train_idx]))
    np.save(tmp_path / "test.npy", np.asarray(data[split_result.test_idx]))
    if split_result.val_idx.size > 0 and getattr(args, "val_frac", 1) > 0:
        np.save(tmp_path / "val.npy", np.asarray(data[split_result.val_idx]))

    _write_split_meta(
        tmp_path / "split_meta.txt",
        args,
        data,
        split_result,
        meta,
        canonicalizer_name,
        splitter_name,
    )
