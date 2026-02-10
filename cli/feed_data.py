import argparse

from module.data_canonicalizer.registry import (
    available_canonicalizers,
    available_splitters,
    canonicalize_and_split,
    save_splits,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test arrays.")
    parser.add_argument(
        "--canonicalizer",
        choices=available_canonicalizers(),
        default=None,
        help="Optional canonicalizer override.",
    )
    parser.add_argument(
        "--splitter",
        choices=available_splitters(),
        default=None,
        help="Optional splitter override.",
    )
    parser.add_argument(
        "--data_mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="Data loading mode: 0=single .npy dataset, 1=WRF exp files.",
    )
    parser.add_argument(
        "--data_path",
        default="/N/slate/kmluong/PROJECT2/level_2_data/wrf_tropical_cyclone_track_5_dataset_X.npy",
        help="Path to the input .npy dataset",
    )
    parser.add_argument(
        "--exp_ids_path",
        default="",
        help="Optional path to sample-level experiment IDs (.npy); used with wrf_experiment splitter",
    )
    parser.add_argument(
        "--wrf_dir",
        default="/N/slate/kmluong/PROJECT2/WRF/wrf_data",
        help="Directory containing per-experiment WRF .npy files",
    )
    parser.add_argument("--x_resolution", default="d01", help="X resolution string in filename")
    parser.add_argument("--imsize", type=int, default=100, help="Spatial size used in filename")
    parser.add_argument(
        "--train_exp",
        type=int,
        nargs="+",
        default=None,
        help="Experiment numbers for training (e.g. 1 2 3 4 8 9 10)",
    )
    parser.add_argument(
        "--val_exp",
        type=int,
        nargs="+",
        default=None,
        help="Experiment numbers for validation",
    )
    parser.add_argument(
        "--test_exp",
        type=int,
        nargs="+",
        default=None,
        help="Experiment numbers for testing",
    )
    parser.add_argument(
        "--exp_split",
        default="12348910+57+6",
        help="Compact exp split string: train+val+test (default: 12348910+57+6)",
    )
    parser.add_argument("--train_frac", type=float, default=0.7, help="Training split fraction")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test split fraction")
    parser.add_argument(
        "--num_segments",
        type=int,
        default=1,
        help="Number of val/test segments (0 = disable segmented split)",
    )
    parser.add_argument("--segment_seed", type=int, default=None, help="Random seed for segmented split")
    parser.add_argument("--step_in", type=int, default=3, help="The number of frames used as input")
    parser.add_argument(
        "--temp",
        "-tmp",
        dest="temp_dir",
        default="/N/slate/kmluong/PROJECT2/tmp",
        help="Temp directory for train/val/test split arrays",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    canonicalizer, splitter, canonicalized, split_result = canonicalize_and_split(args)

    data = canonicalized.data
    if data.ndim != 5:
        raise ValueError(f"Expected [N, T, H, W, C], got {data.shape}")
    if data.shape[1] < args.step_in + 1:
        raise ValueError("Need at least step_in+1 temporal frames")

    save_splits(args, data, split_result, canonicalized.meta, canonicalizer.name, splitter.name)


if __name__ == "__main__":
    main()
