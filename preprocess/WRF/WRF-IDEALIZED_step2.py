import os
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk WRF idealized .npy sequences into fixed-length frame samples"
    )
    parser.add_argument(
        "-i", "--indir",
        type=str,
        default="/N/slate/kmluong/PROJECT2/WRF/wrf_data",
        help="Directory containing step1 .npy files (one per experiment)"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default="/N/slate/kmluong/PROJECT2/WRF/wrf_data_lv2",
        help="Directory to write chunked .npy outputs"
    )
    parser.add_argument(
        "-f", "--frames",
        type=int,
        default=5,
        help="Number of consecutive frames per sample"
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="wrf_idealized_frames_{frames}",
        help="Output filename prefix; may include {frames}"
    )
    parser.add_argument(
        "--channel_first",
        action="store_true",
        help="Keep channel dimension before H and W (default moves channel to last)"
    )
    return parser.parse_args()


def chunk_file(path, frames, channel_first=False):
    """Load one step1 .npy file and chunk it into samples."""
    arr = np.load(path)
    if arr.ndim != 4:
        raise ValueError(f"{path} has shape {arr.shape}, expected 4D (time, C, H, W)")

    n_time, c, h, w = arr.shape
    n_samples = n_time // frames
    if n_samples == 0:
        return None, None

    trimmed = arr[: n_samples * frames]
    chunks = trimmed.reshape(n_samples, frames, c, h, w)

    if not channel_first:
        chunks = np.transpose(chunks, (0, 1, 3, 4, 2))  # -> (samples, frames, H, W, C)

    exp_id = os.path.splitext(os.path.basename(path))[0]
    exp_ids = np.array([exp_id] * n_samples)
    return chunks, exp_ids


def process(indir, outdir, frames, prefix, channel_first):
    os.makedirs(outdir, exist_ok=True)
    X_parts, meta_parts = [], []

    for fname in sorted(os.listdir(indir)):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(indir, fname)
        try:
            chunks, exp_ids = chunk_file(fpath, frames, channel_first)
        except Exception as e:
            print(f"Skip {fname}: {e}")
            continue

        if chunks is None:
            print(f"Skip {fname}: length < frames ({frames})")
            continue

        X_parts.append(chunks)
        meta_parts.append(exp_ids)
        print(f"Added {chunks.shape[0]} samples from {fname}")

    if not X_parts:
        raise RuntimeError("No samples created; check input directory and frames length.")

    X_all = np.concatenate(X_parts, axis=0)
    meta_all = np.concatenate(meta_parts, axis=0)

    out_prefix = prefix.format(frames=frames)
    x_path = os.path.join(outdir, f"{out_prefix}_X.npy")
    meta_path = os.path.join(outdir, f"{out_prefix}_exp_ids.npy")

    np.save(x_path, X_all)
    np.save(meta_path, meta_all)

    print(f"Saved X to {x_path} with shape {X_all.shape}")
    print(f"Saved experiment ids to {meta_path}")


if __name__ == "__main__":
    args = parse_args()
    process(
        indir=args.indir,
        outdir=args.outdir,
        frames=args.frames,
        prefix=args.prefix,
        channel_first=args.channel_first,
    )
