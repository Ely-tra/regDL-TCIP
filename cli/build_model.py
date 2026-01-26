import argparse
import pathlib

from module.models.registry import available_architectures, build_config, write_yaml
from module.models.workflow import save_dummy_model

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "configs" / "model"


def parse_args():
    parser = argparse.ArgumentParser(description="Build model config, save checkpoint, and write history YAML.")
    parser.add_argument(
        "--architecture",
        choices=available_architectures(),
        default="afno_v1",
        help="Model architecture name.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to YAML history file.",
    )
    parser.add_argument("--num_vars", type=int, default=11, help="Number of input/output variables")
    parser.add_argument("--num_times", type=int, default=3, help="Number of input time steps")
    parser.add_argument("--height", type=int, default=100, help="Grid height")
    parser.add_argument("--width", type=int, default=100, help="Grid width")
    parser.add_argument("--num_blocks", type=int, default=6, help="Number of AFNO blocks")
    parser.add_argument("--film_zdim", type=int, default=128, help="FiLM conditioning embedding dim.")
    parser.add_argument("--e_channels", type=int, default=256, help="Expressive channels")
    parser.add_argument("--hidden_factor", type=int, default=8, help="Hidden expansion factor")
    parser.add_argument("--mlp_expansion_ratio", type=int, default=4, help="MLP expansion ratio")
    parser.add_argument("--stem_channels", type=int, default=256, help="Input CNN expressive power")
    parser.add_argument(
        "--checkpoint_dir",
        default="/N/slate/kmluong/PROJECT2/checkpoints",
        help="Checkpoint output directory",
    )
    parser.add_argument("--checkpoint_name", default="AFNO-TCP.pt", help="Model filename")
    parser.add_argument(
        "--model_config_path",
        default="",
        help="Optional path to write model config JSON",
    )
    return parser.parse_args()


def run(args):
    builder, config = build_config(args)
    output_path = pathlib.Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / builder.yaml_name
    write_yaml(output_path, config)
    print(f"Saved model history to {output_path}", flush=True)
    save_dummy_model(args)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
