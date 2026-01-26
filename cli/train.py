import argparse

from module.training.registry import available_trainers, run_training


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"yes", "true", "t", "y", "1"}:
        return True
    if v in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AFNO model from temp split arrays.")
    parser.add_argument(
        "--trainer",
        choices=available_trainers(),
        default="afno_tcp_v1",
        help="Training pipeline name.",
    )
    parser.add_argument("--step_in", type=int, default=3, help="The number of frames used as input")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader worker count")
    parser.add_argument("--pin_memory", type=str2bool, default=True, help="Pin CPU memory in dataloader")
    parser.add_argument("--num_vars", type=int, default=11, help="Number of input/output variables")
    parser.add_argument("--num_times", type=int, default=3, help="Number of input time steps")
    parser.add_argument("--height", type=int, default=100, help="Grid height")
    parser.add_argument("--width", type=int, default=100, help="Grid width")
    parser.add_argument("--num_blocks", type=int, default=8, help="Number of AFNO blocks, or hidden layers")
    parser.add_argument("--rim", type=int, default=1, help="Boundary pixels enforced from target.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--film_zdim", type=int, default=64, help="FiLM conditioning embedding dim.")
    parser.add_argument(
        "--checkpoint_dir",
        default="/N/slate/kmluong/PROJECT2/checkpoints",
        help="Checkpoint output directory",
    )
    parser.add_argument("--e_channels", type=int, default=256, help="Expressive channels")
    parser.add_argument("--hidden_factor", type=int, default=8, help="Hidden expansion factor")
    parser.add_argument("--mlp_expansion_ratio", type=int, default=4, help="MLP expansion ratio")
    parser.add_argument("--stem_channels", type=int, default=256, help="Input CNN expressive power")
    parser.add_argument("--checkpoint_name", default="test_model.pt", help="Checkpoint filename")
    parser.add_argument("--init_model_path", default="", help="Optional path to a dummy/initial model state_dict")
    parser.add_argument(
        "--model_config_path",
        default="",
        help="Optional path to model config JSON from architecture step",
    )
    parser.add_argument(
        "--temp",
        "-tmp",
        dest="temp_dir",
        default="/N/slate/kmluong/PROJECT2/tmp",
        help="Temp directory for train/val/test split arrays",
    )
    parser.add_argument("--loss_gamma_center", type=float, default=1.0, help="Center emphasis strength")
    parser.add_argument("--loss_center_width", type=float, default=0.25, help="Center Gaussian sigma")
    parser.add_argument(
        "--loss_center_width_mode",
        choices=["ratio", "pixels"],
        default="ratio",
        help="Interpret center width as ratio or pixels",
    )
    parser.add_argument("--loss_gamma_extreme", type=float, default=1.0, help="Extreme emphasis strength")
    parser.add_argument(
        "--loss_extreme_mode",
        choices=["abs", "zscore", "percentile"],
        default="zscore",
        help="Extreme scoring mode",
    )
    parser.add_argument("--loss_extreme_q", type=float, default=0.70, help="Extreme percentile threshold")
    parser.add_argument("--loss_extreme_scale", type=float, default=1.0, help="Extreme shaping scale")
    parser.add_argument(
        "--loss_combine",
        choices=["mul", "add"],
        default="mul",
        help="Combine center/extreme weights",
    )
    parser.add_argument("--loss_normalize_weights", type=str2bool, default=True, help="Normalize weights to mean 1")
    parser.add_argument("--loss_eps", type=float, default=1e-6, help="Loss numerical stability epsilon")
    parser.add_argument("--loss_alpha", type=float, default=0.0, help="Center term weight")
    parser.add_argument("--loss_beta", type=float, default=0.5, help="Extreme term weight")
    parser.add_argument("--loss_zeta", type=float, default=1, help="High-frequency loss weight")
    parser.add_argument(
        "--high_freq_component_loss",
        type=str2bool,
        default=False,
        help="Use magnitude-spectrum loss instead of residual-spectrum loss",
    )
    parser.add_argument(
        "--high_freq_cutoff_ratio",
        type=float,
        default=0.5,
        help="High-frequency cutoff ratio in [0, 1].",
    )
    parser.add_argument("--L1_weight", type=float, default=1.0, help="Weight for L1 loss (>= 0)")
    parser.add_argument("--L2_weight", type=float, default=0.0, help="Weight for L2 loss (>= 0)")
    parser.add_argument("--Center_weight", type=float, default=0.0, help="Weight for center loss (>= 0)")
    parser.add_argument("--Extreme_weight", type=float, default=0.0, help="Weight for extreme loss (>= 0)")
    parser.add_argument("--HighFreq_weight", type=float, default=0.0, help="Weight for high-freq loss (>= 0)")
    return parser.parse_args()


def run(args):
    run_training(args)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
