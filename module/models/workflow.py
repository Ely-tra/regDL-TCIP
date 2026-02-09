import json
import pathlib

import torch

from module.models.registry import resolve_model_class


def _model_config_dict(args):
    architecture = getattr(args, "architecture", None) or "afno_v1"
    return {
        "architecture": architecture,
        "num_vars": args.num_vars,
        "num_times": args.num_times,
        "height": args.height,
        "width": args.width,
        "num_blocks": args.num_blocks,
        "film_zdim": args.film_zdim,
        "e_channels": args.e_channels,
        "hidden_factor": args.hidden_factor,
        "mlp_expansion_ratio": args.mlp_expansion_ratio,
        "stem_channels": args.stem_channels,
        "checkpoint_name": args.checkpoint_name,
    }


def save_dummy_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = resolve_model_class(args)
    dummy = model_cls(
        num_vars=args.num_vars,
        num_times=args.num_times,
        H=args.height,
        W=args.width,
        num_blocks=args.num_blocks,
        film_zdim=args.film_zdim,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        return_physical=False,
        mlp_expansion_ratio=args.mlp_expansion_ratio,
        hidden_factor=args.hidden_factor,
        channels=args.e_channels,
        stem_channels=args.stem_channels,
    ).to(device)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / args.checkpoint_name
    torch.save(dummy.state_dict(), checkpoint_path)
    print(f"Saved dummy model to {checkpoint_path}", flush=True)

    config_path = args.model_config_path
    if not config_path:
        config_path = str(checkpoint_dir / "model_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(_model_config_dict(args), f, indent=2)
    print(f"Saved model config to {config_path}", flush=True)
