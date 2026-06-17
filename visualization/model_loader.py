from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module.models.registry import resolve_model_class  # noqa: E402


_ARCH_ALIASES = {
    "afno_v1": "afno_v1",
    "afno_no_bc": "afno_no_bc",
    "afno_no_bc_v1": "afno_no_bc",
}


def _normalize_architecture(name: str) -> str:
    arch = (name or "").strip().lower()
    if arch in _ARCH_ALIASES:
        return _ARCH_ALIASES[arch]
    raise ValueError(f"Unsupported architecture in config: {name!r}")


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must be a dict-like object, got: {type(obj).__name__}")
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def _has_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in state_dict.keys())


def _build_model_from_config(config: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    required = [
        "architecture",
        "num_vars",
        "num_times",
        "height",
        "width",
        "num_blocks",
        "film_zdim",
        "e_channels",
        "hidden_factor",
        "mlp_expansion_ratio",
        "stem_channels",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    architecture = _normalize_architecture(str(config["architecture"]))
    args = SimpleNamespace(architecture=architecture)
    model_cls = resolve_model_class(args)

    num_vars = int(config["num_vars"])
    zeros = torch.zeros(num_vars)
    ones = torch.ones(num_vars)
    use_x_scaler = _has_prefix(state_dict, "x_scaler.")
    use_y_scaler = _has_prefix(state_dict, "y_scaler.")

    return model_cls(
        num_vars=num_vars,
        num_times=int(config["num_times"]),
        H=int(config["height"]),
        W=int(config["width"]),
        num_blocks=int(config["num_blocks"]),
        film_zdim=int(config["film_zdim"]),
        x_mean=zeros if use_x_scaler else None,
        x_std=ones if use_x_scaler else None,
        y_mean=zeros if use_y_scaler else None,
        y_std=ones if use_y_scaler else None,
        return_physical=False,
        channels=int(config["e_channels"]),
        hidden_factor=int(config["hidden_factor"]),
        mlp_expansion_ratio=int(config["mlp_expansion_ratio"]),
        stem_channels=int(config["stem_channels"]),
    )


def load_model_from_paths(
    config_path: str | Path,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    eval_mode: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """
    Build and load a model using only:
    1) JSON model config path
    2) PT checkpoint path
    """
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    ckpt_obj = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(ckpt_obj)
    model = _build_model_from_config(config, state_dict)

    load_msg = model.load_state_dict(state_dict, strict=strict)
    if eval_mode:
        model.eval()
    return model, {"config": config, "load_msg": load_msg}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load regDL-TCIP model from JSON config + PT checkpoint."
    )
    parser.add_argument("--config", required=True, help="Path to model config JSON")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--device", default="cpu", help="torch map_location, e.g. cpu or cuda:0")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable strict key matching in load_state_dict",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model, meta = load_model_from_paths(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        map_location=args.device,
        strict=not args.no_strict,
        eval_mode=True,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {model.__class__.__name__}")
    print(f"Total params: {num_params}")
    print(f"Architecture: {meta['config'].get('architecture')}")
    print(f"load_state_dict: {meta['load_msg']}")


if __name__ == "__main__":
    main()
