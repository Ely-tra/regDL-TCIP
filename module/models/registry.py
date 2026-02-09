from __future__ import annotations

import pathlib
from dataclasses import dataclass

from .architectures.afno_no_bc import TC_AFNO_NoBC
from .architectures.afno_tcp import TC_AFNO_Intensity
from .builders.afno_no_bc_v1 import AfnoNoBcV1Builder
from .builders.afno_v1 import AfnoV1Builder


@dataclass(frozen=True)
class _ArchSpec:
    builder: object
    model_cls: type
    training_policy: str


_ARCH_SPECS = {
    "afno_v1": _ArchSpec(
        builder=AfnoV1Builder(),
        model_cls=TC_AFNO_Intensity,
        training_policy="bc",
    ),
    "afno_no_bc": _ArchSpec(
        builder=AfnoNoBcV1Builder(),
        model_cls=TC_AFNO_NoBC,
        training_policy="no_bc",
    ),
}


def available_architectures() -> tuple[str, ...]:
    return tuple(_ARCH_SPECS.keys())


def _resolve_arch_name(args) -> str:
    return getattr(args, "architecture", None) or "afno_v1"


def resolve_builder(args):
    name = _resolve_arch_name(args)
    if name not in _ARCH_SPECS:
        raise ValueError(f"Unknown architecture: {name}")
    return _ARCH_SPECS[name].builder


def resolve_model_class(args):
    name = _resolve_arch_name(args)
    if name not in _ARCH_SPECS:
        raise ValueError(f"Unknown architecture: {name}")
    return _ARCH_SPECS[name].model_cls


def resolve_training_policy(args) -> str:
    name = _resolve_arch_name(args)
    if name not in _ARCH_SPECS:
        raise ValueError(f"Unknown architecture: {name}")
    return _ARCH_SPECS[name].training_policy


def build_config(args):
    builder = resolve_builder(args)
    config = builder.build_config(args)
    return builder, config


def _yaml_scalar(value) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if not value or value.strip() != value:
            escaped = value.replace("\"", "\\\"")
            return f"\"{escaped}\""
        special = ":#{}[],&*!|>'\"%@`"
        if any(ch in value for ch in special):
            escaped = value.replace("\"", "\\\"")
            return f"\"{escaped}\""
        return value
    escaped = str(value).replace("\"", "\\\"")
    return f"\"{escaped}\""


def _dump_yaml(obj, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(obj, dict):
        lines: list[str] = []
        for key, val in obj.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_dump_yaml(val, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(val)}")
        return lines
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(_dump_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    return [f"{pad}{_yaml_scalar(obj)}"]


def write_yaml(path: pathlib.Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _dump_yaml(config)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
