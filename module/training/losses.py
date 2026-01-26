from typing import Callable, Optional, Literal, Dict, Tuple

import torch
import torch.fft as fft
import torch.nn.functional as F


def loss_map_l1(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(y_pred, y_true, reduction="none")


def loss_map_l2(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_pred, y_true, reduction="none")


def loss_l1(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return loss_map_l1(y_pred, y_true).mean()


def loss_l2(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return loss_map_l2(y_pred, y_true).mean()


LOSS_REGISTRY = {
    "L1": loss_l1,
    "L2": loss_l2,
}

LOSS_MAP_REGISTRY = {
    "L1": loss_map_l1,
    "L2": loss_map_l2,
}

BASE_LOSSES = tuple(LOSS_REGISTRY.keys())
AUX_LOSSES = ("Center", "Extreme", "HighFreq")


def available_losses() -> tuple[str, ...]:
    return BASE_LOSSES


def available_aux_losses() -> tuple[str, ...]:
    return AUX_LOSSES


def _validate_weight(name: str, weight: float) -> float:
    if weight < 0.0:
        raise ValueError(f"{name}_weight must be >= 0, got {weight}.")
    return weight


def collect_loss_weights(args, names: tuple[str, ...]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for name in names:
        attr = f"{name}_weight"
        if hasattr(args, attr):
            weight = float(getattr(args, attr))
            weights[name] = _validate_weight(name, weight)
    return weights


_HF_MASK_CACHE: Dict[Tuple[int, int, float, torch.device, torch.dtype], torch.Tensor] = {}


def _make_center_gaussian(
    H: int,
    W: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    sigma2 = max(sigma * sigma, eps)
    g = torch.exp(-0.5 * d2 / sigma2)
    return g.view(1, 1, H, W)


def _center_weight_map(
    y_true: torch.Tensor,
    gamma_center: float,
    center_width: float,
    center_width_mode: str,
    eps: float,
    center_map_fn: Optional[Callable[[int, int, torch.device, torch.dtype], torch.Tensor]] = None,
) -> torch.Tensor:
    B, C, H, W = y_true.shape
    if gamma_center <= 0:
        return torch.ones((B, C, H, W), device=y_true.device, dtype=y_true.dtype)

    if center_width_mode == "ratio":
        sigma = center_width * float(min(H, W))
    else:
        sigma = center_width

    if center_map_fn is not None:
        g = center_map_fn(H, W, y_true.device, y_true.dtype)
    else:
        g = _make_center_gaussian(H=H, W=W, sigma=sigma, device=y_true.device, dtype=y_true.dtype, eps=eps)

    wc = 1.0 + gamma_center * g
    return wc.expand(B, C, H, W)


def _extreme_score(
    y_true: torch.Tensor,
    extreme_mode: str,
    eps: float,
    extreme_score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    if extreme_score_fn is not None:
        return extreme_score_fn(y_true)

    if extreme_mode == "abs":
        return torch.abs(y_true)

    if extreme_mode in ["zscore", "percentile"]:
        mean = y_true.mean(dim=(-2, -1), keepdim=True)
        std = y_true.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
        z = (y_true - mean) / std
        return torch.abs(z)

    raise ValueError(f"Unknown extreme_mode={extreme_mode}")


def _extreme_weight_map(
    y_true: torch.Tensor,
    gamma_extreme: float,
    extreme_mode: str,
    extreme_q: float,
    extreme_scale: float,
    eps: float,
    extreme_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    extreme_score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    B, C, H, W = y_true.shape
    if gamma_extreme <= 0:
        return torch.ones((B, C, H, W), device=y_true.device, dtype=y_true.dtype)

    score = _extreme_score(y_true, extreme_mode, eps, extreme_score_fn)
    if extreme_weight_fn is not None:
        we = extreme_weight_fn(score)
        return we

    if extreme_mode == "percentile":
        flat = score.view(B, C, -1)
        thr = torch.quantile(flat, q=extreme_q, dim=-1, keepdim=True)
        thr = thr.view(B, C, 1, 1)
        excess = (score - thr).clamp_min(0.0)
    else:
        excess = score

    shaped = torch.tanh(excess / max(extreme_scale, eps))
    we = 1.0 + gamma_extreme * shaped
    return we


def _make_high_freq_mask(
    H: int,
    W: int,
    cutoff_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (H, W, float(cutoff_ratio), device, dtype)
    cached = _HF_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    ky = fft.fftfreq(H, d=1.0, device=device, dtype=dtype)[:, None]
    kx = fft.rfftfreq(W, d=1.0, device=device, dtype=dtype)[None, :]
    kk = torch.sqrt(ky ** 2 + kx ** 2)
    cutoff = cutoff_ratio * 0.5
    mask = (kk >= cutoff).float()
    mask = mask.view(1, 1, H, W // 2 + 1)
    _HF_MASK_CACHE[key] = mask
    return mask


def center_loss(
    base_map: torch.Tensor,
    y_true: torch.Tensor,
    gamma_center: float,
    center_width: float,
    center_width_mode: str,
    normalize_weights: bool,
    eps: float,
    center_map_fn: Optional[Callable[[int, int, torch.device, torch.dtype], torch.Tensor]] = None,
) -> torch.Tensor:
    wc = _center_weight_map(
        y_true=y_true,
        gamma_center=gamma_center,
        center_width=center_width,
        center_width_mode=center_width_mode,
        eps=eps,
        center_map_fn=center_map_fn,
    )
    if normalize_weights:
        wc = wc / wc.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (wc * base_map).mean()


def extreme_loss(
    base_map: torch.Tensor,
    y_true: torch.Tensor,
    gamma_extreme: float,
    extreme_mode: str,
    extreme_q: float,
    extreme_scale: float,
    normalize_weights: bool,
    eps: float,
    extreme_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    extreme_score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    we = _extreme_weight_map(
        y_true=y_true,
        gamma_extreme=gamma_extreme,
        extreme_mode=extreme_mode,
        extreme_q=extreme_q,
        extreme_scale=extreme_scale,
        eps=eps,
        extreme_weight_fn=extreme_weight_fn,
        extreme_score_fn=extreme_score_fn,
    )
    if normalize_weights:
        we = we / we.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (we * base_map).mean()


def high_freq_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    high_freq_mode: str,
    high_freq_cutoff_ratio: float,
) -> torch.Tensor:
    _, _, H, W = y_true.shape
    mask = _make_high_freq_mask(H, W, high_freq_cutoff_ratio, y_true.device, y_true.dtype)
    if high_freq_mode == "residual":
        diff = y_pred - y_true
        spec = fft.rfft2(diff, norm="ortho")
        mag = torch.abs(spec)
        hf = mag * mask
        return hf.mean()
    if high_freq_mode == "magnitude":
        pred_spec = fft.rfft2(y_pred, norm="ortho")
        true_spec = fft.rfft2(y_true, norm="ortho")
        mag = torch.abs(pred_spec) - torch.abs(true_spec)
        hf = torch.abs(mag) * mask
        return hf.mean()
    raise ValueError(f"Unknown high_freq_mode={high_freq_mode}")


def build_weighted_loss(args):
    base_weights = collect_loss_weights(args, BASE_LOSSES)
    aux_weights = collect_loss_weights(args, AUX_LOSSES)

    has_base = any(weight > 0 for weight in base_weights.values())
    center_weight = aux_weights.get("Center", 0.0)
    extreme_weight = aux_weights.get("Extreme", 0.0)
    high_freq_weight = aux_weights.get("HighFreq", 0.0)

    if not has_base and (center_weight > 0 or extreme_weight > 0):
        raise ValueError("Center/Extreme loss requires at least one base loss weight > 0.")
    if not has_base and center_weight <= 0 and extreme_weight <= 0 and high_freq_weight <= 0:
        raise ValueError("At least one loss weight must be > 0.")

    def _loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        total = y_pred.new_tensor(0.0)
        base_map = None
        for name, weight in base_weights.items():
            if weight <= 0:
                continue
            loss_map_fn = LOSS_MAP_REGISTRY.get(name)
            if loss_map_fn is None:
                raise ValueError(f"Unknown loss name: {name}")
            loss_map = loss_map_fn(y_pred, y_true)
            base_map = loss_map * weight if base_map is None else base_map + loss_map * weight
            total = total + weight * loss_map.mean()

        if center_weight > 0:
            if base_map is None:
                raise ValueError("Center loss requires at least one base loss weight > 0.")
            total = total + center_weight * center_loss(
                base_map=base_map,
                y_true=y_true,
                gamma_center=args.loss_gamma_center,
                center_width=args.loss_center_width,
                center_width_mode=args.loss_center_width_mode,
                normalize_weights=args.loss_normalize_weights,
                eps=args.loss_eps,
            )
        if extreme_weight > 0:
            if base_map is None:
                raise ValueError("Extreme loss requires at least one base loss weight > 0.")
            total = total + extreme_weight * extreme_loss(
                base_map=base_map,
                y_true=y_true,
                gamma_extreme=args.loss_gamma_extreme,
                extreme_mode=args.loss_extreme_mode,
                extreme_q=args.loss_extreme_q,
                extreme_scale=args.loss_extreme_scale,
                normalize_weights=args.loss_normalize_weights,
                eps=args.loss_eps,
            )
        if high_freq_weight > 0:
            total = total + high_freq_weight * high_freq_loss(
                y_pred=y_pred,
                y_true=y_true,
                high_freq_mode="magnitude" if args.high_freq_component_loss else "residual",
                high_freq_cutoff_ratio=args.high_freq_cutoff_ratio,
            )
        return total

    return _loss
