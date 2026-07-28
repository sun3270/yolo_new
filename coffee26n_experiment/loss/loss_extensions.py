"""Independent CoffeeLoss ramps, WIoU, TailBCE and PairMargin helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F


def smooth_ramp(progress: float, start: float = 0.10, end: float = 0.60) -> float:
    if end <= start:
        return 1.0 if progress >= end else 0.0
    x = min(max((float(progress) - start) / (end - start), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def wiou_ramp(progress: float, start: float = 0.10, end: float = 0.60) -> float:
    return smooth_ramp(progress, start, end)


def tail_ramp(progress: float, start: float = 0.10, end: float = 0.60) -> float:
    return smooth_ramp(progress, start, end)


def pair_ramp(progress: float, start: float = 0.10, end: float = 0.60) -> float:
    return smooth_ramp(progress, start, end)


def build_tail_weights(counts: Iterable[float] | dict[int, float] | None, nc: int, exponent: float = 0.5,
                       cap: float = 2.4, enabled: bool = True, device: torch.device | None = None) -> torch.Tensor:
    """Build non-downweighting tail weights from current train-label counts."""
    if not enabled:
        return torch.ones(nc, dtype=torch.float32, device=device)
    if counts is None:
        raise ValueError("TailBCE enabled but no current dataset class counts were provided.")
    values = [counts.get(index, 0) for index in range(nc)] if isinstance(counts, dict) else list(counts)
    if len(values) != nc or any(float(value) <= 0 for value in values):
        raise ValueError(f"TailBCE counts must contain one positive count per class, got {values!r} for nc={nc}.")
    values_t = torch.tensor([float(value) for value in values], dtype=torch.float32, device=device)
    weights = (values_t.max() / values_t).pow(float(exponent)).clamp(min=1.0, max=float(cap))
    return weights


def build_legacy_prog_weights(counts: Iterable[float] | dict[int, float] | None, nc: int,
                              exponent: float = 0.5, device: torch.device | None = None) -> torch.Tensor:
    """Reproduce the old mean-normalized ProgLoss weights for the L1 control only."""
    if counts is None:
        raise ValueError("coffee_l1 requires current dataset class counts.")
    values = [counts.get(index, 0) for index in range(nc)] if isinstance(counts, dict) else list(counts)
    if len(values) != nc or any(float(value) <= 0 for value in values):
        raise ValueError(f"coffee_l1 counts must contain one positive count per class, got {values!r} for nc={nc}.")
    values_t = torch.tensor([float(value) for value in values], dtype=torch.float32, device=device)
    raw = (values_t.max() / values_t).pow(float(exponent))
    return (raw / raw.mean()).clamp(min=0.75, max=1.8)


def wiou_box_loss(pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, ciou_loss: torch.Tensor,
                  running_mean: torch.Tensor, config: dict[str, Any], training: bool) -> torch.Tensor:
    """Compute bounded non-monotonic WIoU with detached focus statistics."""
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, -1)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_min(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_min(0)
    area1 = (b1_x2 - b1_x1).clamp_min(0) * (b1_y2 - b1_y1).clamp_min(0)
    area2 = (b2_x2 - b2_x1).clamp_min(0) * (b2_y2 - b2_y1).clamp_min(0)
    iou = inter / (area1 + area2 - inter + 1e-7)
    base = (1.0 - iou).clamp(0.0, 2.0)
    with torch.no_grad():
        mean = base.detach().mean().clamp_min(1e-7)
        if training:
            momentum = float(config.get("wiou_momentum", 0.0001))
            running_mean.mul_(1.0 - momentum).add_(mean * momentum)
        beta = (base.detach() / running_mean.to(base).clamp_min(1e-7)).clamp(1e-7, 10.0)
        alpha = float(config.get("wiou_alpha", 1.7))
        delta = float(config.get("wiou_delta", 2.7))
        focus = beta / (delta * torch.pow(torch.tensor(alpha, device=base.device, dtype=base.dtype), beta - delta))
        focus = focus.clamp(float(config.get("wiou_focus_min", 0.5)), float(config.get("wiou_focus_max", 3.0)))
        c_x1, c_y1 = b1_x1.minimum(b2_x1), b1_y1.minimum(b2_y1)
        c_x2, c_y2 = b1_x2.maximum(b2_x2), b1_y2.maximum(b2_y2)
        c2 = (c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2) + 1e-7
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
        distance = torch.exp((rho2 / c2).clamp(0.0, math.log(float(config.get("wiou_distance_gain_max", 1.8)))))
    return base * focus * distance


def pair_margin_loss(logits: torch.Tensor, target_scores: torch.Tensor, fg_mask: torch.Tensor,
                     pairs: list[tuple[int, int]], counts: Iterable[float] | dict[int, float] | None,
                     config: dict[str, Any], progress: float) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute quality-gated pair margins from raw class logits."""
    zero = logits.new_zeros(())
    if not config.get("enabled", False) or not pairs:
        return zero, {"eligible": 0, "violations": 0, "loss": 0.0, "pairs": {}}
    q, true_ids = target_scores.max(dim=-1)
    eligible = fg_mask & (q >= float(config.get("quality_threshold", 0.30)))
    if isinstance(counts, dict):
        counts_list = [float(counts.get(index, 0)) for index in range(logits.shape[-1])]
    else:
        counts_list = [float(value) for value in counts] if counts is not None else [1.0] * logits.shape[-1]
    terms, weights, violations = [], [], 0
    pair_diagnostics: dict[str, dict[str, float | int]] = {}
    base = float(config.get("margin_base", 0.15))
    kappa = float(config.get("margin_kappa", 0.05))
    minimum, maximum = float(config.get("margin_min", 0.15)), float(config.get("margin_max", 0.30))
    beta = float(config.get("softplus_beta", 10.0))
    for true_id, rival_id in pairs:
        mask = eligible & (true_ids == true_id)
        if not mask.any():
            pair_diagnostics[f"{true_id}:{rival_id}"] = {"eligible": 0, "violations": 0, "loss": 0.0}
            continue
        margin = base + kappa * math.log((counts_list[rival_id] + 1e-7) / (counts_list[true_id] + 1e-7))
        margin = min(max(margin, minimum), maximum)
        difference = logits[..., true_id] - logits[..., rival_id]
        violation = difference[mask] < margin
        violations += int(violation.sum().item())
        pair_term = q[mask] * F.softplus(beta * (margin - difference[mask])) / beta
        pair_weight = q[mask]
        terms.append(pair_term)
        weights.append(pair_weight)
        pair_diagnostics[f"{true_id}:{rival_id}"] = {
            "eligible": int(mask.sum().item()),
            "violations": int(violation.sum().item()),
            "loss": float(
                (pair_term.sum(dtype=torch.float32) / pair_weight.sum(dtype=torch.float32).clamp_min(1e-7))
                .detach()
                .cpu()
            ),
        }
    if not terms:
        return zero, {"eligible": int(eligible.sum().item()), "violations": 0, "loss": 0.0, "pairs": pair_diagnostics}
    gain = float(config.get("lambda_max", 0.20)) * pair_ramp(
        progress, float(config.get("ramp_start", 0.10)), float(config.get("ramp_end", 0.60))
    )
    term_values = torch.cat(terms)
    term_weights = torch.cat(weights)
    value = term_values.sum(dtype=torch.float32) / term_weights.sum(dtype=torch.float32).clamp_min(1e-7)
    value = value * gain
    for diagnostics in pair_diagnostics.values():
        diagnostics["loss"] = float(diagnostics["loss"]) * gain
    return value, {
        "eligible": int(eligible.sum().item()),
        "violations": violations,
        "loss": float(value.detach().cpu()),
        "pairs": pair_diagnostics,
    }
