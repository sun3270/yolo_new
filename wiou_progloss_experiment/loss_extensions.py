"""WIoU + ProgLoss helpers for the EdgeLite loss experiment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def get_arg(hyp: Any, name: str, default: Any) -> Any:
    """Read an optional hyperparameter from an Ultralytics args namespace."""
    if isinstance(hyp, dict):
        return hyp.get(name, default)
    return getattr(hyp, name, default) if hyp is not None else default


def smooth_progress(progress: float, start: float, end: float) -> float:
    """Return a smooth 0-1 ramp for progressive loss scheduling."""
    if end <= start:
        return 1.0 if progress >= end else 0.0
    x = min(max((progress - start) / (end - start), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def progress_blend(hyp: Any, progress: float) -> float:
    """Return the active WIoU blend amount under the ProgLoss schedule."""
    if not get_arg(hyp, "progloss_enabled", False):
        return 1.0
    return smooth_progress(
        progress,
        float(get_arg(hyp, "progloss_warmup_ratio", 0.10)),
        float(get_arg(hyp, "progloss_ramp_end_ratio", 0.60)),
    )


def xyxy_iou_distance_gain(
    box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw IoU and WIoU distance gain for aligned xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    raw_iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    distance_gain = torch.exp((rho2 / c2).clamp(min=0.0))
    return raw_iou, distance_gain


def normalized_wasserstein_loss(box1: torch.Tensor, box2: torch.Tensor, constant: float = 12.8, eps: float = 1e-7) -> torch.Tensor:
    """Return NWD loss for aligned xyxy boxes, useful for tiny-box sensitivity checks."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    b1_w, b1_h = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
    b2_w, b2_h = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)
    wasserstein = (b1_cx - b2_cx).pow(2) + (b1_cy - b2_cy).pow(2) + ((b1_w - b2_w).pow(2) + (b1_h - b2_h).pow(2)) / 4
    return 1.0 - torch.exp(-torch.sqrt(wasserstein.clamp(min=eps)) / max(float(constant), eps))


def wiou_box_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    ciou_loss: torch.Tensor | None,
    running_mean: torch.Tensor,
    hyp: Any,
    training: bool,
) -> torch.Tensor:
    """Compute dynamic non-monotonic WIoU loss for aligned xyxy boxes."""
    raw_iou, distance_gain = xyxy_iou_distance_gain(pred_bboxes, target_bboxes)
    fallback_base = str(get_arg(hyp, "wiou_fallback_base", "raw_iou")).lower()
    if fallback_base == "nwd":
        base_loss = normalized_wasserstein_loss(
            pred_bboxes,
            target_bboxes,
            constant=float(get_arg(hyp, "nwd_constant", 12.8)),
        ).clamp(min=0.0, max=2.0)
    elif fallback_base == "ciou_focus" and ciou_loss is not None:
        base_loss = ciou_loss.clamp(min=0.0, max=2.0)
    else:
        base_loss = (1.0 - raw_iou).clamp(min=0.0, max=2.0)

    with torch.no_grad():
        batch_mean = base_loss.detach().mean().clamp(min=1e-7)
        momentum = float(get_arg(hyp, "wiou_momentum", 0.0001))
        if training:
            running_mean.mul_(1.0 - momentum).add_(batch_mean * momentum)
        mean = running_mean.to(device=base_loss.device, dtype=base_loss.dtype).clamp(min=1e-7)
        beta = (base_loss.detach() / mean).clamp(min=1e-7, max=10.0)
        alpha = torch.tensor(float(get_arg(hyp, "wiou_alpha", 1.7)), device=base_loss.device, dtype=base_loss.dtype)
        delta = float(get_arg(hyp, "wiou_delta", 2.7))
        focus = beta / (delta * torch.pow(alpha, beta - delta))
        focus = focus.clamp(float(get_arg(hyp, "wiou_focus_min", 0.5)), float(get_arg(hyp, "wiou_focus_max", 3.0)))
        if get_arg(hyp, "wiou_use_distance_gain", True):
            distance_gain = distance_gain.clamp(max=float(get_arg(hyp, "wiou_distance_gain_max", 1.8)))
        else:
            distance_gain = torch.ones_like(distance_gain)

    return base_loss * focus * distance_gain


def build_progloss_class_weights(hyp: Any, nc: int, device: torch.device) -> torch.Tensor:
    """Build clipped mean-normalized class weights from configured class counts."""
    counts = get_arg(hyp, "progloss_class_counts", None)
    if isinstance(counts, str):
        counts = [x.strip() for x in counts.strip("[]").split(",") if x.strip()]
    if not counts or len(counts) != nc:
        return torch.ones(nc, dtype=torch.float, device=device)
    counts = torch.tensor([float(x) for x in counts], dtype=torch.float, device=device).clamp(min=1.0)
    weights = (counts.max() / counts).pow(float(get_arg(hyp, "progloss_tail_power", 0.5)))
    weights = weights / weights.mean().clamp(min=1e-7)
    return weights.clamp(
        min=float(get_arg(hyp, "progloss_tail_weight_min", 0.75)),
        max=float(get_arg(hyp, "progloss_tail_weight_max", 1.8)),
    )


def progloss_gain(hyp: Any, progress: float) -> float:
    """Return the active tail-class weighting gain."""
    if not get_arg(hyp, "progloss_enabled", False):
        return 0.0
    return progress_blend(hyp, progress) * float(get_arg(hyp, "progloss_tail_lambda_max", 0.8))


def classification_loss(
    bce: nn.Module,
    pred_scores: torch.Tensor,
    target_scores: torch.Tensor,
    dtype: torch.dtype,
    class_weights: torch.Tensor,
    hyp: Any,
    progress: float,
) -> torch.Tensor:
    """Compute BCE classification loss with optional positive-only ProgLoss class weights."""
    target_scores = target_scores.to(dtype)
    cls_loss = bce(pred_scores, target_scores)
    gain = progloss_gain(hyp, progress)
    if gain > 0.0:
        active_weights = 1.0 + gain * (class_weights.to(device=pred_scores.device, dtype=dtype) - 1.0)
        positive_scale = 1.0 + (active_weights.view(1, 1, -1) - 1.0) * target_scores
        cls_loss *= positive_scale
    return cls_loss.sum()
