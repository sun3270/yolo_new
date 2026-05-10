"""WIoU + ProgLoss criterion with teacher distillation for P5Slim-512.

The base WIoU/ProgLoss math is reused from `wiou_progloss_experiment`.
This file adds a teacher branch without editing the verified local
Ultralytics package.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from wiou_progloss_experiment.loss_extensions import smooth_progress
from wiou_progloss_experiment.wiou_progloss_loss import (
    DEFAULT_LOSS_CONFIG,
    E2EWIoUProgLoss,
    v8DetectionWIoUProgLoss,
)


DEFAULT_DISTILL_CONFIG = {
    **DEFAULT_LOSS_CONFIG,
    "distill_enabled": True,
    "distill_start_ratio": 0.00,
    "distill_ramp_end_ratio": 0.20,
    "distill_temperature": 2.0,
    "distill_score_weight": 0.35,
    "distill_box_weight": 0.20,
    "distill_feature_weight": 0.08,
    "distill_conf_threshold": 0.20,
    "distill_topk": 600,
    "distill_feature_levels": [0, 1],
}

ACTIVE_DISTILL_CONFIG = DEFAULT_DISTILL_CONFIG.copy()
TEACHER_MODEL = None


def configure_distillation(teacher_model=None, **overrides: Any) -> None:
    """Set distillation options and optional teacher model for this process."""
    global TEACHER_MODEL
    ACTIVE_DISTILL_CONFIG.clear()
    ACTIVE_DISTILL_CONFIG.update(DEFAULT_DISTILL_CONFIG)
    ACTIVE_DISTILL_CONFIG.update({k: v for k, v in overrides.items() if v is not None})
    if teacher_model is not None:
        TEACHER_MODEL = teacher_model


def set_teacher_model(teacher_model) -> None:
    """Attach a frozen teacher model used by the patched criterion."""
    global TEACHER_MODEL
    TEACHER_MODEL = teacher_model


def _parse_output(preds: Any) -> dict[str, Any]:
    return preds[1] if isinstance(preds, tuple) else preds


def _branch(preds: dict[str, Any], name: str = "one2one") -> dict[str, torch.Tensor]:
    return preds[name] if name in preds else preds


def _distill_gain(cfg: dict[str, Any], progress: float) -> float:
    if not cfg.get("distill_enabled", True):
        return 0.0
    return smooth_progress(
        progress,
        float(cfg.get("distill_start_ratio", 0.0)),
        float(cfg.get("distill_ramp_end_ratio", 0.2)),
    )


def _teacher_anchor_mask(teacher_scores: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Build a BxA mask from teacher confidence for dense output distillation."""
    conf = teacher_scores.sigmoid().amax(dim=1)
    mask = conf >= float(cfg.get("distill_conf_threshold", 0.2))
    topk = int(cfg.get("distill_topk", 0) or 0)
    if topk > 0:
        k = min(topk, conf.shape[-1])
        topk_mask = torch.zeros_like(mask)
        topk_idx = conf.topk(k, dim=-1).indices
        topk_mask.scatter_(1, topk_idx, True)
        mask = mask | topk_mask
    if not mask.any():
        mask = conf >= conf.amax(dim=-1, keepdim=True)
    return mask


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=loss.device, dtype=loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(1)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def _dense_output_distill(
    student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor], cfg: dict[str, Any]
) -> torch.Tensor:
    """Distill class logits and raw box outputs on teacher-selected anchors."""
    student_scores = student["scores"]
    teacher_scores = teacher["scores"].detach().to(device=student_scores.device, dtype=student_scores.dtype)
    student_boxes = student["boxes"]
    teacher_boxes = teacher["boxes"].detach().to(device=student_boxes.device, dtype=student_boxes.dtype)

    mask = _teacher_anchor_mask(teacher_scores, cfg)
    temperature = float(cfg.get("distill_temperature", 2.0))
    soft_targets = torch.sigmoid(teacher_scores / temperature)
    score_loss = F.binary_cross_entropy_with_logits(
        student_scores / temperature, soft_targets, reduction="none"
    )
    score_loss = _masked_mean(score_loss, mask) * (temperature * temperature)

    box_loss = F.smooth_l1_loss(student_boxes, teacher_boxes, reduction="none")
    box_loss = _masked_mean(box_loss, mask)

    return float(cfg.get("distill_score_weight", 0.35)) * score_loss + float(
        cfg.get("distill_box_weight", 0.20)
    ) * box_loss


def _feature_distill(student: dict[str, Any], teacher: dict[str, Any], cfg: dict[str, Any]) -> torch.Tensor:
    """Distill same-shaped feature maps, usually P3/P4 for P5Slim."""
    student_feats = student.get("feats", [])
    teacher_feats = teacher.get("feats", [])
    levels = cfg.get("distill_feature_levels", [0, 1])
    if isinstance(levels, str):
        levels = [int(x.strip()) for x in levels.strip("[]").split(",") if x.strip()]
    losses = []
    for i in levels:
        if i >= len(student_feats) or i >= len(teacher_feats):
            continue
        s_feat = student_feats[i]
        t_feat = teacher_feats[i].detach().to(device=s_feat.device, dtype=s_feat.dtype)
        if s_feat.shape != t_feat.shape:
            continue
        losses.append(F.mse_loss(F.normalize(s_feat, dim=1), F.normalize(t_feat, dim=1)))
    if not losses:
        device = student["scores"].device
        return torch.zeros((), device=device)
    return torch.stack(losses).mean() * float(cfg.get("distill_feature_weight", 0.08))


class v8DetectionWIoUProgDistillLoss(v8DetectionWIoUProgLoss):
    """WIoU/ProgLoss branch with optional teacher guidance."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.distill_cfg = ACTIVE_DISTILL_CONFIG.copy()

    def distill_loss(self, student_preds: dict[str, Any], teacher_preds: dict[str, Any]) -> torch.Tensor:
        gain = _distill_gain(self.distill_cfg, self.loss_progress)
        if gain <= 0.0:
            return torch.zeros((), device=self.device)
        student_branch = _branch(student_preds, "one2one")
        teacher_branch = _branch(teacher_preds, "one2one")
        dense_loss = _dense_output_distill(student_branch, teacher_branch, self.distill_cfg)
        feature_loss = _feature_distill(student_branch, teacher_branch, self.distill_cfg)
        return (dense_loss + feature_loss) * gain


class E2EWIoUProgDistillLoss(E2EWIoUProgLoss):
    """End-to-end WIoU/ProgLoss plus teacher-student distillation."""

    def __init__(self, model, loss_fn=v8DetectionWIoUProgDistillLoss):
        super().__init__(model, loss_fn)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        base_loss, loss_items = super().__call__(preds, batch)
        if TEACHER_MODEL is None or not torch.is_grad_enabled():
            return base_loss, loss_items

        student_preds = _parse_output(preds)
        with torch.no_grad():
            teacher_preds = _parse_output(TEACHER_MODEL(batch["img"]))
        distill_loss = self.one2one.distill_loss(student_preds, teacher_preds)
        batch_size = student_preds["one2one"]["scores"].shape[0]
        return base_loss + distill_loss * batch_size, loss_items


def patch_detection_model_loss() -> None:
    """Patch DetectionModel to use the distillation criterion in this process."""
    from ultralytics.nn.tasks import DetectionModel  # noqa: PLC0415

    def init_criterion(self):
        if getattr(self, "end2end", False):
            return E2EWIoUProgDistillLoss(self)
        return v8DetectionWIoUProgDistillLoss(self)

    DetectionModel.init_criterion = init_criterion

