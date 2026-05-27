"""Custom WIoU + ProgLoss criterion for the EdgeLite experiment.

This module keeps the experimental loss outside
`edgelite_experiment/local_ultralytics/ultralytics/utils/loss.py`, so the native
Ultralytics loss remains available for later comparisons.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, make_anchors
from wiou_progloss_experiment.loss_extensions import (
    build_progloss_class_weights,
    classification_loss,
    get_arg,
    progress_blend,
    wiou_box_loss,
)

DEFAULT_LOSS_CONFIG = {
    "wiou_enabled": True,
    "wiou_alpha": 1.7,
    "wiou_delta": 2.7,
    "wiou_momentum": 0.0001,
    "wiou_focus_min": 0.5,
    "wiou_focus_max": 3.0,
    "wiou_use_distance_gain": True,
    "wiou_distance_gain_max": 1.8,
    "wiou_fallback_base": "raw_iou",
    "progloss_enabled": True,
    "progloss_warmup_ratio": 0.10,
    "progloss_ramp_end_ratio": 0.60,
    "progloss_tail_power": 0.5,
    "progloss_tail_lambda_max": 0.8,
    "progloss_tail_weight_min": 0.75,
    "progloss_tail_weight_max": 1.8,
    "progloss_class_counts": [501, 606, 332, 618, 709, 163],
}
ACTIVE_LOSS_CONFIG = DEFAULT_LOSS_CONFIG.copy()


def configure_wiou_progloss(**overrides: Any) -> None:
    """Configure the active WIoU + ProgLoss experiment for the current process."""
    ACTIVE_LOSS_CONFIG.clear()
    ACTIVE_LOSS_CONFIG.update(DEFAULT_LOSS_CONFIG)
    ACTIVE_LOSS_CONFIG.update({k: v for k, v in overrides.items() if v is not None})


class BboxWIoUProgLoss(BboxLoss):
    """Bbox loss with optional WIoU dynamic non-monotonic focusing."""

    def __init__(self, reg_max: int = 16, cfg: dict[str, Any] | None = None):
        """Initialize WIoU state while keeping the native DFL/L1 branches."""
        super().__init__(reg_max)
        self.cfg = cfg or ACTIVE_LOSS_CONFIG
        self.progress = 0.0
        self.register_buffer("wiou_running_mean", torch.tensor(1.0))

    def set_progress(self, progress: float) -> None:
        """Set current epoch progress for progressive WIoU blending."""
        self.progress = min(max(float(progress), 0.0), 1.0)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute WIoU/CIoU and DFL/L1 losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg, target_fg = pred_bboxes[fg_mask], target_bboxes[fg_mask]
        iou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
        ciou_loss = 1.0 - iou
        if get_arg(self.cfg, "wiou_enabled", True):
            wiou_loss = wiou_box_loss(pred_fg, target_fg, ciou_loss, self.wiou_running_mean, self.cfg, self.training)
            blend = progress_blend(self.cfg, self.progress)
            box_loss = ciou_loss * (1.0 - blend) + wiou_loss * blend
        else:
            box_loss = ciou_loss
        loss_iou = (box_loss * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class v8DetectionWIoUProgLoss(v8DetectionLoss):
    """Detection loss with WIoU box loss and ProgLoss classification weighting."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        """Initialize native detection loss, then replace only the experimental components."""
        super().__init__(model, tal_topk, tal_topk2)
        self.exp_cfg = ACTIVE_LOSS_CONFIG.copy()
        self.loss_epoch = 0
        self.loss_progress = 0.0
        self.bbox_loss = BboxWIoUProgLoss(self.reg_max, self.exp_cfg).to(self.device)
        self.progloss_class_weights = build_progloss_class_weights(self.exp_cfg, self.nc, self.device)

    def update(self) -> None:
        """Advance progressive loss state after each completed epoch."""
        self.loss_epoch += 1
        self.loss_progress = min(self.loss_epoch / max(int(getattr(self.hyp, "epochs", 1)) - 1, 1), 1.0)
        self.bbox_loss.set_progress(self.loss_progress)

    def _classification_loss(
        self, pred_scores: torch.Tensor, target_scores: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Compute classification loss through the experiment helper."""
        return classification_loss(
            self.bce,
            pred_scores,
            target_scores,
            dtype,
            self.progloss_class_weights,
            self.exp_cfg,
            self.loss_progress,
        )

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Return assigned targets and loss with the experimental classification term."""
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self._classification_loss(pred_scores, target_scores, dtype) / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class E2EWIoUProgLoss(E2ELoss):
    """End-to-end detection loss wired to the WIoU + ProgLoss branch."""

    def __init__(self, model, loss_fn=v8DetectionWIoUProgLoss):
        """Initialize one-to-many and one-to-one experiment losses."""
        super().__init__(model, loss_fn)

    def update(self) -> None:
        """Update E2E branch weights and propagate epoch progress to both branches."""
        super().update()
        self.one2many.update()
        self.one2one.update()


def patch_detection_model_loss() -> None:
    """Patch DetectionModel in the current process to use the experiment loss."""
    from ultralytics.nn.tasks import DetectionModel

    def init_criterion(self):
        return E2EWIoUProgLoss(self) if getattr(self, "end2end", False) else v8DetectionWIoUProgLoss(self)

    DetectionModel.init_criterion = init_criterion
