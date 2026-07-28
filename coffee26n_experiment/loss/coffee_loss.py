"""CoffeeLoss adapter for native YOLO26 end-to-end detection."""

from __future__ import annotations

import functools
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, make_anchors

from .loss_extensions import (
    build_legacy_prog_weights,
    build_tail_weights,
    pair_margin_loss,
    tail_ramp,
    wiou_box_loss,
    wiou_ramp,
)


class CoffeeBboxLoss(BboxLoss):
    """Native BboxLoss with independent WIoU ramp and normalized L1 at reg_max=1."""

    def __init__(self, reg_max: int, config: dict[str, Any]):
        super().__init__(reg_max)
        self.config = config
        self.progress = 0.0
        self.register_buffer("wiou_running_mean", torch.tensor(1.0))
        self.last_terms = {"ciou": 0.0, "wiou": 0.0, "wiou_blend": 0.0, "regression": 0.0}

    def update_progress(self, progress: float) -> None:
        self.progress = min(max(float(progress), 0.0), 1.0)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg, target_fg = pred_bboxes[fg_mask], target_bboxes[fg_mask]
        ciou_loss = 1.0 - bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
        cfg = self.config
        ramp = wiou_ramp(self.progress, float(cfg.get("wiou_ramp_start", 0.10)), float(cfg.get("wiou_ramp_end", 0.60)))
        if cfg.get("wiou_enabled", True):
            wiou = wiou_box_loss(pred_fg, target_fg, ciou_loss, self.wiou_running_mean, cfg, self.training)
            box_loss = ciou_loss * (1.0 - ramp) + wiou * ramp
        else:
            wiou = ciou_loss
            box_loss = ciou_loss
        loss_iou = (box_loss * weight).sum() / target_scores_sum
        self.last_terms.update(
            ciou=float(((ciou_loss * weight).sum() / target_scores_sum).detach().cpu()),
            wiou=float(((wiou * weight).sum() / target_scores_sum).detach().cpu()),
            wiou_blend=float(ramp if cfg.get("wiou_enabled", True) else 0.0),
        )
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_reg = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_reg = loss_reg.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes) * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_norm = pred_dist * stride
            pred_norm[..., 0::2] /= imgsz[1]
            pred_norm[..., 1::2] /= imgsz[0]
            loss_reg = F.l1_loss(pred_norm[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            loss_reg = loss_reg.sum() / target_scores_sum
        self.last_terms["regression"] = float(loss_reg.detach().cpu())
        return loss_iou, loss_reg


class CoffeeLoss(v8DetectionLoss):
    """WIoU + TailBCE + quality-gated PairMargin with the native assignment contract."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, *,
                 profile: Mapping[str, Any], config: Mapping[str, Any]):
        super().__init__(model, tal_topk, tal_topk2)
        self.profile = dict(profile)
        self.config = dict(config)
        self.progress = 0.0
        self.loss_epoch = 0
        self.loss_name = str(self.config.get("loss_name", "coffee_l1"))
        tail_cfg = dict(self.profile.get("tail_bce", {}))
        if self.loss_name == "coffee_l2":
            tail_cfg.update(enabled=True, cap=1.8)
        elif self.loss_name in {"coffee_l3", "coffee_l4"}:
            tail_cfg.update(enabled=True, cap=2.4)
        elif self.loss_name == "coffee_l1":
            tail_cfg.update(enabled=False)
        else:
            raise ValueError(f"CoffeeLoss received unsupported loss_name={self.loss_name!r}.")
        self.tail_cfg = tail_cfg
        counts = self.config.get("class_counts") or tail_cfg.get("counts")
        self.class_counts = counts
        self.tail_weights = build_tail_weights(counts, self.nc, float(tail_cfg.get("exponent", 0.5)), float(tail_cfg.get("cap", 2.4)), bool(tail_cfg.get("enabled", False)), self.device)
        self.prog_weights = (
            build_legacy_prog_weights(counts, self.nc, float(tail_cfg.get("exponent", 0.5)), self.device)
            if self.loss_name == "coffee_l1"
            else torch.ones(self.nc, dtype=torch.float32, device=self.device)
        )
        self.bbox_loss = CoffeeBboxLoss(self.reg_max, {**self.config.get("loss", {}), "wiou_enabled": self.config.get("loss", {}).get("wiou_enabled", True)}).to(self.device)
        self.pair_cfg = dict(self.profile.get("pair_margin", {}))
        self.pair_cfg["enabled"] = self.loss_name == "coffee_l4" and bool(self.pair_cfg.get("enabled", False))
        self.pairs = [tuple(pair) for pair in self.config.get("pair_ids", [])] if self.loss_name == "coffee_l4" else []
        if self.loss_name == "coffee_l4" and not self.pair_cfg["enabled"]:
            raise ValueError("coffee_l4 requires PairMargin to be enabled; an empty auto-selected list is a valid no-op.")
        self.last_terms = {
            "base_bce": 0.0,
            "legacy_prog": 0.0,
            "tail_bce": 0.0,
            "pair_margin": 0.0,
            "norm_l1": 0.0,
            "ciou": 0.0,
            "wiou": 0.0,
            "wiou_blend": 0.0,
        }
        self.last_pair_diagnostics = {"eligible": 0, "violations": 0, "loss": 0.0, "pairs": {}}

    def update_progress(self, progress: float) -> None:
        self.progress = min(max(float(progress), 0.0), 1.0)
        self.bbox_loss.update_progress(self.progress)

    def update(self) -> None:
        self.loss_epoch += 1
        total_epochs = max(int(getattr(self.hyp, "epochs", 1)) - 1, 1)
        self.update_progress(self.loss_epoch / total_epochs)

    def _classification_loss(self, pred_scores, target_scores, dtype, fg_mask):
        target_scores = target_scores.to(dtype)
        base_bce = self.bce(pred_scores, target_scores)
        raw_bce = base_bce
        tail_cfg = self.tail_cfg
        tail_gain = tail_ramp(self.progress, float(tail_cfg.get("ramp_start", 0.10)), float(tail_cfg.get("ramp_end", 0.60)))
        if self.loss_name == "coffee_l1":
            active = 1.0 + 0.8 * tail_gain * (self.prog_weights.to(pred_scores) - 1.0)
            raw_bce = raw_bce * (1.0 + (active.view(1, 1, -1) - 1.0) * target_scores)
            self.last_terms["legacy_prog"] = float(
                (raw_bce.float() - base_bce.float()).detach().sum().cpu()
            )
        elif tail_cfg.get("enabled", False):
            active = 1.0 + tail_gain * (self.tail_weights.to(pred_scores) - 1.0)
            raw_bce = raw_bce * (1.0 + (active.view(1, 1, -1) - 1.0) * target_scores)
        pair_value, diagnostics = pair_margin_loss(pred_scores, target_scores, fg_mask, self.pairs, self.class_counts, self.pair_cfg, self.progress)
        self.last_pair_diagnostics = diagnostics
        base_sum = base_bce.sum(dtype=torch.float32)
        raw_sum = raw_bce.sum(dtype=torch.float32)
        self.last_terms["base_bce"] = float(base_sum.detach().cpu())
        self.last_terms["tail_bce"] = float((raw_sum - base_sum).detach().cpu()) if self.loss_name != "coffee_l1" else 0.0
        self.last_terms["pair_margin"] = diagnostics["loss"]
        return raw_sum, pair_value

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        self.bbox_loss.last_terms = {"ciou": 0.0, "wiou": 0.0, "wiou_blend": 0.0, "regression": 0.0}
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
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
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,
        )
        target_scores_sum = target_scores.sum(dtype=torch.float32).clamp_min(1.0)
        cls_raw, pair_value = self._classification_loss(pred_scores, target_scores, dtype, fg_mask)
        loss[1] = cls_raw / target_scores_sum + pair_value
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                               target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        self.last_terms["norm_l1"] = float(loss[2].detach().cpu())
        self.last_terms["ciou"] = self.bbox_loss.last_terms["ciou"]
        self.last_terms["wiou"] = self.bbox_loss.last_terms["wiou"]
        self.last_terms["wiou_blend"] = self.bbox_loss.last_terms["wiou_blend"]
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()


class CoffeeE2ELoss(E2ELoss):
    """Native YOLO26 E2E wrapper with identical CoffeeLoss branches and synced ramps."""

    def __init__(self, model, *, profile, config):
        factory = functools.partial(CoffeeLoss, profile=profile, config=config)
        super().__init__(model, loss_fn=factory)
        self.total_epochs = max(int(getattr(model.args, "epochs", 1)), 1)

    def update(self) -> None:
        super().update()
        progress = min(self.updates / max(self.total_epochs - 1, 1), 1.0)
        self.one2many.update_progress(progress)
        self.one2one.update_progress(progress)
