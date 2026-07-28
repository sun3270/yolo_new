from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from coffee26n_experiment.loss.loss_extensions import (
    build_legacy_prog_weights,
    build_tail_weights,
    pair_margin_loss,
    pair_ramp,
    tail_ramp,
    wiou_ramp,
)
from coffee26n_experiment.config import resolve_profile, resolve_profile_runtime
from coffee26n_experiment.model_utils import build_model, setup_local_imports
from coffee26n_experiment.loss.coffee_loss import CoffeeLoss


ROOT = Path(__file__).resolve().parents[1]


def test_ramps_are_independent_and_bounded():
    assert wiou_ramp(0.0) == 0.0
    assert tail_ramp(1.0) == 1.0
    assert pair_ramp(0.35) > 0.0
    assert 0.0 <= wiou_ramp(0.5) <= 1.0


def test_tail_weights_only_upweight_rare_classes():
    weights = build_tail_weights({0: 100, 1: 25}, 2, exponent=0.5, cap=1.8, enabled=True)
    assert torch.allclose(weights, torch.tensor([1.0, 1.8]))
    with pytest.raises(ValueError):
        build_tail_weights({0: 1, 1: 0}, 2, enabled=True)


def test_legacy_prog_control_keeps_mean_normalization():
    weights = build_legacy_prog_weights({0: 100, 1: 25}, 2)
    assert torch.allclose(weights, torch.tensor([0.75, 1.3333334]), atol=1e-6)


def test_pair_margin_quality_gate_and_gradient_direction():
    logits = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    target = torch.tensor([[[0.8, 0.1]]])
    fg = torch.tensor([[True]])
    value, info = pair_margin_loss(logits, target, fg, [(0, 1)], {0: 10, 1: 1}, {"enabled": True, "lambda_max": 1.0}, 1.0)
    assert value.item() > 0 and info["violations"] == 1
    value.backward()
    assert logits.grad[0, 0, 0] < 0
    assert logits.grad[0, 0, 1] > 0


def test_pair_margin_reports_each_pair_independently():
    logits = torch.tensor([[[0.0, 0.0, 0.0], [0.8, 0.1, 0.2]]])
    target = torch.tensor([[[0.9, 0.0, 0.0], [0.0, 0.8, 0.0]]])
    fg = torch.tensor([[True, True]])
    _, info = pair_margin_loss(
        logits,
        target,
        fg,
        [(0, 1), (1, 2)],
        {0: 100, 1: 20, 2: 5},
        {"enabled": True, "lambda_max": 1.0},
        1.0,
    )
    assert info["pairs"]["0:1"]["eligible"] == 1
    assert info["pairs"]["1:2"]["eligible"] == 1
    assert info["pairs"]["0:1"]["loss"] != info["pairs"]["1:2"]["loss"]


def test_pair_margin_empty_branch_is_finite_for_large_fp16_logits():
    logits = torch.full((96, 18900, 5), -0.01, dtype=torch.float16)
    target = torch.zeros_like(logits)
    fg = torch.zeros((96, 18900), dtype=torch.bool)
    value, info = pair_margin_loss(
        logits,
        target,
        fg,
        [(1, 0)],
        {0: 1246, 1: 505},
        {"enabled": True},
        1.0,
    )
    assert value.item() == 0.0
    assert torch.isfinite(value)
    assert info["eligible"] == 0


def test_coffee_classification_reduction_is_finite_for_real_val_shape_fp16():
    holder = SimpleNamespace(
        bce=torch.nn.BCEWithLogitsLoss(reduction="none"),
        loss_name="coffee_l3",
        tail_cfg={"enabled": False},
        pair_cfg={"enabled": False},
        pairs=[],
        class_counts={index: 1 for index in range(5)},
        progress=1.0,
        last_terms={},
        last_pair_diagnostics={},
    )
    logits = torch.full((96, 18900, 5), -0.01, dtype=torch.float16)
    targets = torch.zeros_like(logits)
    fg = torch.zeros((96, 18900), dtype=torch.bool)
    cls_raw, pair_value = CoffeeLoss._classification_loss(holder, logits, targets, logits.dtype, fg)
    assert torch.isfinite(cls_raw)
    assert torch.isfinite(pair_value)
    assert cls_raw.dtype == torch.float32


def test_coffee_e2e_branches_and_regmax_one_l1():
    setup_local_imports()
    counts = {index: index + 1 for index in range(6)}
    model = build_model(ROOT / "configs" / "models" / "coffee26n_core.yaml", 6, loss_name="coffee_l1", profile={"tail_bce": {"enabled": False}, "pair_margin": {"enabled": False}}, loss_config={"loss_name": "coffee_l1", "class_counts": counts, "pair_ids": [], "loss": {"wiou_enabled": True}})
    criterion = model.init_criterion()
    assert type(criterion.one2many).__name__ == "CoffeeLoss"
    assert criterion.one2many.bbox_loss.dfl_loss is None
    model.train()
    preds = model(torch.zeros(1, 3, 64, 64))
    batch = {"img": torch.zeros(1, 3, 64, 64), "batch_idx": torch.zeros(1, dtype=torch.long), "cls": torch.zeros(1), "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])}
    total, items = criterion(preds, batch)
    assert torch.isfinite(total).all() and torch.isfinite(items).all()
    total.sum().backward()


def test_loss_names_have_fixed_tail_and_pair_semantics():
    counts = {index: index + 1 for index in range(6)}
    profile = {"tail_bce": {"enabled": False}, "pair_margin": {"enabled": True, "pairs": [(0, 1)]}}
    for name, cap, pair_enabled in (("coffee_l2", 1.8, False), ("coffee_l3", 2.4, False), ("coffee_l4", 2.4, True)):
        model = build_model(ROOT / "configs" / "models" / "coffee26n_core.yaml", 6, loss_name=name, profile=profile, loss_config={"loss_name": name, "class_counts": counts, "pair_ids": [(0, 1)], "loss": {"wiou_enabled": True}})
        branch = model.init_criterion().one2many
        assert branch.tail_cfg["enabled"] and branch.tail_cfg["cap"] == cap
        assert branch.pair_cfg["enabled"] is pair_enabled


def test_e2e_pair_and_tail_configuration_reaches_both_branches():
    counts = {index: index + 1 for index in range(6)}
    model = build_model(ROOT / "configs" / "models" / "coffee26n_core.yaml", 6, loss_name="coffee_l4",
                        profile={"tail_bce": {"enabled": True}, "pair_margin": {"enabled": True}},
                        loss_config={"loss_name": "coffee_l4", "class_counts": counts, "pair_ids": [(0, 1)], "loss": {"wiou_enabled": True}})
    criterion = model.init_criterion()
    assert criterion.one2many.tail_cfg["enabled"] and criterion.one2one.tail_cfg["enabled"]
    assert criterion.one2many.pair_cfg["enabled"] and criterion.one2one.pair_cfg["enabled"]
    assert criterion.one2many.pairs == criterion.one2one.pairs == [(0, 1)]


def test_auto_full_statistics_reach_both_coffee_l4_branches():
    dataset = {"names": ["dominant", "medium", "rare"], "class_counts": {0: 1000, 1: 500, 2: 100}}
    profile, _ = resolve_profile("auto_full")
    runtime = resolve_profile_runtime(profile, dataset)
    model = build_model(
        ROOT / "configs" / "models" / "coffee26n_full.yaml",
        3,
        loss_name="coffee_l4",
        profile=runtime,
        loss_config={"loss_name": "coffee_l4", "class_counts": runtime["class_counts"], "pair_ids": runtime["pair_ids"], "loss": runtime["loss"]},
    )
    criterion = model.init_criterion()
    for branch in (criterion.one2many, criterion.one2one):
        assert branch.tail_cfg["enabled"]
        assert branch.pair_cfg["enabled"]
        assert branch.class_counts == {0: 1000.0, 1: 500.0, 2: 100.0}
        assert branch.pairs == [(1, 0), (2, 0)]


def test_regmax_greater_than_one_keeps_native_dfl(tmp_path):
    config = tmp_path / "regmax4.yaml"
    config.write_text((ROOT / "configs" / "models" / "coffee26n_core.yaml").read_text(encoding="utf-8").replace("reg_max: 1", "reg_max: 4"), encoding="utf-8")
    model = build_model(config, 1, loss_name="coffee_l1", profile={"tail_bce": {"enabled": False}, "pair_margin": {"enabled": False}},
                        loss_config={"loss_name": "coffee_l1", "class_counts": {0: 1}, "pair_ids": [], "loss": {"wiou_enabled": True}})
    assert model.init_criterion().one2many.bbox_loss.dfl_loss is not None
