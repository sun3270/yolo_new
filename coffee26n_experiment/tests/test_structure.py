from __future__ import annotations

from pathlib import Path

import torch
import pytest

from coffee26n_experiment.config import MODEL_FILES
from coffee26n_experiment.model_utils import build_model, detect_metadata, finite_forward, parameter_budget, transfer_weights


ROOT = Path(__file__).resolve().parents[1]


def test_all_variants_build_and_keep_p3_p4_p5():
    for variant, filename in MODEL_FILES.items():
        model = build_model(ROOT / "configs" / "models" / filename, 1)
        metadata = detect_metadata(model)
        assert metadata["stride"] == [8.0, 16.0, 32.0]
        assert len(metadata["sources"]) == 3
        assert finite_forward(model, (64, 64))["tensor_count"] > 0


def test_dynamic_nc_and_rectangular_forward():
    for nc, variant in ((6, "core"), (80, "full")):
        model = build_model(ROOT / "configs" / "models" / MODEL_FILES[variant], nc)
        assert detect_metadata(model)["nc"] == nc
        result = finite_forward(model, (64, 96))
        assert result["tensor_count"] > 0


def test_full_variant_supports_training_batch_one():
    model = build_model(ROOT / "configs" / "models" / MODEL_FILES["full"], 5)
    model.train()
    output = model(torch.zeros(1, 3, 64, 64))
    tensors = []

    def collect(value):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(output)
    assert tensors and all(torch.isfinite(value).all() for value in tensors)


def test_parameter_budget_is_dynamic_and_below_yolo26s():
    cfg = ROOT / "configs" / "models" / "coffee26n_full.yaml"
    model = build_model(cfg, 6)
    budget = parameter_budget(cfg, 6, model=model)
    assert budget["hard_limit"] > budget["model_params"]
    assert budget["within_hard_limit"]
    assert budget["within_recommended_limit"]


def test_pretrained_transfer_is_auditable():
    model = build_model(ROOT / "configs" / "models" / "coffee26n_core.yaml", 6)
    report = transfer_weights(model, ROOT / "yolo26n.pt", "core", target_nc=detect_metadata(model)["nc"])
    assert report["target_numel_loaded"] > 0
    assert report["coverage_percent"] == 100.0
    assert report["expected_head_mismatches"]


def test_pretrained_transfer_covers_every_registered_variant():
    for variant, filename in MODEL_FILES.items():
        model = build_model(ROOT / "configs" / "models" / filename, 5)
        report = transfer_weights(model, ROOT / "yolo26n.pt", variant, target_nc=5)
        assert report["coverage_percent"] == 100.0, variant
        assert not report["missing_target_keys"], variant
        assert not report["skipped_shape_keys"], variant


def test_mixdown_odd_spatial_dimensions_match_native_downsample():
    from coffee26n_experiment.model_utils import setup_local_imports

    setup_local_imports()
    from ultralytics.nn.modules.block import MixDown

    output = MixDown(8, 8)(torch.randn(1, 8, 5, 7))
    assert tuple(output.shape[-2:]) == (3, 4)


def test_resbridge_rejects_wrong_p3_p5_ratio():
    from coffee26n_experiment.model_utils import setup_local_imports

    setup_local_imports()
    from ultralytics.nn.modules.block import ResBridge

    bridge = ResBridge(16, 32)
    with pytest.raises(ValueError, match="exactly 4x"):
        bridge((torch.randn(1, 16, 15, 16), torch.randn(1, 32, 4, 4)))


def test_resbridge_r0_r3_ablation_semantics():
    from coffee26n_experiment.model_utils import setup_local_imports

    setup_local_imports()
    from ultralytics.nn.modules.block import MonitoredRoleAwareAttnFuse2, ResBridge

    p3, p5 = torch.randn(2, 16, 16, 16), torch.randn(2, 32, 4, 4)
    variants = {
        "r0": (False, False),
        "r1": (True, False),
        "r2": (False, True),
        "r3": (True, True),
    }
    for residual, attention in variants.values():
        bridge = ResBridge(16, 32, residual=residual, role_attention=attention)
        output = bridge((p3, p5))
        assert output.shape == p5.shape
        assert isinstance(bridge.attn, MonitoredRoleAwareAttnFuse2) is attention
        if attention:
            assert bridge.attn.monitor_summary()["steps"] == 1
            bridge.eval()
            bridge((p3, p5))
            assert bridge.attn.monitor_summary()["steps"] == 1


def test_custom_module_repeat_is_fail_closed():
    from coffee26n_experiment.model_utils import setup_local_imports

    setup_local_imports()
    from ultralytics.nn.tasks import parse_model

    with pytest.raises(ValueError, match="repeat n=1"):
        parse_model({"nc": 1, "scale": "n", "backbone": [[-1, 2, "MixDown", [16]]], "head": []}, 3, verbose=False)


def test_transfer_covers_mixdown_base_branches_for_all_variants():
    from coffee26n_experiment.model_utils import transfer_weights

    expected = {"core": (19, 22), "p4mid": (19, 23), "p5lk13": (20, 23), "full": (20, 24)}
    for variant, layers in expected.items():
        model = build_model(ROOT / "configs" / "models" / MODEL_FILES[variant], 1)
        report = transfer_weights(model, ROOT / "yolo26n.pt", variant, target_nc=1)
        remapped = {(entry["target"].split(".")[1], entry["target"]) for entry in report["remapped_keys"]}
        for layer in layers:
            assert any(target_layer == str(layer) and ".base." in target_key for target_layer, target_key in remapped)
        assert report["eligible_native_target_numel"] >= report["target_numel_loaded"]
