from __future__ import annotations

import json
from pathlib import Path

import torch

from coffee26n_experiment.augmentations.bgmix import BatchCoffeeBgMix, _gaussian_blur, apply_bgmix
from coffee26n_experiment.manifest import verify_source_manifest
from coffee26n_experiment.model_utils import setup_local_imports


ROOT = Path(__file__).resolve().parents[1]


def test_bgmix_preserves_expanded_boxes_and_skips_eval():
    image = torch.linspace(0, 1, 3 * 32 * 32).reshape(3, 32, 32)
    boxes = torch.tensor([[8.0, 8.0, 24.0, 24.0]])
    output, info, preserve = apply_bgmix(image, boxes, p=1.0, box_expand=0.0, blur_kernel=3, force=True)
    assert info["status"] == "applied"
    assert torch.equal(output[:, preserve], image[:, preserve])
    eval_output, eval_info, _ = apply_bgmix(image, boxes, training=False)
    assert eval_info["status"] == "skipped_not_training"
    assert torch.equal(eval_output, image)


def test_bgmix_overlapping_boundary_boxes_and_empty_guard():
    image = torch.rand(3, 32, 32)
    boxes = torch.tensor([[-2.0, -1.0, 12.0, 12.0], [8.0, 8.0, 20.0, 20.0]])
    output, info, preserve = apply_bgmix(image, boxes, p=1.0, box_expand=0.05, feather_px=3, blur_kernel=3, force=True)
    assert info["status"] == "applied" and preserve[0, 0]
    assert torch.equal(output[:, preserve], image[:, preserve])
    empty_output, empty_info, empty_mask = apply_bgmix(image, torch.empty(0, 4), force=True)
    assert empty_info["status"] == "skipped_no_boxes"
    assert torch.equal(empty_output, image) and not empty_mask.any()


def test_bgmix_blur_is_separable_gaussian_not_box_average():
    image = torch.zeros(1, 9, 9)
    image[:, 4, 4] = 1.0
    blurred = _gaussian_blur(image, 5)
    assert torch.isclose(blurred.sum(), torch.tensor(1.0), atol=1e-6)
    assert blurred[0, 4, 4] > blurred[0, 4, 3] > blurred[0, 4, 2]
    assert not torch.isclose(blurred[0, 4, 4], blurred[0, 4, 3])


def test_import_isolation_and_manifest():
    paths = setup_local_imports()
    assert all("coffee26n_experiment\\local_ultralytics" in value for value in paths.values())
    assert not verify_source_manifest(ROOT)
    payload = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    assert payload["package_root"] == "coffee26n_experiment"
    assert payload["native_source_root"] == "ultralytics"
    assert all(not Path(entry["source_path"]).is_absolute() for entry in payload["files"])


def test_bgmix_is_inserted_before_format_only_for_training():
    setup_local_imports()
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset

    dataset = object.__new__(YOLODataset)
    dataset.augment = True
    dataset.rect = False
    dataset.imgsz = 64
    dataset.cache = None
    dataset.data = {}
    dataset.use_segments = dataset.use_keypoints = dataset.use_obb = False
    transforms = dataset.build_transforms(get_cfg(overrides={"coffee_bgmix": 0.5}))
    names = [type(transform).__name__ for transform in transforms.transforms]
    assert names[-2:] == ["CoffeeBgMix", "Format"]
    transforms = dataset.build_transforms(get_cfg(overrides={"coffee_bgmix": 0.5, "coffee_bgmix_mode": "batch_gpu"}))
    assert "CoffeeBgMix" not in [type(transform).__name__ for transform in transforms.transforms]
    dataset.augment = False
    names = [type(transform).__name__ for transform in dataset.build_transforms(get_cfg()).transforms]
    assert "CoffeeBgMix" not in names


def test_bgmix_stats_are_shared_and_resettable():
    setup_local_imports()
    from ultralytics.data.augment import CoffeeBgMix

    transform = CoffeeBgMix(1.0, 0.02, 4, 5)
    with transform._stats.get_lock():
        transform._stats[0] = 3
        transform._stats[1] = 2
        transform._stats[3] = 1.5
        transform._stats[4] = 3
        transform._stats[5] = 1
    stats = transform.snapshot_stats(reset=True)
    assert stats == {
        "applied": 3,
        "skipped_no_boxes": 2,
        "skipped_probability": 0,
        "preserve_ratio": 0.5,
        "last_transform": "blur",
    }
    assert transform.snapshot_stats()["applied"] == 0


def test_batch_gpu_bgmix_preserves_boxes_and_updates_stats():
    torch.manual_seed(0)
    images = torch.linspace(0, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    original = images.clone()
    batch_idx = torch.tensor([0, 1])
    boxes = torch.tensor([[0.50, 0.50, 0.50, 0.50], [0.25, 0.25, 0.25, 0.25]])
    transform = BatchCoffeeBgMix(1.0, 0.0, 3, 5)
    output = transform(images, batch_idx, boxes)
    assert torch.equal(output[0, :, 8:24, 8:24], original[0, :, 8:24, 8:24])
    assert torch.equal(output[1, :, 4:12, 4:12], original[1, :, 4:12, 4:12])
    assert not torch.equal(output, original)
    stats = transform.stats
    assert stats["applied"] == 2 and stats["skipped_no_boxes"] == 0
    transform.reset()
    assert transform.stats["applied"] == 0


def test_batch_gpu_bgmix_handles_amp_blur_dtype(monkeypatch):
    images = torch.linspace(0, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    original = images.clone()
    batch_idx = torch.tensor([0, 1])
    boxes = torch.tensor([[0.50, 0.50, 0.50, 0.50], [0.25, 0.25, 0.25, 0.25]])
    transform = BatchCoffeeBgMix(1.0, 0.0, 3, 5)
    monkeypatch.setattr(torch, "randint", lambda high, size: torch.zeros(size, dtype=torch.long))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = transform(images, batch_idx, boxes)

    assert output.dtype == original.dtype
    assert torch.equal(output[0, :, 8:24, 8:24], original[0, :, 8:24, 8:24])
    assert torch.equal(output[1, :, 4:12, 4:12], original[1, :, 4:12, 4:12])
    assert not torch.equal(output, original)
