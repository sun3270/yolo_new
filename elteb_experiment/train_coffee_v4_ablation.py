"""Train one V4 ablation variant with WIoU + ProgLoss.

Usage:
    python elteb_experiment/train_coffee_v4_ablation.py

Architecture selection:
    V4_ARCH=native
    V4_ARCH=edgelite
    V4_ARCH=edgelite_bibridge

Ablation selection:
    V4_VARIANT=baseline
    V4_VARIANT=clahe
    V4_VARIANT=laplacian
    V4_VARIANT=elteb
    V4_VARIANT=elteb_lite

The default is the latest EdgeLite-BiBridge baseline with WIoU + ProgLoss.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from build_preprocessed_dataset import DEFAULT_OUTPUT_ROOT, build_dataset


EXP_ROOT = Path(__file__).resolve().parent
ROOT = EXP_ROOT.parent
EDGE_ROOT = ROOT / "edgelite_experiment"
WIOU_ROOT = ROOT / "wiou_progloss_experiment"
LOCAL_ULTRALYTICS = EDGE_ROOT / "local_ultralytics"
WEIGHTS = ROOT / "yolo26n.pt"
SOURCE_DATA = Path(os.environ.get("V4_DATA", ROOT / "coffee_self_sum" / "coffee_self_sum.yaml"))
LOSS_CONFIG = WIOU_ROOT / "loss_config.yaml"

BASE_CFGS = {
    "native": EDGE_ROOT / "configs" / "yolo26n_original_copy.yaml",
    "edgelite": EDGE_ROOT / "configs" / "yolo26n_edgelite.yaml",
    "edgelite_bibridge": EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
}

ELTEB_CFGS = {
    ("native", "elteb"): EXP_ROOT / "configs" / "yolo26n_native_elteb.yaml",
    ("native", "elteb_lite"): EXP_ROOT / "configs" / "yolo26n_native_elteb_lite.yaml",
    ("edgelite", "elteb"): EXP_ROOT / "configs" / "yolo26n_edgelite_elteb.yaml",
    ("edgelite", "elteb_lite"): EXP_ROOT / "configs" / "yolo26n_edgelite_elteb_lite.yaml",
    ("edgelite_bibridge", "elteb"): EXP_ROOT / "configs" / "yolo26n_edgelite_bibridge_elteb.yaml",
    ("edgelite_bibridge", "elteb_lite"): EXP_ROOT / "configs" / "yolo26n_edgelite_bibridge_elteb_lite.yaml",
}

PREPROCESS_VARIANTS = {"clahe", "laplacian"}
ALL_VARIANTS = {"baseline", "clahe", "laplacian", "elteb", "elteb_lite"}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402

from wiou_progloss_experiment.dataset_class_counts import count_train_labels, format_class_counts  # noqa: E402
from wiou_progloss_experiment.wiou_progloss_loss import (  # noqa: E402
    DEFAULT_LOSS_CONFIG,
    configure_wiou_progloss,
    patch_detection_model_loss,
)


def select_cuda_device() -> int:
    """Require CUDA so training does not silently fall back to CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Install a CUDA-enabled PyTorch build in this env before training."
        )
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return 0


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def format_metric(results, key: str) -> str:
    """Safely format a metric from an Ultralytics results object."""
    value = results.results_dict.get(key)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def resolve_variant(arch: str, variant_name: str) -> tuple[Path, str]:
    """Return model YAML and run name for the selected V4 ablation."""
    if arch not in BASE_CFGS:
        raise ValueError(f"Unknown V4_ARCH={arch!r}. Choose one of {sorted(BASE_CFGS)}.")
    if variant_name not in ALL_VARIANTS:
        raise ValueError(f"Unknown V4_VARIANT={variant_name!r}. Choose one of {sorted(ALL_VARIANTS)}.")
    if variant_name in {"elteb", "elteb_lite"}:
        cfg = ELTEB_CFGS[(arch, variant_name)]
    else:
        cfg = BASE_CFGS[arch]
    return cfg, f"v4_{arch}_wiou_progloss_{variant_name}_coffee_self_sum"


def resolve_variant_data(variant_name: str, overwrite_preprocess: bool) -> Path:
    """Return the dataset YAML for a V4 variant, building preprocessing data when needed."""
    if variant_name not in PREPROCESS_VARIANTS:
        return SOURCE_DATA
    output_root = Path(os.environ.get("V4_PREPROCESS_OUTPUT", DEFAULT_OUTPUT_ROOT))
    return build_dataset(variant_name, SOURCE_DATA, output_root, overwrite=overwrite_preprocess)


def configure_loss(data: Path) -> None:
    loss_config = YAML.load(LOSS_CONFIG) if LOSS_CONFIG.exists() else {}
    loss_overrides = {k: loss_config.get(k) for k in DEFAULT_LOSS_CONFIG if k in loss_config}
    class_counts = count_train_labels(data)
    loss_overrides["progloss_class_counts"] = class_counts
    print(f"ProgLoss train class counts: {class_counts}")
    print(f"ProgLoss class count detail: {format_class_counts(data, class_counts)}")
    configure_wiou_progloss(**loss_overrides)
    patch_detection_model_loss()


def main():
    """Train the selected V4 ablation variant."""
    arch = os.environ.get("V4_ARCH", "edgelite_bibridge").lower()
    variant_name = os.environ.get("V4_VARIANT", "baseline").lower()
    cfg, default_run_name = resolve_variant(arch, variant_name)
    if not cfg.exists():
        raise FileNotFoundError(f"Model yaml not found: {cfg}")

    device = select_cuda_device()
    overwrite_preprocess = os.environ.get("V4_PREPROCESS_OVERWRITE", "0") == "1"
    data = resolve_variant_data(variant_name, overwrite_preprocess)
    if not data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data}")
    configure_loss(data)

    run_name = os.environ.get("V4_RUN_NAME", default_run_name)
    print(f"Active architecture: {arch}")
    print(f"Ablation variant: {variant_name}")
    print(f"Model YAML: {cfg}")
    print(f"Data YAML: {data}")

    model = YOLO(str(cfg))
    if WEIGHTS.exists():
        model = model.load(str(WEIGHTS))
        print(f"Loaded pretrained weights: {WEIGHTS}")
    else:
        print("yolo26n.pt not found; training will start from YAML initialization.")
    model.info(verbose=False)

    results = model.train(
        data=str(data),
        epochs=env_int("V4_EPOCHS", 300),
        imgsz=env_int("V4_IMGSZ", 960),
        batch=env_int("V4_BATCH", 64),
        cache="ram",
        workers=env_int("V4_WORKERS", 8),
        device=device,
        project=str(ROOT / "runs" / "train"),
        name=run_name,
        amp=True,
        patience=30,
        save_period=20,
        verbose=True,
        deterministic=False,
        close_mosaic=10,
        lr0=0.012,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=4.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\n=== Training complete ===")
    print(f"Architecture:  {arch}")
    print(f"Variant:       {variant_name}")
    print(f"Data YAML:     {data}")
    print(f"Best mAP50:    {format_metric(results, 'metrics/mAP50(B)')}")
    print(f"Best mAP50-95: {format_metric(results, 'metrics/mAP50-95(B)')}")
    print(f"Best weights:  {results.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
