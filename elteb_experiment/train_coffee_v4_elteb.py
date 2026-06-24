"""Train YOLO26n ELTEB variants.

Usage:
    python elteb_experiment/train_coffee_v4_elteb.py

Variant selection:
    V4_ELTEB_ARCH=native
    V4_ELTEB_ARCH=edgelite
    V4_ELTEB_ARCH=edgelite_bibridge

    V4_ELTEB_VARIANT=elteb
    V4_ELTEB_VARIANT=elteb_lite

Optional loss:
    V4_ELTEB_LOSS=native
    V4_ELTEB_LOSS=wiou_progloss

The default is the latest EdgeLite-BiBridge + ELTEB structure with native loss.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


EXP_ROOT = Path(__file__).resolve().parent
ROOT = EXP_ROOT.parent
EDGE_ROOT = ROOT / "edgelite_experiment"
WIOU_ROOT = ROOT / "wiou_progloss_experiment"
LOCAL_ULTRALYTICS = EDGE_ROOT / "local_ultralytics"
WEIGHTS = ROOT / "yolo26n.pt"
LOSS_CONFIG = WIOU_ROOT / "loss_config.yaml"

VARIANTS = {
    ("native", "elteb"): {
        "cfg": EXP_ROOT / "configs" / "yolo26n_native_elteb.yaml",
        "name": "v4_native_elteb_coffee_self_sum",
    },
    ("native", "elteb_lite"): {
        "cfg": EXP_ROOT / "configs" / "yolo26n_native_elteb_lite.yaml",
        "name": "v4_native_elteb_lite_coffee_self_sum",
    },
    ("edgelite", "elteb"): {
        "cfg": EXP_ROOT / "configs" / "yolo26n_edgelite_elteb.yaml",
        "name": "v4_edgelite_elteb_coffee_self_sum",
    },
    ("edgelite", "elteb_lite"): {
        "cfg": EXP_ROOT / "configs" / "yolo26n_edgelite_elteb_lite.yaml",
        "name": "v4_edgelite_elteb_lite_coffee_self_sum",
    },
    ("edgelite_bibridge", "elteb"): {
        "cfg": EXP_ROOT / "configs" / "yolo26n_edgelite_bibridge_elteb.yaml",
        "name": "v4_edgelite_bibridge_elteb_coffee_self_sum",
    },
    ("edgelite_bibridge", "elteb_lite"): {
        "cfg": EXP_ROOT / "configs" / "yolo26n_edgelite_bibridge_elteb_lite.yaml",
        "name": "v4_edgelite_bibridge_elteb_lite_coffee_self_sum",
    },
}


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


def resolve_variant() -> tuple[str, str, dict]:
    arch = os.environ.get("V4_ELTEB_ARCH", "edgelite_bibridge").lower()
    branch = os.environ.get("V4_ELTEB_VARIANT", "elteb").lower()
    key = (arch, branch)
    if key not in VARIANTS:
        valid = [f"{a}+{b}" for a, b in sorted(VARIANTS)]
        raise ValueError(f"Unknown V4_ELTEB_ARCH/V4_ELTEB_VARIANT={key!r}. Choose one of {valid}.")
    return arch, branch, VARIANTS[key]


def configure_optional_loss(loss_name: str, data: Path) -> None:
    if loss_name == "native":
        return
    if loss_name != "wiou_progloss":
        raise ValueError("V4_ELTEB_LOSS must be 'native' or 'wiou_progloss'.")
    loss_config = YAML.load(LOSS_CONFIG) if LOSS_CONFIG.exists() else {}
    loss_overrides = {k: loss_config.get(k) for k in DEFAULT_LOSS_CONFIG if k in loss_config}
    class_counts = count_train_labels(data)
    loss_overrides["progloss_class_counts"] = class_counts
    print(f"ProgLoss train class counts: {class_counts}")
    print(f"ProgLoss class count detail: {format_class_counts(data, class_counts)}")
    configure_wiou_progloss(**loss_overrides)
    patch_detection_model_loss()


def main():
    arch, branch, variant = resolve_variant()
    loss_name = os.environ.get("V4_ELTEB_LOSS", "native").lower()
    data = Path(os.environ.get("V4_ELTEB_DATA", ROOT / "coffee_self_sum" / "coffee_self_sum.yaml"))
    if not data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data}")
    cfg = Path(variant["cfg"])
    if not cfg.exists():
        raise FileNotFoundError(f"Model yaml not found: {cfg}")

    device = select_cuda_device()
    configure_optional_loss(loss_name, data)

    run_name = os.environ.get("V4_ELTEB_RUN_NAME", str(variant["name"]) if loss_name == "native" else f"{variant['name']}_{loss_name}")
    print(f"Active architecture: {arch}")
    print(f"ELTEB variant: {branch}")
    print(f"Loss: {loss_name}")
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
        epochs=env_int("V4_ELTEB_EPOCHS", 300),
        imgsz=env_int("V4_ELTEB_IMGSZ", 960),
        batch=env_int("V4_ELTEB_BATCH", 64),
        cache="ram",
        workers=env_int("V4_ELTEB_WORKERS", 8),
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
    print(f"ELTEB variant: {branch}")
    print(f"Loss:          {loss_name}")
    print(f"Best mAP50:    {format_metric(results, 'metrics/mAP50(B)')}")
    print(f"Best mAP50-95: {format_metric(results, 'metrics/mAP50-95(B)')}")
    print(f"Best weights:  {results.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
