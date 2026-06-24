"""Train YOLO26n-EdgeLite-SimAM-BiBridge on coffee_self_sum.

Usage:
    python edgelite_experiment/train_coffee_edgelite_simam_bibridge.py

Optional environment variables:
    EDGE_DATA=path/to/data.yaml
    EDGE_EPOCHS=300
    EDGE_IMGSZ=960
    EDGE_BATCH=64
    EDGE_WORKERS=8
    EDGE_SIMAM_BIBRIDGE_RUN_NAME=name
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


EXP_ROOT = Path(__file__).resolve().parent
ROOT = EXP_ROOT.parent
LOCAL_ULTRALYTICS = EXP_ROOT / "local_ultralytics"
CFG = EXP_ROOT / "configs" / "yolo26n_edgelite_simam_bibridge.yaml"
WEIGHTS = ROOT / "yolo26n.pt"
DATA = Path(os.environ.get("EDGE_DATA", ROOT / "coffee_self_sum" / "coffee_self_sum.yaml"))

sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO  # noqa: E402


def select_cuda_device() -> int:
    """Require CUDA so training does not silently fall back to CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Install a CUDA-enabled PyTorch build in this env before training."
        )
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return 0


def format_metric(results, key: str) -> str:
    """Safely format a metric from an Ultralytics results object."""
    value = results.results_dict.get(key)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def main():
    device = select_cuda_device()
    if not CFG.exists():
        raise FileNotFoundError(f"Model yaml not found: {CFG}")
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {DATA}")

    model = YOLO(str(CFG))
    if WEIGHTS.exists():
        model = model.load(str(WEIGHTS))
        print(f"Loaded pretrained weights: {WEIGHTS}")
    else:
        print("yolo26n.pt not found; training will start from YAML initialization.")
    model.info(verbose=False)

    results = model.train(
        data=str(DATA),
        epochs=env_int("EDGE_EPOCHS", 300),
        imgsz=env_int("EDGE_IMGSZ", 960),
        batch=env_int("EDGE_BATCH", 64),
        cache="ram",
        workers=env_int("EDGE_WORKERS", 8),
        device=device,
        project=str(ROOT / "runs" / "train"),
        name=os.environ.get("EDGE_SIMAM_BIBRIDGE_RUN_NAME", "yolo26n_edgelite_simam_bibridge_coffee_self_sum"),
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
    print(f"Best mAP50:    {format_metric(results, 'metrics/mAP50(B)')}")
    print(f"Best mAP50-95: {format_metric(results, 'metrics/mAP50-95(B)')}")
    print(f"Best weights:  {results.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
