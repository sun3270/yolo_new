"""Train YOLO26n-EdgeLite with WIoU + ProgLoss on coffee3000.

Usage:
    python wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py

Optional smoke test:
    set WIOU_PROGLOSS_EPOCHS=3
    python wiou_progloss_experiment/train_coffee_edgelite_wiou_progloss.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


EXP_ROOT = Path(__file__).resolve().parent
ROOT = EXP_ROOT.parent
EDGE_ROOT = ROOT / "edgelite_experiment"
LOCAL_ULTRALYTICS = EDGE_ROOT / "local_ultralytics"
CFG = EDGE_ROOT / "configs" / "yolo26n_edgelite.yaml"
WEIGHTS = ROOT / "yolo26n.pt"
DATA = ROOT / "coffee3000" / "coffee3000.yaml"
CLASS_COUNTS = [501, 606, 332, 618, 709, 163]

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


def main():
    device = select_cuda_device()
    epochs = int(os.environ.get("WIOU_PROGLOSS_EPOCHS", "200"))

    model = YOLO(str(CFG))
    if WEIGHTS.exists():
        model = model.load(str(WEIGHTS))
        print(f"Loaded pretrained weights: {WEIGHTS}")
    else:
        print("yolo26n.pt not found; training will start from YAML initialization.")
    model.info(verbose=False)

    results = model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=640,
        batch=32,
        cache="ram",
        workers=2,
        device=device,
        project=str(ROOT / "runs" / "train"),
        name="yolo26n_edgelite_wiou_progloss_coffee3000",
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
        loss_name="wiou_progloss",
        wiou_enabled=True,
        wiou_alpha=1.7,
        wiou_delta=2.7,
        wiou_momentum=0.0001,
        wiou_focus_min=0.5,
        wiou_focus_max=3.0,
        wiou_use_distance_gain=True,
        wiou_distance_gain_max=1.8,
        wiou_fallback_base="raw_iou",
        progloss_enabled=True,
        progloss_warmup_ratio=0.10,
        progloss_ramp_end_ratio=0.60,
        progloss_tail_power=0.5,
        progloss_tail_lambda_max=0.8,
        progloss_tail_weight_min=0.75,
        progloss_tail_weight_max=1.8,
        progloss_class_counts=CLASS_COUNTS,
    )

    print("\n=== Training complete ===")
    print(f"Best mAP50:    {format_metric(results, 'metrics/mAP50(B)')}")
    print(f"Best mAP50-95: {format_metric(results, 'metrics/mAP50-95(B)')}")
    print(f"Best weights:  {results.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
