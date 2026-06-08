"""Train YOLO26n-LGMSF-Lite on the coffee3000 dataset.

This script is intentionally self-contained for high-compute training. It uses
the isolated LGMSF-Lite Ultralytics copy under `lgmsf_lite_experiment/` and
does not require modifying the root `ultralytics/` package.

Usage:
    python train_coffee_lgmsf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
LGMSF_ROOT = ROOT / "lgmsf_lite_experiment"
LOCAL_ULTRALYTICS = LGMSF_ROOT / "local_ultralytics"
CFG = LGMSF_ROOT / "configs" / "yolo26n_lgmsf_lite.yaml"
WEIGHTS = ROOT / "yolo26n.pt"
DATA = ROOT / "coffee3000" / "coffee3000.yaml"

if not LOCAL_ULTRALYTICS.exists():
    raise FileNotFoundError(
        f"Missing LGMSF local Ultralytics package: {LOCAL_ULTRALYTICS}. "
        "Upload/copy the `lgmsf_lite_experiment` directory with this script."
    )
if not CFG.exists():
    raise FileNotFoundError(f"Missing LGMSF model YAML: {CFG}")

sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO  # noqa: E402


def select_cuda_device() -> int:
    """Require CUDA so this training run does not silently fall back to CPU."""
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

    print(f"Using LGMSF config: {CFG}")
    print(f"Using local Ultralytics: {LOCAL_ULTRALYTICS}")
    print(f"Using dataset: {DATA}")

    model = YOLO(str(CFG))
    if WEIGHTS.exists():
        model = model.load(str(WEIGHTS))
        print(f"Loaded pretrained weights: {WEIGHTS}")
    else:
        print("yolo26n.pt not found; training will start from YAML initialization.")
    model.info(verbose=False)

    results = model.train(
        data=str(DATA),
        epochs=200,
        imgsz=640,
        batch=-1,
        cache="ram",
        workers=2,
        device=device,
        project=str(ROOT / "runs" / "train"),
        name="yolo26n_lgmsf_lite_coffee3000_highcompute",
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
