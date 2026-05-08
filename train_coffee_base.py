"""Train a YOLO26n baseline on the coffee3000 dataset.

Usage:
    python train_coffee_base.py
"""

from pathlib import Path

import torch

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "yolo26n.pt"
DATA = ROOT / "coffee3000" / "coffee3000.yaml"


def select_cuda_device() -> int:
    """Require CUDA so a baseline run does not silently fall back to CPU."""
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

    model = YOLO(str(WEIGHTS))
    model.info(verbose=False)

    results = model.train(
        data=str(DATA),
        epochs=200,
        imgsz=640,
        batch=32,
        cache="ram",
        workers=2,
        device=device,
        project=str(ROOT / "runs" / "train"),
        name="yolo26n_coffee3000_laptop",
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
