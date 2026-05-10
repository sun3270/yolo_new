"""Train YOLO26n-EdgeLite-P5Slim512 with WIoU/ProgLoss teacher distillation.

Usage:
    python p5slim512_distill_experiment/train_coffee_p5slim512_distill.py

Quick smoke run:
    set P5SLIM_DISTILL_EPOCHS=3
    python p5slim512_distill_experiment/train_coffee_p5slim512_distill.py
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
CFG = EXP_ROOT / "configs" / "yolo26n_edgelite_p5slim512.yaml"
DATA = ROOT / "coffee3000" / "coffee3000.yaml"
LOSS_CONFIG = EXP_ROOT / "loss_config.yaml"
TEACHER_WEIGHTS = ROOT / "android_phone_deploy" / "model" / "best_edgelite_wiou_progloss.pt"
FALLBACK_WEIGHTS = ROOT / "yolo26n.pt"
CLASS_COUNTS = [501, 606, 332, 618, 709, 163]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from wiou_progloss_experiment.wiou_progloss_loss import (  # noqa: E402
    DEFAULT_LOSS_CONFIG,
    configure_wiou_progloss,
)
from p5slim512_distill_experiment.distill_wiou_progloss_loss import (  # noqa: E402
    DEFAULT_DISTILL_CONFIG,
    configure_distillation,
    patch_detection_model_loss,
    set_teacher_model,
)


def select_cuda_device() -> int:
    """Require CUDA so this experiment does not silently run teacher+student on CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Teacher-student training needs a CUDA-enabled PyTorch build."
        )
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return 0


def load_loss_config() -> dict:
    """Load flat YAML config and keep only known experiment keys."""
    cfg = YAML.load(LOSS_CONFIG) if LOSS_CONFIG.exists() else {}
    cfg.setdefault("progloss_class_counts", CLASS_COUNTS)
    return cfg


def configure_losses(raw_cfg: dict) -> None:
    """Configure WIoU/ProgLoss and distillation patch points."""
    wiou_cfg = {k: raw_cfg.get(k) for k in DEFAULT_LOSS_CONFIG if k in raw_cfg}
    distill_cfg = {k: raw_cfg.get(k) for k in DEFAULT_DISTILL_CONFIG if k in raw_cfg}
    configure_wiou_progloss(**wiou_cfg)
    configure_distillation(**distill_cfg)
    patch_detection_model_loss()


def load_teacher(device: int):
    """Load the frozen EdgeLite + WIoU/ProgLoss teacher."""
    if not TEACHER_WEIGHTS.exists():
        raise FileNotFoundError(f"Teacher weights not found: {TEACHER_WEIGHTS}")
    teacher = YOLO(str(TEACHER_WEIGHTS)).model
    teacher.to(torch.device(f"cuda:{device}"))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    set_teacher_model(teacher)
    print(f"Loaded teacher weights: {TEACHER_WEIGHTS}")
    return teacher


def load_student():
    """Build P5Slim-512 and partially initialize from the teacher when available."""
    student = YOLO(str(CFG))
    init_weights = TEACHER_WEIGHTS if TEACHER_WEIGHTS.exists() else FALLBACK_WEIGHTS
    if init_weights.exists():
        student = student.load(str(init_weights))
        print(f"Loaded student initialization weights: {init_weights}")
    else:
        print("No compatible initialization weights found; training starts from YAML initialization.")
    student.info(verbose=False)
    return student


def format_metric(results, key: str) -> str:
    """Safely format a metric from an Ultralytics results object."""
    value = results.results_dict.get(key)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def main():
    device = select_cuda_device()
    raw_cfg = load_loss_config()
    configure_losses(raw_cfg)
    load_teacher(device)
    model = load_student()

    epochs = int(os.environ.get("P5SLIM_DISTILL_EPOCHS", "300"))
    batch = int(os.environ.get("P5SLIM_DISTILL_BATCH", "32"))
    workers = int(os.environ.get("P5SLIM_DISTILL_WORKERS", "2"))
    name = os.environ.get("P5SLIM_DISTILL_NAME", "v3_p5slim512_second_lightweight_coffee3000")

    results = model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        cache="ram",
        workers=workers,
        device=device,
        project=str(ROOT / "runs" / "train"),
        name=name,
        amp=True,
        patience=40,
        save_period=20,
        verbose=True,
        deterministic=False,
        close_mosaic=10,
        lr0=0.008,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=4.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\n=== P5Slim-512 distillation training complete ===")
    print(f"Best mAP50:    {format_metric(results, 'metrics/mAP50(B)')}")
    print(f"Best mAP50-95: {format_metric(results, 'metrics/mAP50-95(B)')}")
    print(f"Best weights:  {results.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
