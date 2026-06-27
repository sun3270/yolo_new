"""Unified training entrypoint for coffee_self_sum V4 experiments.

Examples:
    python coffee_v4_training/train.py --list
    python coffee_v4_training/train.py --exp edgelite_bibridge_wiou --dry-run
    python coffee_v4_training/train.py --exp edgelite_bibridge_elteb_wiou
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
EDGE_ROOT = ROOT / "edgelite_experiment"
ELTEB_ROOT = ROOT / "elteb_experiment"
WIOU_ROOT = ROOT / "wiou_progloss_experiment"
LOCAL_ULTRALYTICS = EDGE_ROOT / "local_ultralytics"
DEFAULT_DATA = ROOT / "coffee_self_sum" / "coffee_self_sum.yaml"
DEFAULT_WEIGHTS = ROOT / "yolo26n.pt"
DEFAULT_PROJECT = ROOT / "runs" / "train"
DEFAULT_PREPROCESS_OUTPUT = ELTEB_ROOT / "preprocessed_data"
LOSS_CONFIG = WIOU_ROOT / "loss_config.yaml"


def setup_import_paths() -> None:
    """Ensure the experiment-local Ultralytics package wins over the repo root."""
    local = str(LOCAL_ULTRALYTICS)
    root = str(ROOT)
    sys.path = [p for p in sys.path if p not in {local, root}]
    sys.path.insert(0, local)
    sys.path.insert(1, root)


setup_import_paths()

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402

from coffee_v4_training.build_recall_datasets import build_variant as build_recall_dataset  # noqa: E402
from coffee_v4_training.dataset_diagnostics import apply_path_remap, parse_path_remaps  # noqa: E402
from elteb_experiment.build_preprocessed_dataset import build_dataset  # noqa: E402
from wiou_progloss_experiment.dataset_class_counts import count_train_labels, format_class_counts  # noqa: E402
from wiou_progloss_experiment.wiou_progloss_loss import (  # noqa: E402
    DEFAULT_LOSS_CONFIG,
    configure_wiou_progloss,
    patch_detection_model_loss,
)


@dataclass(frozen=True)
class Experiment:
    """One trainable coffee V4 experiment."""

    exp_id: str
    cfg: Path
    run_name: str
    loss: str = "native"
    preprocess: str | None = None
    loss_overrides: dict[str, Any] | None = None
    train_overrides: dict[str, Any] | None = None
    note: str = ""


EXPERIMENTS: dict[str, Experiment] = {
    "native": Experiment(
        "native",
        EDGE_ROOT / "configs" / "yolo26n_original_copy.yaml",
        "v4_native_coffee_self_sum",
        note="Original YOLO26n copy, native loss.",
    ),
    "native_wiou": Experiment(
        "native_wiou",
        EDGE_ROOT / "configs" / "yolo26n_original_copy.yaml",
        "v4_native_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="Original YOLO26n copy with WIoU + ProgLoss.",
    ),
    "edgelite": Experiment(
        "edgelite",
        EDGE_ROOT / "configs" / "yolo26n_edgelite.yaml",
        "v4_edgelite_coffee_self_sum",
        note="EdgeLite structure, native loss.",
    ),
    "edgelite_wiou": Experiment(
        "edgelite_wiou",
        EDGE_ROOT / "configs" / "yolo26n_edgelite.yaml",
        "v4_edgelite_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="EdgeLite structure with WIoU + ProgLoss.",
    ),
    "edgelite_simam_wiou": Experiment(
        "edgelite_simam_wiou",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_simam.yaml",
        "v4_edgelite_simam_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="EdgeLite + SimAM control.",
    ),
    "edgelite_bibridge": Experiment(
        "edgelite_bibridge",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_coffee_self_sum",
        note="Current V4 bidirectional P3/P5 bridge, native loss.",
    ),
    "edgelite_bibridge_wiou": Experiment(
        "edgelite_bibridge_wiou",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="Current V4 main baseline.",
    ),
    "edgelite_simam_bibridge_wiou": Experiment(
        "edgelite_simam_bibridge_wiou",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_simam_bibridge.yaml",
        "v4_edgelite_simam_bibridge_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="BiBridge plus SimAM control.",
    ),
    "edgelite_bibridge_clahe_wiou": Experiment(
        "edgelite_bibridge_clahe_wiou",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_clahe_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="clahe",
        note="Preprocessing-only CLAHE ablation.",
    ),
    "edgelite_bibridge_laplacian_wiou": Experiment(
        "edgelite_bibridge_laplacian_wiou",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_laplacian_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="laplacian",
        note="Preprocessing-only Laplacian ablation.",
    ),
    "native_elteb_wiou": Experiment(
        "native_elteb_wiou",
        ELTEB_ROOT / "configs" / "yolo26n_native_elteb.yaml",
        "v4_native_elteb_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="ELTEB mechanism control on native backbone.",
    ),
    "native_elteb_lite_wiou": Experiment(
        "native_elteb_lite_wiou",
        ELTEB_ROOT / "configs" / "yolo26n_native_elteb_lite.yaml",
        "v4_native_elteb_lite_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="ELTEBLite mechanism control on native backbone.",
    ),
    "edgelite_elteb_wiou": Experiment(
        "edgelite_elteb_wiou",
        ELTEB_ROOT / "configs" / "yolo26n_edgelite_elteb.yaml",
        "v4_edgelite_elteb_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="ELTEB on EdgeLite without BiBridge.",
    ),
    "edgelite_elteb_lite_wiou": Experiment(
        "edgelite_elteb_lite_wiou",
        ELTEB_ROOT / "configs" / "yolo26n_edgelite_elteb_lite.yaml",
        "v4_edgelite_elteb_lite_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="ELTEBLite on EdgeLite without BiBridge.",
    ),
    "edgelite_bibridge_elteb": Experiment(
        "edgelite_bibridge_elteb",
        ELTEB_ROOT / "configs" / "yolo26n_edgelite_bibridge_elteb.yaml",
        "v4_edgelite_bibridge_elteb_coffee_self_sum",
        note="BiBridge + ELTEB, native loss.",
    ),
    "edgelite_bibridge_elteb_wiou": Experiment(
        "edgelite_bibridge_elteb_wiou",
        ELTEB_ROOT / "configs" / "yolo26n_edgelite_bibridge_elteb.yaml",
        "v4_edgelite_bibridge_elteb_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="BiBridge + ELTEB with WIoU + ProgLoss.",
    ),
    "edgelite_bibridge_elteb_lite": Experiment(
        "edgelite_bibridge_elteb_lite",
        ELTEB_ROOT / "configs" / "yolo26n_edgelite_bibridge_elteb_lite.yaml",
        "v4_edgelite_bibridge_elteb_lite_coffee_self_sum",
        note="BiBridge + ELTEBLite, native loss.",
    ),
    "edgelite_bibridge_elteb_lite_wiou": Experiment(
        "edgelite_bibridge_elteb_lite_wiou",
        ELTEB_ROOT / "configs" / "yolo26n_edgelite_bibridge_elteb_lite.yaml",
        "v4_edgelite_bibridge_elteb_lite_wiou_progloss_coffee_self_sum",
        loss="wiou_progloss",
        note="BiBridge + ELTEBLite with WIoU + ProgLoss.",
    ),
    "edgelite_bibridge_wiou_sahi": Experiment(
        "edgelite_bibridge_wiou_sahi",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_sahi_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="sahi",
        note="BiBridge + WIoU/ProgLoss with sliced small-object derived dataset.",
    ),
    "edgelite_bibridge_wiou_copypaste": Experiment(
        "edgelite_bibridge_wiou_copypaste",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_copypaste_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="copypaste",
        note="BiBridge + WIoU/ProgLoss with tail-class Copy-Paste derived dataset.",
    ),
    "edgelite_bibridge_wiou_roi": Experiment(
        "edgelite_bibridge_wiou_roi",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_roi_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="roi",
        note="BiBridge + WIoU/ProgLoss with lightweight ROI/background suppression derived dataset.",
    ),
    "edgelite_bibridge_wiou_sanitize_labels": Experiment(
        "edgelite_bibridge_wiou_sanitize_labels",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_sanitize_labels_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="sanitize_labels",
        note="BiBridge + WIoU/ProgLoss on geometry-sanitized labels.",
    ),
    "edgelite_bibridge_wiou_sanitize_copypaste_context": Experiment(
        "edgelite_bibridge_wiou_sanitize_copypaste_context",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_sanitize_copypaste_context_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="sanitize_copypaste_context",
        note="BiBridge + WIoU/ProgLoss with label sanitation followed by context-aware Copy-Paste.",
    ),
    "edgelite_bibridge_wiou_sam_refined": Experiment(
        "edgelite_bibridge_wiou_sam_refined",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_sam_refined_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="sam_refined",
        note="BiBridge + WIoU/ProgLoss on a manually reviewed SAM2-refined derived dataset.",
    ),
    "edgelite_bibridge_wiou_copypaste_context": Experiment(
        "edgelite_bibridge_wiou_copypaste_context",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_copypaste_context_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="copypaste_context",
        note="BiBridge + WIoU/ProgLoss with context-aware weighted Copy-Paste derived dataset.",
    ),
    "edgelite_bibridge_wiou_sahi_trainonly": Experiment(
        "edgelite_bibridge_wiou_sahi_trainonly",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_sahi_trainonly_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="sahi_trainonly",
        note="BiBridge + WIoU/ProgLoss with train-only sliced small-object dataset.",
    ),
    "edgelite_bibridge_wiou_roi_leafmask": Experiment(
        "edgelite_bibridge_wiou_roi_leafmask",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_wiou_roi_leafmask_coffee_self_sum",
        loss="wiou_progloss",
        preprocess="roi_leafmask",
        note="BiBridge + WIoU/ProgLoss with leaf-mask ROI/background suppression dataset.",
    ),
    "edgelite_bibridge_nwd": Experiment(
        "edgelite_bibridge_nwd",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_edgelite_bibridge_nwd_progloss_coffee_self_sum",
        loss="wiou_progloss_nwd",
        loss_overrides={"wiou_fallback_base": "nwd", "nwd_constant": 12.8},
        note="BiBridge + ProgLoss with NWD bbox base loss for tiny-box sensitivity ablation.",
    ),
    "edgelite_bibridge_wiou_y26_recipe": Experiment(
        "edgelite_bibridge_wiou_y26_recipe",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_strong_y26_recipe_coffee_self_sum",
        loss="wiou_progloss",
        train_overrides={
            "optimizer": "MuSGD",
            "lr0": 0.010,
            "lrf": 0.010,
            "momentum": 0.900,
            "weight_decay": 0.0005,
            "box": 8.0,
            "cls": 0.65,
            "dfl": 1.0,
        },
        note="M0: BiBridge with explicit YOLO26-style MuSGD/ProgLoss training recipe.",
    ),
    "edgelite_bibridge_wiou_stal_lite": Experiment(
        "edgelite_bibridge_wiou_stal_lite",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge.yaml",
        "v4_strong_stal_lite_coffee_self_sum",
        loss="wiou_progloss",
        loss_overrides={"tal_topk": 13, "tal_topk2": 3, "small_box_topk_boost": True, "small_box_topk": 20},
        train_overrides={"optimizer": "MuSGD", "cls": 0.65, "dfl": 1.0},
        note="M1: BiBridge with STAL-lite broader positive assignment for small boxes.",
    ),
    "edgelite_bibridge_wiou_p2": Experiment(
        "edgelite_bibridge_wiou_p2",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_bibridge_p2.yaml",
        "v4_strong_bibridge_p2_coffee_self_sum",
        loss="wiou_progloss",
        train_overrides={"optimizer": "MuSGD", "cls": 0.65, "dfl": 1.0},
        note="M2: BiBridge with an added P2 detection head for small edge lesions.",
    ),
    "edgelite_hyperace_wiou": Experiment(
        "edgelite_hyperace_wiou",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_hyperace.yaml",
        "v4_strong_hyperace_coffee_self_sum",
        loss="wiou_progloss",
        train_overrides={"optimizer": "MuSGD", "cls": 0.65, "dfl": 1.0},
        note="M3: HyperACE-lite P3/P4/P5 high-order fusion with WIoU/ProgLoss.",
    ),
    "edgelite_hyperace_p2_wiou": Experiment(
        "edgelite_hyperace_p2_wiou",
        EDGE_ROOT / "configs" / "yolo26n_edgelite_hyperace_p2.yaml",
        "v4_strong_hyperace_p2_coffee_self_sum",
        loss="wiou_progloss",
        train_overrides={"optimizer": "MuSGD", "cls": 0.65, "dfl": 1.0},
        note="M3: HyperACE-lite plus P2 head for aggressive recall screening.",
    ),
}


CORE_EXPERIMENTS = (
    "native_wiou",
    "edgelite_wiou",
    "edgelite_bibridge_wiou",
    "edgelite_bibridge_clahe_wiou",
    "edgelite_bibridge_laplacian_wiou",
    "edgelite_bibridge_elteb_wiou",
    "edgelite_bibridge_elteb_lite_wiou",
)

MAIN_EXPERIMENTS = (
    "edgelite_bibridge_wiou",
    "edgelite_bibridge_elteb_wiou",
    "edgelite_bibridge_elteb_lite_wiou",
)

ELTEB_CONTROL_EXPERIMENTS = (
    "native_elteb_wiou",
    "native_elteb_lite_wiou",
    "edgelite_elteb_wiou",
    "edgelite_elteb_lite_wiou",
    "edgelite_bibridge_elteb_wiou",
    "edgelite_bibridge_elteb_lite_wiou",
)

RECALL_EXPERIMENTS = (
    "edgelite_bibridge_wiou_sahi",
    "edgelite_bibridge_wiou_copypaste",
    "edgelite_bibridge_wiou_roi",
    "edgelite_bibridge_wiou_sanitize_labels",
    "edgelite_bibridge_wiou_sanitize_copypaste_context",
    "edgelite_bibridge_wiou_sam_refined",
    "edgelite_bibridge_wiou_copypaste_context",
    "edgelite_bibridge_wiou_roi_leafmask",
    "edgelite_bibridge_nwd",
)

STRONG_EXPERIMENTS = (
    "edgelite_bibridge_wiou_y26_recipe",
    "edgelite_bibridge_wiou_stal_lite",
    "edgelite_bibridge_wiou_p2",
    "edgelite_hyperace_wiou",
    "edgelite_hyperace_p2_wiou",
)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def resolve_path(value: str | Path | None, default: Path) -> Path:
    raw = str(value) if value is not None and str(value) != "" else str(default)
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def path_from_cli_env(cli_value: Path | None, env_name: str, default: Path) -> Path:
    return resolve_path(cli_value if cli_value is not None else os.environ.get(env_name), default)


def select_device(cli_device: str | None) -> str | int:
    device = cli_device or os.environ.get("COFFEE_DEVICE")
    if device is not None and device != "":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Set COFFEE_DEVICE=cpu only for debugging, "
            "or run training in a CUDA-enabled environment."
        )
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return 0


def preprocess_yaml_path(source_yaml: Path, output_root: Path, variant: str) -> Path:
    dataset_name = source_yaml.stem
    return (output_root / f"{dataset_name}_{variant}" / f"{dataset_name}_{variant}.yaml").resolve()


def remap_yaml_value(value: Any, remaps: tuple[tuple[str, str], ...]) -> Any:
    """Apply path remaps to YAML path fields while preserving lists and relative paths."""
    if isinstance(value, str):
        remapped = apply_path_remap(value, remaps)
        return remapped
    if isinstance(value, list):
        return [remap_yaml_value(item, remaps) for item in value]
    return value


def materialize_remapped_dataset_yaml(data: Path, project: Path, run_name: str, path_remap: str | None = None) -> Path:
    """Write a temporary dataset YAML with path/train/val/test remapped for the current machine."""
    remaps = parse_path_remaps(path_remap)
    if not remaps:
        return data
    cfg = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    remapped_cfg = dict(cfg)
    for key in ("path", "train", "val", "test"):
        if key in remapped_cfg:
            remapped_cfg[key] = remap_yaml_value(remapped_cfg[key], remaps)
    if remapped_cfg == cfg:
        return data
    output_dir = project.resolve() / "_remapped_data" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_yaml = output_dir / f"{data.stem}_remapped.yaml"
    output_yaml.write_text(yaml.safe_dump(remapped_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_yaml.resolve()


def run_last_checkpoint(project: Path, run_name: str) -> Path:
    """Return the conventional last.pt path for a run name."""
    return (project / run_name / "weights" / "last.pt").resolve()


def resolve_resume_checkpoint(args: argparse.Namespace, project: Path, run_name: str) -> Path | None:
    """Resolve the checkpoint to resume from, defaulting to same-run last.pt when present."""
    if getattr(args, "fresh", False) and (getattr(args, "resume", False) or getattr(args, "resume_from", None)):
        raise ValueError("--fresh cannot be combined with --resume or --resume-from.")
    if getattr(args, "fresh", False):
        return None

    resume_from = getattr(args, "resume_from", None)
    if resume_from:
        checkpoint = resolve_path(resume_from, ROOT)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        return checkpoint

    checkpoint = run_last_checkpoint(project, run_name)
    if getattr(args, "resume", False):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found for run {run_name!r}: {checkpoint}")
        return checkpoint

    auto_resume = getattr(args, "auto_resume", None)
    if auto_resume is None:
        auto_resume = env_bool("COFFEE_AUTO_RESUME", True)
    return checkpoint if auto_resume and checkpoint.exists() else None


def resolve_sanitize_copypaste_context(
    source_data: Path,
    preprocess_output: Path,
    overwrite_preprocess: bool,
    build_preprocess: bool,
) -> Path:
    """Resolve or build the clean-label plus context Copy-Paste derived dataset."""
    final_yaml = preprocess_yaml_path(source_data, preprocess_output, "sanitize_copypaste_context")
    if not build_preprocess:
        return final_yaml
    if final_yaml.exists() and not overwrite_preprocess:
        return final_yaml

    sanitize_output = preprocess_output.resolve() / f"{source_data.stem}_sanitize_labels"
    sanitize_yaml = sanitize_output / f"{source_data.stem}_sanitize_labels.yaml"
    if not sanitize_yaml.exists() or overwrite_preprocess:
        sanitize_yaml = build_recall_dataset(
            "sanitize_labels",
            source_data,
            sanitize_output,
            argparse.Namespace(overwrite=overwrite_preprocess),
        )

    final_output = preprocess_output.resolve() / f"{source_data.stem}_sanitize_copypaste_context"
    return build_recall_dataset(
        "copypaste_context",
        sanitize_yaml,
        final_output,
        argparse.Namespace(
            overwrite=overwrite_preprocess,
            copy_class_weights="1:2,2:3,3:2,5:3",
            max_paste_iou=0.10,
            max_paste_attempts=40,
            copies_per_source=1,
            target_classes="1,2,3,5",
            seed=2026,
        ),
    )


def resolve_sam_refined_dataset(
    source_data: Path,
    preprocess_output: Path,
    build_preprocess: bool,
) -> Path:
    """Resolve the SAM2 review dataset; it must be built explicitly first."""
    yaml_path = preprocess_yaml_path(source_data, preprocess_output, "sam_refined")
    if yaml_path.exists() or not build_preprocess:
        return yaml_path
    raise FileNotFoundError(
        "SAM-refined dataset yaml not found: "
        f"{yaml_path}. Run coffee_v4_training/sam2_refine_labels.py first and manually review labels."
    )


def resolve_data_for_experiment(
    exp: Experiment,
    source_data: Path,
    preprocess_output: Path,
    overwrite_preprocess: bool,
    build_preprocess: bool,
) -> Path:
    if exp.preprocess is None:
        return source_data
    if exp.preprocess == "sanitize_copypaste_context":
        return resolve_sanitize_copypaste_context(source_data, preprocess_output, overwrite_preprocess, build_preprocess)
    if exp.preprocess == "sam_refined":
        return resolve_sam_refined_dataset(source_data, preprocess_output, build_preprocess)
    if build_preprocess:
        existing_yaml = preprocess_yaml_path(source_data, preprocess_output, exp.preprocess)
        if existing_yaml.exists() and not overwrite_preprocess:
            return existing_yaml
        if exp.preprocess in {"sanitize_labels", "copypaste", "copypaste_context", "sahi", "sahi_trainonly", "roi", "roi_leafmask"}:
            output = preprocess_output.resolve() / f"{source_data.stem}_{exp.preprocess}"
            return build_recall_dataset(exp.preprocess, source_data, output, argparse.Namespace(overwrite=overwrite_preprocess))
        return build_dataset(exp.preprocess, source_data, preprocess_output, overwrite=overwrite_preprocess)
    return preprocess_yaml_path(source_data, preprocess_output, exp.preprocess)


def configure_loss(data: Path, overrides: dict[str, Any] | None = None) -> None:
    loss_config = YAML.load(LOSS_CONFIG) if LOSS_CONFIG.exists() else {}
    loss_overrides = {k: loss_config.get(k) for k in DEFAULT_LOSS_CONFIG if k in loss_config}
    if overrides:
        loss_overrides.update(overrides)
    class_counts = count_train_labels(data)
    loss_overrides["progloss_class_counts"] = class_counts
    print(f"ProgLoss train class counts: {class_counts}")
    print(f"ProgLoss class count detail: {format_class_counts(data, class_counts)}")
    configure_wiou_progloss(**loss_overrides)
    patch_detection_model_loss()


def train_kwargs(args: argparse.Namespace, data: Path, project: Path, run_name: str, device: str | int) -> dict[str, Any]:
    return {
        "data": str(data),
        "epochs": args.epochs if args.epochs is not None else env_int("COFFEE_EPOCHS", 300),
        "imgsz": args.imgsz if args.imgsz is not None else env_int("COFFEE_IMGSZ", 960),
        "batch": args.batch if args.batch is not None else env_int("COFFEE_BATCH", 64),
        "cache": args.cache if args.cache is not None else os.environ.get("COFFEE_CACHE", "ram"),
        "workers": args.workers if args.workers is not None else env_int("COFFEE_WORKERS", 8),
        "device": device,
        "project": str(project),
        "name": run_name,
        "amp": True,
        "patience": 30,
        "save_period": 20,
        "verbose": True,
        "deterministic": False,
        "close_mosaic": 10,
        "lr0": env_float("COFFEE_LR0", 0.012),
        "lrf": env_float("COFFEE_LRF", 0.01),
        "momentum": env_float("COFFEE_MOMENTUM", 0.937),
        "weight_decay": env_float("COFFEE_WEIGHT_DECAY", 0.0005),
        "warmup_epochs": env_float("COFFEE_WARMUP_EPOCHS", 4.0),
        "box": env_float("COFFEE_BOX", 7.5),
        "cls": env_float("COFFEE_CLS", 0.5),
        "dfl": env_float("COFFEE_DFL", 1.5),
    }


def format_metric(results, key: str) -> str:
    value = results.results_dict.get(key)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def list_experiments() -> None:
    print("Available experiments:")
    for exp_id, exp in EXPERIMENTS.items():
        prep = exp.preprocess or "-"
        print(f"  {exp_id:38s} loss={exp.loss:13s} preprocess={prep:9s} cfg={exp.cfg}")


def run_experiment(args: argparse.Namespace) -> None:
    exp_id = args.exp or os.environ.get("COFFEE_EXP", "edgelite_bibridge_wiou")
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {exp_id!r}. Run with --list to see valid ids.")
    exp = EXPERIMENTS[exp_id]

    data = path_from_cli_env(args.data, "COFFEE_DATA", DEFAULT_DATA)
    weights = path_from_cli_env(args.weights, "COFFEE_WEIGHTS", DEFAULT_WEIGHTS)
    project = path_from_cli_env(args.project, "COFFEE_PROJECT", DEFAULT_PROJECT)
    preprocess_output = path_from_cli_env(args.preprocess_output, "COFFEE_PREPROCESS_OUTPUT", DEFAULT_PREPROCESS_OUTPUT)
    run_name = args.name or os.environ.get("COFFEE_RUN_NAME", exp.run_name)

    if not exp.cfg.exists():
        raise FileNotFoundError(f"Model yaml not found: {exp.cfg}")
    if not data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data}")

    active_data = resolve_data_for_experiment(
        exp,
        data,
        preprocess_output,
        overwrite_preprocess=args.overwrite_preprocess,
        build_preprocess=not args.dry_run,
    )
    if not args.dry_run and not active_data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {active_data}")
    if active_data.exists():
        active_data = materialize_remapped_dataset_yaml(active_data, project, run_name, args.path_remap)

    device = "dry-run" if args.dry_run else select_device(args.device)
    resume_checkpoint = resolve_resume_checkpoint(args, project, run_name)
    kwargs = train_kwargs(args, active_data, project, run_name, device)
    if exp.train_overrides:
        kwargs.update(exp.train_overrides)
    if resume_checkpoint is not None:
        kwargs["resume"] = True

    print(f"Experiment: {exp.exp_id}")
    print(f"Note:       {exp.note}")
    print(f"Model YAML: {exp.cfg}")
    print(f"Data YAML:  {active_data}")
    print(f"Loss:       {exp.loss}")
    print(f"Run name:   {run_name}")
    print(f"Resume:     {resume_checkpoint if resume_checkpoint else '-'}")
    print("Train kwargs:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

    if args.dry_run:
        return

    if exp.loss in {"wiou_progloss", "wiou_progloss_nwd"}:
        configure_loss(active_data, exp.loss_overrides)
    elif exp.loss != "native":
        raise ValueError(f"Unsupported loss: {exp.loss}")

    if resume_checkpoint is not None:
        model = YOLO(str(resume_checkpoint))
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    else:
        model = YOLO(str(exp.cfg))
        if not args.no_weights and str(weights).lower() not in {"none", "false", "0"} and weights.exists():
            model = model.load(str(weights))
            print(f"Loaded pretrained weights: {weights}")
        else:
            print("Pretrained weights not loaded; training will start from YAML initialization.")
    model.info(verbose=False)

    results = model.train(**kwargs)

    print("\n=== Training complete ===")
    print(f"Experiment:    {exp.exp_id}")
    print(f"Best mAP50:    {format_metric(results, 'metrics/mAP50(B)')}")
    print(f"Best mAP50-95: {format_metric(results, 'metrics/mAP50-95(B)')}")
    print(f"Best weights:  {results.save_dir / 'weights' / 'best.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp", choices=sorted(EXPERIMENTS), default=None, help="Experiment id to train.")
    parser.add_argument("--list", action="store_true", help="List experiments and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve paths and print settings without training.")
    parser.add_argument("--data", type=Path, default=None, help="Dataset YAML. Defaults to coffee_self_sum.")
    parser.add_argument("--weights", type=Path, default=None, help="Pretrained weights path.")
    parser.add_argument("--no-weights", action="store_true", help="Skip pretrained weights even when present.")
    parser.add_argument("--resume", action="store_true", help="Resume from project/name/weights/last.pt and fail if it is missing.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume from an explicit last.pt checkpoint.")
    parser.add_argument("--fresh", action="store_true", help="Disable default same-run auto-resume and start a fresh run.")
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically resume when project/name/weights/last.pt exists. Defaults to COFFEE_AUTO_RESUME or true.",
    )
    parser.add_argument("--project", type=Path, default=None, help="Ultralytics project output directory.")
    parser.add_argument("--name", default=None, help="Run name override.")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, 0,1, or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers.")
    parser.add_argument("--cache", default=None, help="Ultralytics cache setting.")
    parser.add_argument("--preprocess-output", type=Path, default=None, help="Output root for preprocessing variants.")
    parser.add_argument("--overwrite-preprocess", action="store_true", help="Rebuild preprocessing datasets.")
    parser.add_argument("--path-remap", default=None, help="Path remap pairs OLD=>NEW;OLD2=>NEW2 for portable dataset YAMLs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        list_experiments()
        return
    run_experiment(args)


if __name__ == "__main__":
    main()

