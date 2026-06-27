"""Sweep recall-sensitive inference settings for hard-case images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = ROOT / "edgelite_experiment" / "local_ultralytics"


def setup_import_paths() -> None:
    """Use the same project-local Ultralytics fork as the V4 training entrypoint."""
    local = str(LOCAL_ULTRALYTICS)
    root = str(ROOT)
    sys.path = [p for p in sys.path if p not in {local, root}]
    sys.path.insert(0, local)
    sys.path.insert(1, root)

    cached = sys.modules.get("ultralytics")
    cached_file = Path(getattr(cached, "__file__", "")) if cached is not None else None
    if cached_file is not None and not cached_file.resolve().is_relative_to(LOCAL_ULTRALYTICS.resolve()):
        for name in list(sys.modules):
            if name == "ultralytics" or name.startswith("ultralytics."):
                del sys.modules[name]


setup_import_paths()

from coffee_v4_training.dataset_diagnostics import IMAGE_EXTS, image_to_label_path, load_dataset, read_labels, resolve_split_dirs


@dataclass(frozen=True)
class SweepConfig:
    conf: float
    iou: float
    max_det: int
    nms_mode: str = "standard"
    sahi_tile_size: int = 0

    @property
    def name(self) -> str:
        sahi = f"sahi{self.sahi_tile_size}" if self.sahi_tile_size else "full"
        return f"conf{self.conf:g}_iou{self.iou:g}_max{self.max_det}_{self.nms_mode}_{sahi}"

    @property
    def nms_status(self) -> str:
        if self.nms_mode == "standard":
            return "standard"
        if self.nms_mode == "soft":
            return "unsupported_soft_nms_requires_raw_predictions"
        return f"unsupported_{self.nms_mode}"


@dataclass(frozen=True)
class DetectionBox:
    cls: int
    conf: float
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class ImageEvalResult:
    image: str
    gt_total: int
    detection_total: int
    matched_total: int
    false_positive_total: int
    missed_total: int
    low_conf_candidate_total: int
    precision: float
    recall: float
    per_class: dict[str, dict[str, int]]


def parse_float_list(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def build_sweep_configs(
    confs: list[float],
    ious: list[float],
    max_dets: list[int],
    nms_modes: list[str],
    sahi_tile_sizes: list[int],
) -> list[SweepConfig]:
    """Return the Cartesian product of recall sweep settings."""
    return [
        SweepConfig(conf=conf, iou=iou, max_det=max_det, nms_mode=nms_mode, sahi_tile_size=sahi_tile)
        for conf in confs
        for iou in ious
        for max_det in max_dets
        for nms_mode in nms_modes
        for sahi_tile in sahi_tile_sizes
    ]


def effective_probe_conf(config: SweepConfig, probe_conf: float) -> float:
    """Return the low threshold used to collect candidate boxes for this config."""
    return min(config.conf, probe_conf)


def collect_images(data: Path | None, image_dir: Path | None, split: str) -> list[Path]:
    """Collect images from either a YOLO split or an explicit folder."""
    if image_dir is not None:
        roots = [image_dir.resolve()]
    elif data is not None:
        info = load_dataset(data)
        roots = [path.resolve() for path in resolve_split_dirs(info, split)]
    else:
        raise ValueError("Either --data or --image-dir is required.")
    images: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".txt":
            for line in root.read_text(encoding="utf-8").splitlines():
                image = Path(line.strip())
                if image and not image.is_absolute():
                    image = (root.parent / image).resolve()
                if image.suffix.lower() in IMAGE_EXTS:
                    images.append(image)
        elif root.is_file() and root.suffix.lower() in IMAGE_EXTS:
            images.append(root)
        elif root.exists():
            images.extend(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    return sorted(images)


def xywh_iou(a, b) -> float:
    """Return IoU for normalized YOLO xywh-like objects."""
    ax1, ay1 = a.x - a.w / 2, a.y - a.h / 2
    ax2, ay2 = a.x + a.w / 2, a.y + a.h / 2
    bx1, by1 = b.x - b.w / 2, b.y - b.h / 2
    bx2, by2 = b.x + b.w / 2, b.y + b.h / 2
    inter_w = max(min(ax2, bx2) - max(ax1, bx1), 0.0)
    inter_h = max(min(ay2, by2) - max(ay1, by1), 0.0)
    inter = inter_w * inter_h
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def evaluate_image_predictions(
    image: Path,
    predictions: list[DetectionBox],
    nc: int,
    match_iou: float = 0.50,
    candidate_conf_floor: float = 0.03,
) -> ImageEvalResult:
    """Match predictions to one image's GT labels and summarize recall."""
    gt_boxes, _ = read_labels(image_to_label_path(image), nc)
    per_class = {
        str(cls): {"gt": 0, "detections": 0, "matched": 0, "false_positives": 0, "missed": 0, "low_conf_candidates": 0}
        for cls in range(nc)
    }
    for gt in gt_boxes:
        per_class[str(gt.cls)]["gt"] += 1

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    kept_predictions = sorted(
        ((idx, pred) for idx, pred in enumerate(predictions) if pred.conf >= candidate_conf_floor),
        key=lambda item: item[1].conf,
        reverse=True,
    )
    for _, pred in kept_predictions:
        if str(pred.cls) in per_class:
            per_class[str(pred.cls)]["detections"] += 1
    for pred_idx, pred in kept_predictions:
        best_gt = None
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt or gt.cls != pred.cls:
                continue
            iou = xywh_iou(pred, gt)
            if iou >= match_iou and iou > best_iou:
                best_gt = gt_idx
                best_iou = iou
        if best_gt is not None:
            matched_gt.add(best_gt)
            matched_pred.add(pred_idx)
            per_class[str(pred.cls)]["matched"] += 1

    for pred_idx, pred in kept_predictions:
        if pred_idx not in matched_pred and str(pred.cls) in per_class:
            per_class[str(pred.cls)]["false_positives"] += 1

    low_conf_candidates = 0
    for gt_idx, gt in enumerate(gt_boxes):
        if gt_idx in matched_gt:
            continue
        per_class[str(gt.cls)]["missed"] += 1
        has_low_conf_candidate = any(
            pred.cls == gt.cls and pred.conf < candidate_conf_floor and xywh_iou(pred, gt) >= match_iou
            for pred in predictions
        )
        if has_low_conf_candidate:
            low_conf_candidates += 1
            per_class[str(gt.cls)]["low_conf_candidates"] += 1

    matched_total = len(matched_gt)
    detection_total = len(kept_predictions)
    false_positive_total = detection_total - len(matched_pred)
    gt_total = len(gt_boxes)
    return ImageEvalResult(
        image=str(image),
        gt_total=gt_total,
        detection_total=detection_total,
        matched_total=matched_total,
        false_positive_total=false_positive_total,
        missed_total=gt_total - matched_total,
        low_conf_candidate_total=low_conf_candidates,
        precision=matched_total / detection_total if detection_total else 0.0,
        recall=matched_total / gt_total if gt_total else 0.0,
        per_class=per_class,
    )


def aggregate_eval_results(results: list[ImageEvalResult], names: list[str], nms_status: str) -> dict:
    """Aggregate per-image recall results for one sweep config."""
    gt_total = sum(result.gt_total for result in results)
    detection_total = sum(result.detection_total for result in results)
    matched_total = sum(result.matched_total for result in results)
    false_positive_total = sum(result.false_positive_total for result in results)
    low_conf_total = sum(result.low_conf_candidate_total for result in results)
    classes = {
        name: {
            "gt": 0,
            "detections": 0,
            "matched": 0,
            "false_positives": 0,
            "missed": 0,
            "low_conf_candidates": 0,
            "precision": 0.0,
            "recall": 0.0,
        }
        for name in names
    }
    for result in results:
        for cls_id, stats in result.per_class.items():
            name = names[int(cls_id)] if int(cls_id) < len(names) else cls_id
            for key in ("gt", "detections", "matched", "false_positives", "missed", "low_conf_candidates"):
                classes[name][key] += stats[key]
    for stats in classes.values():
        stats["precision"] = stats["matched"] / stats["detections"] if stats["detections"] else 0.0
        stats["recall"] = stats["matched"] / stats["gt"] if stats["gt"] else 0.0
    return {
        "nms_status": nms_status,
        "images": len(results),
        "gt": gt_total,
        "detections": detection_total,
        "matched": matched_total,
        "false_positives": false_positive_total,
        "missed": gt_total - matched_total,
        "low_conf_candidates": low_conf_total,
        "precision": matched_total / detection_total if detection_total else 0.0,
        "recall": matched_total / gt_total if gt_total else 0.0,
        "classes": classes,
    }


def write_recall_summary(
    results_by_config: dict[str, list[ImageEvalResult]],
    output: Path,
    names: list[str],
    nms_status_by_config: dict[str, str],
) -> Path:
    """Write recall-oriented JSON summary for sweep outputs."""
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "configs": {
            config: aggregate_eval_results(results, names, nms_status_by_config.get(config, "standard"))
            for config, results in results_by_config.items()
        }
    }
    path = output / "recall_sweep_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_plan(configs: list[SweepConfig], images: list[Path], output: Path) -> Path:
    """Write the sweep matrix without requiring model weights."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "recall_sweep_plan.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("config", "conf", "iou", "max_det", "nms_mode", "nms_status", "sahi_tile_size", "images"),
        )
        writer.writeheader()
        for config in configs:
            writer.writerow(
                {
                    "config": config.name,
                    "conf": config.conf,
                    "iou": config.iou,
                    "max_det": config.max_det,
                    "nms_mode": config.nms_mode,
                    "nms_status": config.nms_status,
                    "sahi_tile_size": config.sahi_tile_size,
                    "images": len(images),
                }
            )
    return path


def boxes_from_result(result, image: Path) -> list[DetectionBox]:
    """Extract normalized xywh detections from one Ultralytics result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    h, w = getattr(result, "orig_shape", (None, None))
    if not h or not w:
        from PIL import Image  # noqa: PLC0415

        with Image.open(image) as im:
            w, h = im.size
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    detections = []
    for cls, conf, (x1, y1, x2, y2) in zip(classes, confs, xyxy, strict=False):
        detections.append(
            DetectionBox(
                cls=int(cls),
                conf=float(conf),
                x=((x1 + x2) / 2) / w,
                y=((y1 + y2) / 2) / h,
                w=(x2 - x1) / w,
                h=(y2 - y1) / h,
            )
        )
    return detections


def run_ultralytics_sweep(
    model_path: Path,
    configs: list[SweepConfig],
    images: list[Path],
    output: Path,
    device: str | None,
    data: Path | None = None,
    match_iou: float = 0.50,
    probe_conf: float = 0.03,
) -> Path:
    """Run Ultralytics predictions and summarize detection counts.

    Soft-NMS is recorded in the matrix for comparison planning. Ultralytics
    prediction itself uses the standard NMS path unless a project-local custom
    postprocessor is added later.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(model_path))
    names = []
    if data is not None:
        names = load_dataset(data).names
    path = output / "recall_sweep_results.csv"
    results_by_config: dict[str, list[ImageEvalResult]] = {}
    nms_status_by_config: dict[str, str] = {}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "config",
                "image",
                "conf",
                "iou",
                "max_det",
                "nms_mode",
                "nms_status",
                "sahi_tile_size",
                "detections",
                "gt",
                "matched",
                "false_positives",
                "missed",
                "low_conf_candidates",
                "precision",
                "recall",
            ),
        )
        writer.writeheader()
        for config in configs:
            config_results = []
            nms_status_by_config[config.name] = config.nms_status
            for image in images:
                if config.nms_status == "standard":
                    predict_conf = effective_probe_conf(config, probe_conf)
                    results = model.predict(
                        source=str(image),
                        conf=predict_conf,
                        iou=config.iou,
                        max_det=config.max_det,
                        device=device,
                        verbose=False,
                    )
                    predictions = boxes_from_result(results[0], image) if results else []
                else:
                    predictions = []
                eval_result = None
                if names:
                    eval_result = evaluate_image_predictions(
                        image,
                        predictions,
                        len(names),
                        match_iou=match_iou,
                        candidate_conf_floor=config.conf,
                    )
                    config_results.append(eval_result)
                writer.writerow(
                    {
                        "config": config.name,
                        "image": str(image),
                        "conf": config.conf,
                        "iou": config.iou,
                        "max_det": config.max_det,
                        "nms_mode": config.nms_mode,
                        "nms_status": config.nms_status,
                        "sahi_tile_size": config.sahi_tile_size,
                        "detections": len(predictions),
                        "gt": eval_result.gt_total if eval_result else "",
                        "matched": eval_result.matched_total if eval_result else "",
                        "false_positives": eval_result.false_positive_total if eval_result else "",
                        "missed": eval_result.missed_total if eval_result else "",
                        "low_conf_candidates": eval_result.low_conf_candidate_total if eval_result else "",
                        "precision": f"{eval_result.precision:.6f}" if eval_result else "",
                        "recall": (
                            f"{eval_result.recall:.6f}"
                            if eval_result
                            else ""
                        ),
                    }
                )
            if config_results:
                results_by_config[config.name] = config_results
    if results_by_config and names:
        write_recall_summary(results_by_config, output, names, nms_status_by_config)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=None, help="Optional weights path. If omitted, only writes plan CSV.")
    parser.add_argument("--data", type=Path, default=None, help="YOLO dataset YAML.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Explicit hard-case image folder.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, default=Path("runs/coffee_v4_recall_sweep"))
    parser.add_argument("--confs", default="0.03,0.05,0.10,0.25")
    parser.add_argument("--ious", default="0.45,0.60,0.75")
    parser.add_argument("--max-dets", default="100,300,600")
    parser.add_argument("--nms-modes", default="standard,soft")
    parser.add_argument("--sahi-tile-sizes", default="0,640")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--probe-conf", type=float, default=0.03, help="Lowest confidence used to collect candidate boxes before per-config filtering.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = collect_images(args.data, args.image_dir, args.split)
    configs = build_sweep_configs(
        parse_float_list(args.confs),
        parse_float_list(args.ious),
        parse_int_list(args.max_dets),
        parse_str_list(args.nms_modes),
        parse_int_list(args.sahi_tile_sizes),
    )
    output = args.output.resolve()
    plan_path = write_plan(configs, images, output)
    print(f"Sweep plan: {plan_path}")
    print(f"Images: {len(images)}")
    print(f"Configs: {len(configs)}")
    if args.model is not None:
        results_path = run_ultralytics_sweep(
            args.model,
            configs,
            images,
            output,
            args.device,
            data=args.data,
            match_iou=args.match_iou,
            probe_conf=args.probe_conf,
        )
        print(f"Sweep results: {results_path}")


if __name__ == "__main__":
    main()
