"""Offline SAM2-assisted label review for coffee YOLO datasets.

This tool never mutates the source dataset. In the default review mode it
copies the dataset, writes a review CSV, and creates preview images that compare
the original YOLO box with a candidate visible-region box. If SAM2 is not
installed or no SAM2 checkpoint is provided, the candidate is produced by a
lightweight leaf-color mask so the review workflow still runs locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coffee_v4_training.build_recall_datasets import (  # noqa: E402
    box_iou_xyxy,
    clipped_normalized_box,
    format_label,
    is_leaf_like_rgb,
    pixel_to_yolo,
    prepare_output,
    split_entries,
    write_dataset_yaml,
    yolo_to_pixels,
)
from coffee_v4_training.dataset_diagnostics import (  # noqa: E402
    IMAGE_EXTS,
    LabelBox,
    image_to_label_path,
    load_dataset,
    read_labels,
)


CSV_FIELDS = (
    "split",
    "image",
    "label",
    "box_index",
    "class_id",
    "class_name",
    "original_x",
    "original_y",
    "original_w",
    "original_h",
    "candidate_x",
    "candidate_y",
    "candidate_w",
    "candidate_h",
    "iou",
    "mask_area_ratio",
    "touches_border",
    "needs_review",
    "review_reason",
    "sam_status",
)


def normalized_xyxy(box: LabelBox) -> tuple[float, float, float, float]:
    return (
        box.x - box.w / 2,
        box.y - box.h / 2,
        box.x + box.w / 2,
        box.y + box.h / 2,
    )


def normalized_iou(a: LabelBox, b: LabelBox) -> float:
    return box_iou_xyxy(normalized_xyxy(a), normalized_xyxy(b))


def box_touches_border(box: LabelBox, margin: float = 0.02) -> bool:
    x1, y1, x2, y2 = normalized_xyxy(box)
    return x1 <= margin or y1 <= margin or x2 >= 1.0 - margin or y2 >= 1.0 - margin


def leaf_mask_candidate(image: Image.Image, box: LabelBox) -> tuple[LabelBox | None, str]:
    """Return a candidate bbox using a cheap leaf-color mask inside the GT box."""
    x1, y1, x2, y2 = yolo_to_pixels(box, image.width, image.height)
    if x2 <= x1 or y2 <= y1:
        return None, "fallback_leafmask_invalid_box"

    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    original_w, original_h = crop.size
    scale_x = scale_y = 1.0
    max_side = max(original_w, original_h)
    if max_side > 256:
        scale = 256 / max_side
        resized = crop.resize((max(int(original_w * scale), 1), max(int(original_h * scale), 1)))
        scale_x = original_w / resized.width
        scale_y = original_h / resized.height
        crop = resized

    xs: list[int] = []
    ys: list[int] = []
    pixels = crop.load()
    for py in range(crop.height):
        for px in range(crop.width):
            if is_leaf_like_rgb(pixels[px, py]):
                xs.append(px)
                ys.append(py)
    if not xs or not ys:
        return box, "fallback_leafmask_empty"

    cx1 = x1 + min(xs) * scale_x
    cy1 = y1 + min(ys) * scale_y
    cx2 = x1 + (max(xs) + 1) * scale_x
    cy2 = y1 + (max(ys) + 1) * scale_y
    candidate = pixel_to_yolo(box.cls, cx1, cy1, cx2, cy2, image.width, image.height)
    return candidate or box, "fallback_leafmask"


def load_sam2_predictor(config: str | None, checkpoint: str | None, device: str, require_sam2: bool) -> tuple[Any | None, str]:
    """Best-effort SAM2 loader; returns None unless all optional pieces exist."""
    if not config or not checkpoint:
        if require_sam2:
            raise RuntimeError("--require-sam2 needs --sam2-config and --sam2-checkpoint.")
        return None, "sam2_not_requested_fallback_leafmask"
    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional SAM2 install
        if require_sam2:
            raise RuntimeError(f"SAM2 import failed: {exc}") from exc
        return None, "sam2_unavailable_fallback_leafmask"
    model = build_sam2(config, checkpoint, device=device)
    return SAM2ImagePredictor(model), "sam2"


def sam2_candidate(predictor: Any, image: Image.Image, box: LabelBox) -> LabelBox | None:
    """Run SAM2 with the YOLO box as prompt and convert the mask to a bbox."""
    import numpy as np  # type: ignore

    x1, y1, x2, y2 = yolo_to_pixels(box, image.width, image.height)
    predictor.set_image(np.array(image.convert("RGB")))
    masks, _scores, _logits = predictor.predict(
        box=np.array([x1, y1, x2, y2], dtype=np.float32),
        multimask_output=False,
    )
    mask = masks[0]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return pixel_to_yolo(box.cls, float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), image.width, image.height)


def review_reason(original: LabelBox, candidate: LabelBox, iou: float, iou_gate: float, area_ratio: float) -> list[str]:
    reasons: list[str] = []
    if iou < iou_gate:
        reasons.append("low_mask_iou")
    if box_touches_border(original) or box_touches_border(candidate):
        reasons.append("border_touching_box")
    if area_ratio < 0.35:
        reasons.append("candidate_too_small")
    if area_ratio > 1.75:
        reasons.append("candidate_too_large")
    return reasons


def draw_preview(image: Image.Image, original: LabelBox, candidate: LabelBox, out_path: Path) -> None:
    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle(yolo_to_pixels(original, image.width, image.height), outline=(255, 40, 40), width=3)
    draw.rectangle(yolo_to_pixels(candidate, image.width, image.height), outline=(40, 220, 80), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path)


def copy_dataset_and_review(
    data_yaml: Path,
    output: Path,
    mode: str,
    iou_gate: float,
    overwrite: bool,
    max_previews: int,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    device: str = "cuda",
    require_sam2: bool = False,
) -> Path:
    if mode != "review":
        raise ValueError("Only mode='review' is supported; manually approve CSV rows before applying label changes.")

    info = load_dataset(data_yaml)
    output = output.resolve()
    prepare_output(output, overwrite)
    predictor, sam_status = load_sam2_predictor(sam2_config, sam2_checkpoint, device, require_sam2)

    rows: list[dict[str, str]] = []
    copied_images = 0
    copied_labels = 0
    review_count = 0
    preview_count = 0

    for split in ("train", "val", "test"):
        for image_root in split_entries(info, split):
            if not image_root.exists():
                continue
            images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
            for image_path in images:
                rel = image_path.relative_to(image_root)
                out_image = output / "images" / split / rel
                out_label = output / "labels" / split / rel.with_suffix(".txt")
                out_image.parent.mkdir(parents=True, exist_ok=True)
                out_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, out_image)
                copied_images += 1

                source_label = image_to_label_path(image_path)
                boxes, _warnings = read_labels(source_label, len(info.names))
                fixed_boxes = [fixed for box in boxes for fixed, _clipped in [clipped_normalized_box(box)] if fixed is not None]
                out_label.write_text(format_label(fixed_boxes), encoding="utf-8")
                copied_labels += 1

                with Image.open(image_path) as raw:
                    image = raw.convert("RGB")
                    for index, original in enumerate(fixed_boxes):
                        if predictor is not None:
                            candidate = sam2_candidate(predictor, image, original)
                            candidate_status = "sam2"
                        else:
                            candidate, candidate_status = leaf_mask_candidate(image, original)
                        candidate = candidate or original
                        iou = normalized_iou(original, candidate)
                        area_ratio = candidate.area / original.area if original.area > 0 else 0.0
                        reasons = review_reason(original, candidate, iou, iou_gate, area_ratio)
                        needs_review = bool(reasons)
                        if needs_review:
                            review_count += 1
                            if preview_count < max_previews:
                                preview_path = output / "review_previews" / f"{split}_{image_path.stem}_{index}.jpg"
                                draw_preview(image, original, candidate, preview_path)
                                preview_count += 1
                        rows.append(
                            {
                                "split": split,
                                "image": str((Path("images") / split / rel).as_posix()),
                                "label": str((Path("labels") / split / rel.with_suffix(".txt")).as_posix()),
                                "box_index": str(index),
                                "class_id": str(original.cls),
                                "class_name": info.names[original.cls] if 0 <= original.cls < len(info.names) else str(original.cls),
                                "original_x": f"{original.x:.6f}",
                                "original_y": f"{original.y:.6f}",
                                "original_w": f"{original.w:.6f}",
                                "original_h": f"{original.h:.6f}",
                                "candidate_x": f"{candidate.x:.6f}",
                                "candidate_y": f"{candidate.y:.6f}",
                                "candidate_w": f"{candidate.w:.6f}",
                                "candidate_h": f"{candidate.h:.6f}",
                                "iou": f"{iou:.6f}",
                                "mask_area_ratio": f"{area_ratio:.6f}",
                                "touches_border": "1" if box_touches_border(original) or box_touches_border(candidate) else "0",
                                "needs_review": "1" if needs_review else "0",
                                "review_reason": ";".join(reasons),
                                "sam_status": candidate_status if predictor is None else sam_status,
                            }
                        )

    yaml_path = write_dataset_yaml(info, output, output.name)
    review_csv = output / "review_candidates.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "variant": "sam_refined",
        "source_dataset_yaml": str(info.yaml_path),
        "derived_dataset_yaml": str(yaml_path.resolve()),
        "source_mutation": "read_only",
        "parameters": {
            "mode": mode,
            "iou_gate": iou_gate,
            "sam2_config": sam2_config,
            "sam2_checkpoint": sam2_checkpoint,
            "sam_status": sam_status,
            "review_only": True,
        },
        "outputs": {
            "copied_images": copied_images,
            "copied_labels": copied_labels,
            "objects_reviewed": len(rows),
            "needs_review": review_count,
            "preview_images": preview_count,
            "review_csv": str(review_csv.resolve()),
        },
    }
    (output / "sam2_refine_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return yaml_path


def build_sam2_review_dataset(
    data_yaml: Path,
    output: Path,
    mode: str = "review",
    iou_gate: float = 0.55,
    overwrite: bool = False,
    max_previews: int = 100,
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    device: str = "cuda",
    require_sam2: bool = False,
) -> Path:
    """Public testable wrapper for the SAM2 review dataset builder."""
    return copy_dataset_and_review(
        data_yaml=data_yaml,
        output=output,
        mode=mode,
        iou_gate=iou_gate,
        overwrite=overwrite,
        max_previews=max_previews,
        sam2_config=sam2_config,
        sam2_checkpoint=sam2_checkpoint,
        device=device,
        require_sam2=require_sam2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("review",), default="review")
    parser.add_argument("--iou-gate", type=float, default=0.55)
    parser.add_argument("--max-previews", type=int, default=100)
    parser.add_argument("--sam2-config", default=None)
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-sam2", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_path = build_sam2_review_dataset(
        data_yaml=args.data,
        output=args.output,
        mode=args.mode,
        iou_gate=args.iou_gate,
        overwrite=args.overwrite,
        max_previews=args.max_previews,
        sam2_config=args.sam2_config,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        require_sam2=args.require_sam2,
    )
    print(f"Generated SAM2 review dataset YAML: {yaml_path}")
    print(f"Review CSV: {yaml_path.parent / 'review_candidates.csv'}")


if __name__ == "__main__":
    main()
