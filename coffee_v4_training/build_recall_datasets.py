"""Build recall-oriented derived datasets without mutating the source dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coffee_v4_training.dataset_diagnostics import (
    IMAGE_EXTS,
    DatasetInfo,
    LabelBox,
    image_to_label_path,
    load_dataset,
    read_labels,
    resolve_split_dir,
    resolve_split_dirs,
)


DEFAULT_OUTPUT_ROOT = ROOT / "coffee_v4_training" / "derived_data"
DEFAULT_COPY_PASTE_TARGET_CLASSES = "1,2,3,5"  # ANT_CD, BLS_AB, BLS_CD, CR_AB
DEFAULT_CONTEXT_COPY_CLASS_WEIGHTS = "1:1,2:3,3:2,5:3"  # emphasize BLS_AB and CR_AB
DEFAULT_MAX_PASTE_IOU = 0.15
DEFAULT_MAX_PASTE_ATTEMPTS = 40
DEFAULT_MIN_SANITIZED_BOX_AREA = 1e-6


def prepare_output(output: Path, overwrite: bool) -> None:
    """Create a clean derived dataset directory."""
    output = output.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists, use --overwrite: {output}")
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_entry(info: DatasetInfo, split: str) -> str:
    """Return an absolute image split path for a derived YAML."""
    return str(resolve_split_dir(info, split).resolve()).replace("\\", "/")


def write_dataset_yaml(info: DatasetInfo, output: Path, name: str, train=None, val=None, test=None) -> Path:
    """Write a YOLO YAML for a derived dataset."""
    cfg = {
        "path": str(output.resolve()).replace("\\", "/"),
        "train": train if train is not None else "images/train",
        "val": val if val is not None else "images/val",
        "test": test if test is not None else "images/test",
        "nc": len(info.names),
        "names": {i: name for i, name in enumerate(info.names)},
    }
    yaml_path = output / f"{name}.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output / "classes.txt").write_text("\n".join(info.names) + "\n", encoding="utf-8")
    return yaml_path


def count_images(output: Path, split: str) -> int:
    """Count generated images for one split."""
    image_dir = output / "images" / split
    return sum(1 for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS) if image_dir.exists() else 0


def write_recall_manifest(
    info: DatasetInfo,
    output: Path,
    variant: str,
    parameters: dict,
    outputs: dict,
    yaml_path: Path,
) -> Path:
    """Record how a recall-derived dataset was produced."""
    manifest = {
        "variant": variant,
        "source_dataset_yaml": str(info.yaml_path),
        "derived_dataset_yaml": str(yaml_path.resolve()),
        "source_mutation": "read_only",
        "parameters": parameters,
        "outputs": outputs,
    }
    path = output / "recall_dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def boxes_for_image(path: Path, nc: int) -> list[LabelBox]:
    """Read labels for one image."""
    boxes, _ = read_labels(image_to_label_path(path), nc)
    return boxes


def yolo_to_pixels(box: LabelBox, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert normalized xywh to clipped pixel xyxy."""
    x1 = max(int(round((box.x - box.w / 2) * width)), 0)
    y1 = max(int(round((box.y - box.h / 2) * height)), 0)
    x2 = min(int(round((box.x + box.w / 2) * width)), width)
    y2 = min(int(round((box.y + box.h / 2) * height)), height)
    return x1, y1, x2, y2


def pixel_to_yolo(cls: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> LabelBox | None:
    """Convert clipped pixel xyxy to normalized YOLO box."""
    x1 = min(max(x1, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    x2 = min(max(x2, 0.0), width)
    y2 = min(max(y2, 0.0), height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return LabelBox(
        cls=cls,
        x=((x1 + x2) / 2) / width,
        y=((y1 + y2) / 2) / height,
        w=(x2 - x1) / width,
        h=(y2 - y1) / height,
    )


def format_label(boxes: list[LabelBox]) -> str:
    """Serialize YOLO labels."""
    return "\n".join(format_label_box(box) for box in boxes) + ("\n" if boxes else "")


def format_label_box(box: LabelBox, decimals: int = 8) -> str:
    """Serialize one YOLO label while keeping rounded xywh strictly in image bounds."""
    unit = 10 ** -decimals
    x = min(max(box.x, 0.0), 1.0)
    y = min(max(box.y, 0.0), 1.0)
    w = min(max(box.w, unit), 1.0)
    h = min(max(box.h, unit), 1.0)
    for _ in range(4):
        fx, fy, fw, fh = (float(f"{value:.{decimals}f}") for value in (x, y, w, h))
        if fx - fw / 2 < 0.0:
            x = fw / 2 + unit
            continue
        if fx + fw / 2 > 1.0:
            x = 1.0 - fw / 2 - unit
            continue
        if fy - fh / 2 < 0.0:
            y = fh / 2 + unit
            continue
        if fy + fh / 2 > 1.0:
            y = 1.0 - fh / 2 - unit
            continue
        return f"{box.cls} {fx:.{decimals}f} {fy:.{decimals}f} {fw:.{decimals}f} {fh:.{decimals}f}"
    fx, fy, fw, fh = (float(f"{value:.{decimals}f}") for value in (x, y, w, h))
    return f"{box.cls} {fx:.{decimals}f} {fy:.{decimals}f} {fw:.{decimals}f} {fh:.{decimals}f}"


def clipped_normalized_box(box: LabelBox, min_area: float = DEFAULT_MIN_SANITIZED_BOX_AREA) -> tuple[LabelBox | None, bool]:
    """Clip a normalized YOLO box to image bounds."""
    x1 = max(box.x - box.w / 2, 0.0)
    y1 = max(box.y - box.h / 2, 0.0)
    x2 = min(box.x + box.w / 2, 1.0)
    y2 = min(box.y + box.h / 2, 1.0)
    clipped = (
        abs(x1 - (box.x - box.w / 2)) > 1e-9
        or abs(y1 - (box.y - box.h / 2)) > 1e-9
        or abs(x2 - (box.x + box.w / 2)) > 1e-9
        or abs(y2 - (box.y + box.h / 2)) > 1e-9
    )
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < min_area:
        return None, clipped
    return LabelBox(
        cls=box.cls,
        x=(x1 + x2) / 2,
        y=(y1 + y2) / 2,
        w=x2 - x1,
        h=y2 - y1,
    ), clipped


def box_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute IoU for pixel xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def max_iou_with_labels(
    candidate: tuple[float, float, float, float],
    labels: list[LabelBox],
    width: int,
    height: int,
) -> float:
    """Return the largest IoU between a candidate paste box and existing labels."""
    if not labels:
        return 0.0
    return max(box_iou_xyxy(candidate, yolo_to_pixels(box, width, height)) for box in labels)


def clamp_paste_xy(cx: float, cy: float, patch_w: int, patch_h: int, width: int, height: int) -> tuple[int, int]:
    """Convert a candidate center point to an in-image paste top-left."""
    max_x = max(width - patch_w, 0)
    max_y = max(height - patch_h, 0)
    px = int(round(cx - patch_w / 2))
    py = int(round(cy - patch_h / 2))
    return min(max(px, 0), max_x), min(max(py, 0), max_y)


def is_leaf_like_rgb(rgb: tuple[int, int, int]) -> bool:
    """Cheap green-leaf prior used only for offline paste/ROI dataset generation."""
    r, g, b = rgb
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    saturation = (max_channel - min_channel) / max(max_channel, 1)
    return max_channel >= 35 and g >= r * 0.80 and g >= b * 0.75 and (g >= 45 or saturation >= 0.12)


def has_leaf_context(image: Image.Image, px: int, py: int, patch_w: int, patch_h: int) -> bool:
    """Check whether the paste center/corners look like leaf area."""
    samples = (
        (px + patch_w // 2, py + patch_h // 2),
        (px + patch_w // 4, py + patch_h // 4),
        (px + (patch_w * 3) // 4, py + patch_h // 4),
        (px + patch_w // 4, py + (patch_h * 3) // 4),
        (px + (patch_w * 3) // 4, py + (patch_h * 3) // 4),
    )
    hits = 0
    for sx, sy in samples:
        sx = min(max(sx, 0), image.width - 1)
        sy = min(max(sy, 0), image.height - 1)
        if is_leaf_like_rgb(image.getpixel((sx, sy))):
            hits += 1
    return hits >= 2


def sample_near_existing_label(
    labels: list[LabelBox],
    width: int,
    height: int,
    patch_w: int,
    patch_h: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    """Sample a paste location near an existing GT box without forcing overlap."""
    if not labels:
        return None
    anchor = rng.choice(labels)
    x1, y1, x2, y2 = yolo_to_pixels(anchor, width, height)
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    jitter_x = max(patch_w, box_w) * rng.uniform(-3.0, 3.0)
    jitter_y = max(patch_h, box_h) * rng.uniform(-3.0, 3.0)
    cx = (x1 + x2) / 2 + jitter_x
    cy = (y1 + y2) / 2 + jitter_y
    return clamp_paste_xy(cx, cy, patch_w, patch_h, width, height)


def sample_leaf_context_position(
    image: Image.Image,
    patch_w: int,
    patch_h: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    """Sample a paste location inside a simple leaf-color ROI."""
    max_x = max(image.width - patch_w, 0)
    max_y = max(image.height - patch_h, 0)
    for _ in range(20):
        px = rng.randint(0, max_x) if max_x else 0
        py = rng.randint(0, max_y) if max_y else 0
        if has_leaf_context(image, px, py, patch_w, patch_h):
            return px, py
    return None


def build_copy_paste_dataset(
    data_yaml: Path,
    output: Path,
    target_class_ids: set[int],
    copies_per_source: int = 2,
    seed: int = 2026,
    overwrite: bool = False,
) -> Path:
    """Create a derived dataset with simple lesion Copy-Paste samples.

    The source dataset is referenced read-only from the output YAML instead of
    being copied. This keeps local recall experiments feasible when the source
    image set is large.
    """
    info = load_dataset(data_yaml)
    output = output.resolve()
    prepare_output(output, overwrite)
    rng = random.Random(seed)

    train_dir = resolve_split_dir(info, "train")
    train_images = sorted(path for path in train_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    source_items: list[tuple[Path, LabelBox]] = []
    for image in train_images:
        for box in boxes_for_image(image, len(info.names)):
            if box.cls in target_class_ids:
                source_items.append((image, box))

    if not source_items:
        raise RuntimeError("No source boxes found for requested Copy-Paste classes.")

    generated = 0
    backgrounds = train_images or [item[0] for item in source_items]
    for source_image, source_box in source_items:
        with Image.open(source_image) as src_im:
            src = src_im.convert("RGB")
            sx1, sy1, sx2, sy2 = yolo_to_pixels(source_box, *src.size)
            patch = src.crop((sx1, sy1, sx2, sy2))
        if patch.width < 2 or patch.height < 2:
            continue
        for copy_idx in range(copies_per_source):
            bg_image = rng.choice(backgrounds)
            with Image.open(bg_image) as bg_im:
                bg = bg_im.convert("RGB")
            labels = boxes_for_image(bg_image, len(info.names))
            scale = rng.uniform(0.85, 1.25)
            new_size = (max(2, int(patch.width * scale)), max(2, int(patch.height * scale)))
            resized = patch.resize(new_size)
            if rng.random() < 0.5:
                resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            max_x = max(bg.width - resized.width, 0)
            max_y = max(bg.height - resized.height, 0)
            px = rng.randint(0, max_x) if max_x else 0
            py = rng.randint(0, max_y) if max_y else 0
            bg.paste(resized, (px, py))
            new_box = pixel_to_yolo(source_box.cls, px, py, px + resized.width, py + resized.height, bg.width, bg.height)
            if new_box is None:
                continue
            labels.append(new_box)
            stem = f"cp_{source_image.stem}_{generated:05d}_{copy_idx}"
            image_out = output / "images" / "train" / f"{stem}.jpg"
            label_out = output / "labels" / "train" / f"{stem}.txt"
            bg.save(image_out, quality=95)
            label_out.write_text(format_label(labels), encoding="utf-8")
            generated += 1

    generated_train = str((output / "images" / "train").resolve()).replace("\\", "/")
    train_entries = [split_entry(info, "train")]
    if generated:
        train_entries.append(generated_train)
    yaml_path = write_dataset_yaml(
        info,
        output,
        output.name,
        train=train_entries,
        val=split_entry(info, "val"),
        test=split_entry(info, "test"),
    )
    write_recall_manifest(
        info,
        output,
        "copypaste",
        {
            "target_class_ids": sorted(target_class_ids),
            "copies_per_source": copies_per_source,
            "seed": seed,
        },
        {
            "source_boxes": len(source_items),
            "generated_train_images": generated,
            "train_images": count_images(output, "train"),
            "source_train_images": len(train_images),
            "val_images": "source_reference",
            "test_images": "source_reference",
        },
        yaml_path,
    )
    return yaml_path


def build_context_copy_paste_dataset(
    data_yaml: Path,
    output: Path,
    target_class_ids: set[int],
    copy_class_weights: dict[int, int],
    copies_per_source: int = 1,
    max_paste_iou: float = DEFAULT_MAX_PASTE_IOU,
    max_paste_attempts: int = DEFAULT_MAX_PASTE_ATTEMPTS,
    seed: int = 2026,
    overwrite: bool = False,
) -> Path:
    """Create a context-aware Copy-Paste dataset for weak recall classes.

    Compared with the simple Copy-Paste builder, this variant only accepts
    paste positions that are near existing annotated leaf lesions or inside a
    cheap green-leaf ROI, and it rejects boxes that overlap existing labels too
    strongly. This keeps the augmentation closer to the paper's background and
    multi-instance failure modes without requiring SAM2 at build time.
    """
    info = load_dataset(data_yaml)
    output = output.resolve()
    prepare_output(output, overwrite)
    rng = random.Random(seed)

    train_dir = resolve_split_dir(info, "train")
    train_images = sorted(path for path in train_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    source_items: list[tuple[Path, LabelBox]] = []
    for image in train_images:
        for box in boxes_for_image(image, len(info.names)):
            if box.cls in target_class_ids:
                source_items.append((image, box))

    if not source_items:
        raise RuntimeError("No source boxes found for requested context Copy-Paste classes.")

    generated = 0
    scheduled = 0
    accepted = 0
    rejected_attempts = 0
    failed_scheduled = 0
    backgrounds = train_images or [item[0] for item in source_items]
    for source_image, source_box in source_items:
        with Image.open(source_image) as src_im:
            src = src_im.convert("RGB")
            sx1, sy1, sx2, sy2 = yolo_to_pixels(source_box, *src.size)
            patch = src.crop((sx1, sy1, sx2, sy2))
        if patch.width < 2 or patch.height < 2:
            continue

        class_weight = max(int(copy_class_weights.get(source_box.cls, 1)), 1)
        copies_for_box = max(copies_per_source, 0) * class_weight
        for copy_idx in range(copies_for_box):
            scheduled += 1
            scale = rng.uniform(0.85, 1.25)
            new_size = (max(2, int(patch.width * scale)), max(2, int(patch.height * scale)))
            resized = patch.resize(new_size)
            if rng.random() < 0.5:
                resized = resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            bg_image = rng.choice(backgrounds)
            with Image.open(bg_image) as bg_im:
                bg = bg_im.convert("RGB")
            labels = boxes_for_image(bg_image, len(info.names))
            accepted_box: LabelBox | None = None
            accepted_xy: tuple[int, int] | None = None
            for attempt in range(max_paste_attempts):
                if attempt % 2 == 0:
                    candidate_xy = sample_near_existing_label(labels, bg.width, bg.height, resized.width, resized.height, rng)
                else:
                    candidate_xy = sample_leaf_context_position(bg, resized.width, resized.height, rng)
                if candidate_xy is None:
                    candidate_xy = sample_leaf_context_position(bg, resized.width, resized.height, rng)
                if candidate_xy is None:
                    rejected_attempts += 1
                    continue

                px, py = candidate_xy
                candidate_pixels = (px, py, px + resized.width, py + resized.height)
                if max_iou_with_labels(candidate_pixels, labels, bg.width, bg.height) > max_paste_iou:
                    rejected_attempts += 1
                    continue
                if not labels and not has_leaf_context(bg, px, py, resized.width, resized.height):
                    rejected_attempts += 1
                    continue
                accepted_box = pixel_to_yolo(
                    source_box.cls,
                    px,
                    py,
                    px + resized.width,
                    py + resized.height,
                    bg.width,
                    bg.height,
                )
                if accepted_box is None:
                    rejected_attempts += 1
                    continue
                accepted_xy = px, py
                break

            if accepted_box is None or accepted_xy is None:
                failed_scheduled += 1
                continue

            px, py = accepted_xy
            bg.paste(resized, (px, py))
            labels.append(accepted_box)
            stem = f"cpc_{source_image.stem}_{generated:05d}_{copy_idx}"
            image_out = output / "images" / "train" / f"{stem}.jpg"
            label_out = output / "labels" / "train" / f"{stem}.txt"
            bg.save(image_out, quality=95)
            label_out.write_text(format_label(labels), encoding="utf-8")
            generated += 1
            accepted += 1

    generated_train = str((output / "images" / "train").resolve()).replace("\\", "/")
    train_entries = [split_entry(info, "train")]
    if generated:
        train_entries.append(generated_train)
    yaml_path = write_dataset_yaml(
        info,
        output,
        output.name,
        train=train_entries,
        val=split_entry(info, "val"),
        test=split_entry(info, "test"),
    )
    write_recall_manifest(
        info,
        output,
        "copypaste_context",
        {
            "target_class_ids": sorted(target_class_ids),
            "copy_class_weights": {str(cls): int(weight) for cls, weight in sorted(copy_class_weights.items())},
            "copies_per_source": copies_per_source,
            "max_paste_iou": max_paste_iou,
            "max_paste_attempts": max_paste_attempts,
            "seed": seed,
            "paste_context": "near_existing_gt_or_leaf_roi",
        },
        {
            "source_boxes": len(source_items),
            "scheduled_pastes": scheduled,
            "accepted_pastes": accepted,
            "rejected_paste_attempts": rejected_attempts,
            "failed_scheduled_pastes": failed_scheduled,
            "generated_train_images": generated,
            "train_images": count_images(output, "train"),
            "source_train_images": len(train_images),
            "val_images": "source_reference",
            "test_images": "source_reference",
        },
        yaml_path,
    )
    return yaml_path


def clipped_box_for_tile(box: LabelBox, image_w: int, image_h: int, tx: int, ty: int, tile_w: int, tile_h: int, min_visibility: float) -> LabelBox | None:
    """Clip a box into a tile and return tile-normalized YOLO coordinates."""
    x1, y1, x2, y2 = yolo_to_pixels(box, image_w, image_h)
    ix1 = max(x1, tx)
    iy1 = max(y1, ty)
    ix2 = min(x2, tx + tile_w)
    iy2 = min(y2, ty + tile_h)
    original_area = max((x2 - x1) * (y2 - y1), 1)
    visible_area = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    if visible_area / original_area < min_visibility:
        return None
    return pixel_to_yolo(box.cls, ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty, tile_w, tile_h)


def tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    """Return tile origins that cover one image dimension."""
    if length <= tile_size:
        return [0]
    stride = max(int(tile_size * (1.0 - overlap)), 1)
    origins = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def build_sahi_dataset(
    data_yaml: Path,
    output: Path,
    tile_size: int = 640,
    overlap: float = 0.20,
    splits: tuple[str, ...] = ("train", "val", "test"),
    min_visibility: float = 0.30,
    overwrite: bool = False,
) -> Path:
    """Create a derived sliced dataset for small-object recall experiments."""
    info = load_dataset(data_yaml)
    output = output.resolve()
    prepare_output(output, overwrite)

    for split in splits:
        image_dir = resolve_split_dir(info, split)
        for image in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
            boxes = boxes_for_image(image, len(info.names))
            with Image.open(image) as im:
                src = im.convert("RGB")
            xs = tile_origins(src.width, tile_size, overlap)
            ys = tile_origins(src.height, tile_size, overlap)
            for ty in ys:
                for tx in xs:
                    tile_w = min(tile_size, src.width - tx)
                    tile_h = min(tile_size, src.height - ty)
                    labels = [
                        clipped
                        for box in boxes
                        if (clipped := clipped_box_for_tile(box, src.width, src.height, tx, ty, tile_w, tile_h, min_visibility))
                        is not None
                    ]
                    if not labels and split == "train":
                        continue
                    stem = f"{image.stem}_x{tx}_y{ty}"
                    image_out = output / "images" / split / f"{stem}.jpg"
                    label_out = output / "labels" / split / f"{stem}.txt"
                    image_out.parent.mkdir(parents=True, exist_ok=True)
                    label_out.parent.mkdir(parents=True, exist_ok=True)
                    src.crop((tx, ty, tx + tile_w, ty + tile_h)).save(image_out, quality=95)
                    label_out.write_text(format_label(labels), encoding="utf-8")

    yaml_path = write_dataset_yaml(info, output, output.name)
    write_recall_manifest(
        info,
        output,
        "sahi",
        {
            "tile_size": tile_size,
            "overlap": overlap,
            "splits": list(splits),
            "min_visibility": min_visibility,
        },
        {
            "generated_train_images": count_images(output, "train"),
            "generated_val_images": count_images(output, "val"),
            "generated_test_images": count_images(output, "test"),
        },
        yaml_path,
    )
    return yaml_path


def build_sahi_trainonly_dataset(
    data_yaml: Path,
    output: Path,
    tile_size: int = 640,
    overlap: float = 0.20,
    min_visibility: float = 0.30,
    overwrite: bool = False,
) -> Path:
    """Create train-only sliced data while validating on original full images."""
    info = load_dataset(data_yaml)
    output = output.resolve()
    prepare_output(output, overwrite)

    split = "train"
    image_dir = resolve_split_dir(info, split)
    for image in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
        boxes = boxes_for_image(image, len(info.names))
        with Image.open(image) as im:
            src = im.convert("RGB")
        xs = tile_origins(src.width, tile_size, overlap)
        ys = tile_origins(src.height, tile_size, overlap)
        for ty in ys:
            for tx in xs:
                tile_w = min(tile_size, src.width - tx)
                tile_h = min(tile_size, src.height - ty)
                labels = [
                    clipped
                    for box in boxes
                    if (clipped := clipped_box_for_tile(box, src.width, src.height, tx, ty, tile_w, tile_h, min_visibility))
                    is not None
                ]
                if not labels:
                    continue
                stem = f"{image.stem}_x{tx}_y{ty}"
                image_out = output / "images" / split / f"{stem}.jpg"
                label_out = output / "labels" / split / f"{stem}.txt"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                src.crop((tx, ty, tx + tile_w, ty + tile_h)).save(image_out, quality=95)
                label_out.write_text(format_label(labels), encoding="utf-8")

    yaml_path = write_dataset_yaml(
        info,
        output,
        output.name,
        train=str((output / "images" / "train").resolve()).replace("\\", "/"),
        val=split_entry(info, "val"),
        test=split_entry(info, "test"),
    )
    write_recall_manifest(
        info,
        output,
        "sahi_trainonly",
        {
            "tile_size": tile_size,
            "overlap": overlap,
            "splits": ["train"],
            "min_visibility": min_visibility,
            "eval_splits": "source_reference",
            "drop_empty_train_tiles": True,
        },
        {
            "generated_train_images": count_images(output, "train"),
            "val_images": "source_reference",
            "test_images": "source_reference",
        },
        yaml_path,
    )
    return yaml_path


def build_leaf_mask(rgb: Image.Image, boxes: list[LabelBox]) -> Image.Image:
    """Build a lightweight leaf ROI mask and force GT boxes to remain visible."""
    mask = Image.new("L", rgb.size, 0)
    in_pixels = rgb.load()
    out_pixels = mask.load()
    for y in range(rgb.height):
        for x in range(rgb.width):
            if is_leaf_like_rgb(in_pixels[x, y]):
                out_pixels[x, y] = 255

    mask = mask.filter(ImageFilter.MaxFilter(11)).filter(ImageFilter.MedianFilter(5))
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_pixels(box, rgb.width, rgb.height)
        margin = max(4, int(max(x2 - x1, y2 - y1) * 0.25))
        draw.rectangle(
            (
                max(x1 - margin, 0),
                max(y1 - margin, 0),
                min(x2 + margin, rgb.width),
                min(y2 + margin, rgb.height),
            ),
            fill=255,
        )
    return mask


def apply_leafmask_roi(rgb: Image.Image, boxes: list[LabelBox]) -> Image.Image:
    """Suppress non-leaf background while preserving annotated object regions."""
    mask = build_leaf_mask(rgb, boxes)
    foreground = ImageEnhance.Contrast(rgb).enhance(1.10).filter(ImageFilter.SHARPEN)
    muted = ImageEnhance.Color(rgb).enhance(0.35)
    muted = ImageEnhance.Brightness(muted).enhance(0.70)
    return Image.composite(foreground, muted, mask)


def build_roi_dataset(data_yaml: Path, output: Path, overwrite: bool = False) -> Path:
    """Create a lightweight ROI-enhanced dataset using contrast and edge cues.

    This is a local stand-in for offline SAM2/U-Net mask preprocessing: it
    avoids an external model dependency while preserving the same experiment
    contract, namely a separate derived dataset with reduced background impact.
    """
    info = load_dataset(data_yaml)
    output = output.resolve()
    prepare_output(output, overwrite)
    train_dir = resolve_split_dir(info, "train")
    generated = 0
    for image in sorted(path for path in train_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
        rel = image.relative_to(train_dir)
        image_out = output / "images" / "train" / rel
        label_out = output / "labels" / "train" / rel.with_suffix(".txt")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image) as im:
            rgb = im.convert("RGB")
        enhanced = ImageEnhance.Contrast(rgb).enhance(1.25).filter(ImageFilter.SHARPEN)
        enhanced.save(image_out, quality=95)
        src_label = image_to_label_path(image)
        if src_label.exists():
            shutil.copy2(src_label, label_out)
        generated += 1
    yaml_path = write_dataset_yaml(
        info,
        output,
        output.name,
        train=str((output / "images" / "train").resolve()).replace("\\", "/"),
        val=split_entry(info, "val"),
        test=split_entry(info, "test"),
    )
    write_recall_manifest(
        info,
        output,
        "roi",
        {"contrast": 1.25, "filter": "SHARPEN"},
        {
            "generated_train_images": generated,
            "val_images": "source_reference",
            "test_images": "source_reference",
        },
        yaml_path,
    )
    return yaml_path


def build_roi_leafmask_dataset(data_yaml: Path, output: Path, overwrite: bool = False) -> Path:
    """Create a leaf-mask ROI dataset that suppresses cluttered background."""
    info = load_dataset(data_yaml)
    output = output.resolve()
    prepare_output(output, overwrite)
    train_dir = resolve_split_dir(info, "train")
    generated = 0
    copied_labels = 0
    for image in sorted(path for path in train_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
        rel = image.relative_to(train_dir)
        image_out = output / "images" / "train" / rel
        label_out = output / "labels" / "train" / rel.with_suffix(".txt")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        boxes = boxes_for_image(image, len(info.names))
        with Image.open(image) as im:
            rgb = im.convert("RGB")
        enhanced = apply_leafmask_roi(rgb, boxes)
        enhanced.save(image_out, quality=95)
        src_label = image_to_label_path(image)
        if src_label.exists():
            shutil.copy2(src_label, label_out)
            copied_labels += 1
        generated += 1
    yaml_path = write_dataset_yaml(
        info,
        output,
        output.name,
        train=str((output / "images" / "train").resolve()).replace("\\", "/"),
        val=split_entry(info, "val"),
        test=split_entry(info, "test"),
    )
    write_recall_manifest(
        info,
        output,
        "roi_leafmask",
        {
            "leaf_mask": "rgb_green_prior_maxfilter11_median5",
            "preserve_gt_boxes": True,
            "foreground_contrast": 1.10,
            "background_color": 0.35,
            "background_brightness": 0.70,
        },
        {
            "generated_train_images": generated,
            "copied_train_labels": copied_labels,
            "val_images": "source_reference",
            "test_images": "source_reference",
        },
        yaml_path,
    )
    return yaml_path


def split_entries(info: DatasetInfo, split: str) -> list[Path]:
    """Return resolved image directories for a split, including list-style train entries."""
    return resolve_split_dirs(info, split)


def build_sanitize_labels_dataset(
    data_yaml: Path,
    output: Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    min_box_area: float = DEFAULT_MIN_SANITIZED_BOX_AREA,
    overwrite: bool = False,
) -> Path:
    """Copy a dataset and clip invalid/out-of-bounds YOLO boxes.

    This is intentionally geometry-only. It fixes labels that extend past image
    edges before any heavier SAM/manual refinement stage, while keeping the
    source dataset read-only.
    """
    info = load_dataset(data_yaml)
    output = output.resolve()
    prepare_output(output, overwrite)

    copied_images = 0
    copied_labels = 0
    clipped_boxes = 0
    dropped_boxes = 0
    missing_labels = 0
    invalid_rows = 0
    per_split_images: dict[str, int] = {}

    for split in splits:
        split_image_count = 0
        for image_dir in split_entries(info, split):
            for image in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
                rel = image.relative_to(image_dir)
                image_out = output / "images" / split / rel
                label_out = output / "labels" / split / rel.with_suffix(".txt")
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, image_out)
                copied_images += 1
                split_image_count += 1

                src_label = image_to_label_path(image)
                if not src_label.exists():
                    missing_labels += 1
                    label_out.write_text("", encoding="utf-8")
                    continue

                boxes, warnings = read_labels(src_label, len(info.names))
                invalid_rows += len(warnings)
                fixed_boxes: list[LabelBox] = []
                for box in boxes:
                    fixed, was_clipped = clipped_normalized_box(box, min_box_area)
                    if fixed is None:
                        dropped_boxes += 1
                        continue
                    if was_clipped:
                        clipped_boxes += 1
                    fixed_boxes.append(fixed)
                label_out.write_text(format_label(fixed_boxes), encoding="utf-8")
                copied_labels += 1
        per_split_images[split] = split_image_count

    yaml_path = write_dataset_yaml(info, output, output.name)
    write_recall_manifest(
        info,
        output,
        "sanitize_labels",
        {
            "splits": list(splits),
            "min_box_area": min_box_area,
            "operation": "clip_out_of_bounds_yolo_boxes",
        },
        {
            "copied_images": copied_images,
            "copied_labels": copied_labels,
            "missing_labels": missing_labels,
            "invalid_label_rows_from_parser": invalid_rows,
            "clipped_boxes": clipped_boxes,
            "dropped_boxes": dropped_boxes,
            "split_images": per_split_images,
        },
        yaml_path,
    )
    return yaml_path


def parse_class_ids(raw: str) -> set[int]:
    """Parse comma-separated class ids."""
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def parse_class_weights(raw: str) -> dict[int, int]:
    """Parse comma-separated class weight pairs, e.g. 1:1,2:3."""
    weights: dict[int, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        cls_raw, sep, weight_raw = item.partition(":")
        if sep == "":
            raise ValueError(f"Invalid class weight {item!r}; expected class_id:weight.")
        cls = int(cls_raw.strip())
        weight = int(weight_raw.strip())
        if weight <= 0:
            raise ValueError(f"Class weight must be positive for class {cls}.")
        weights[cls] = weight
    return weights


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--variant",
        choices=("sanitize_labels", "copypaste", "copypaste_context", "sahi", "sahi_trainonly", "roi", "roi_leafmask"),
        required=True,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--target-classes",
        default=DEFAULT_COPY_PASTE_TARGET_CLASSES,
        help="Copy-Paste target class ids. Default targets low-recall classes: ANT_CD, BLS_AB, BLS_CD, CR_AB.",
    )
    parser.add_argument(
        "--copy-class-weights",
        default=DEFAULT_CONTEXT_COPY_CLASS_WEIGHTS,
        help="Context Copy-Paste class weights, e.g. 1:1,2:3,3:2,5:3.",
    )
    parser.add_argument("--copies-per-source", type=int, default=None)
    parser.add_argument("--max-paste-iou", type=float, default=DEFAULT_MAX_PASTE_IOU)
    parser.add_argument("--max-paste-attempts", type=int, default=DEFAULT_MAX_PASTE_ATTEMPTS)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--min-visibility", type=float, default=0.30)
    parser.add_argument("--min-box-area", type=float, default=DEFAULT_MIN_SANITIZED_BOX_AREA)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def build_variant(variant: str, data: Path, output: Path, args: argparse.Namespace | None = None) -> Path:
    """Build one recall derived dataset variant."""
    if variant == "sanitize_labels":
        return build_sanitize_labels_dataset(
            data,
            output,
            min_box_area=float(getattr(args, "min_box_area", DEFAULT_MIN_SANITIZED_BOX_AREA)),
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    if variant == "copypaste":
        copies_per_source = getattr(args, "copies_per_source", None)
        return build_copy_paste_dataset(
            data,
            output,
            target_class_ids=parse_class_ids(getattr(args, "target_classes", DEFAULT_COPY_PASTE_TARGET_CLASSES)),
            copies_per_source=2 if copies_per_source is None else int(copies_per_source),
            seed=int(getattr(args, "seed", 2026)),
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    if variant == "copypaste_context":
        copies_per_source = getattr(args, "copies_per_source", None)
        return build_context_copy_paste_dataset(
            data,
            output,
            target_class_ids=parse_class_ids(getattr(args, "target_classes", DEFAULT_COPY_PASTE_TARGET_CLASSES)),
            copy_class_weights=parse_class_weights(getattr(args, "copy_class_weights", DEFAULT_CONTEXT_COPY_CLASS_WEIGHTS)),
            copies_per_source=1 if copies_per_source is None else int(copies_per_source),
            max_paste_iou=float(getattr(args, "max_paste_iou", DEFAULT_MAX_PASTE_IOU)),
            max_paste_attempts=int(getattr(args, "max_paste_attempts", DEFAULT_MAX_PASTE_ATTEMPTS)),
            seed=int(getattr(args, "seed", 2026)),
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    if variant == "sahi":
        return build_sahi_dataset(
            data,
            output,
            tile_size=int(getattr(args, "tile_size", 640)),
            overlap=float(getattr(args, "overlap", 0.20)),
            min_visibility=float(getattr(args, "min_visibility", 0.30)),
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    if variant == "sahi_trainonly":
        return build_sahi_trainonly_dataset(
            data,
            output,
            tile_size=int(getattr(args, "tile_size", 640)),
            overlap=float(getattr(args, "overlap", 0.20)),
            min_visibility=float(getattr(args, "min_visibility", 0.30)),
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    if variant == "roi":
        return build_roi_dataset(data, output, overwrite=bool(getattr(args, "overwrite", False)))
    if variant == "roi_leafmask":
        return build_roi_leafmask_dataset(data, output, overwrite=bool(getattr(args, "overwrite", False)))
    raise ValueError(f"Unsupported recall dataset variant: {variant}")


def main() -> None:
    args = parse_args()
    output = args.output or DEFAULT_OUTPUT_ROOT / f"{Path(args.data).stem}_{args.variant}"
    yaml_path = build_variant(args.variant, args.data, output, args)
    print(f"Generated dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
