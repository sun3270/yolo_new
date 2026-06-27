"""Recall-focused diagnostics for YOLO coffee datasets.

The script is intentionally read-only for the source dataset. It writes a
compact report that helps separate true missed detections from threshold/NMS
issues before changing model structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image, ImageFilter


IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
HARD_CASE_TAGS = ("small_edge", "clutter_bg", "multi_instance", "tail_class", "bad_border")


@dataclass(frozen=True)
class DatasetInfo:
    yaml_path: Path
    root: Path
    cfg: dict
    names: list[str]
    path_remaps: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LabelBox:
    cls: int
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def touches_border(self) -> bool:
        return self.x - self.w / 2 <= 0.0 or self.x + self.w / 2 >= 1.0 or self.y - self.h / 2 <= 0.0 or self.y + self.h / 2 >= 1.0


@dataclass(frozen=True)
class ImageSample:
    split: str
    image: Path
    label: Path
    boxes: list[LabelBox]
    width: int
    height: int
    edge_density: float


def _normalize_path_text(value: str) -> str:
    """Normalize path text for portable prefix matching without touching the filesystem."""
    return value.replace("\\", "/").rstrip("/")


def parse_path_remaps(raw: str | None = None) -> tuple[tuple[str, str], ...]:
    """Parse COFFEE_PATH_REMAP-style path remaps.

    Format: OLD=>NEW;OLD2=>NEW2. A plain OLD=NEW pair is also accepted.
    This is for moving YAMLs between local paths such as C:/Users/1/Desktop/coffee
    and UNC paths such as //BRUCE/Users/1/Desktop/coffee.
    """
    raw = os.environ.get("COFFEE_PATH_REMAP", "") if raw is None else raw
    remaps: list[tuple[str, str]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        sep = "=>" if "=>" in item else "="
        old, _, new = item.partition(sep)
        old_norm = _normalize_path_text(old.strip())
        new_norm = _normalize_path_text(new.strip())
        if old_norm and new_norm:
            remaps.append((old_norm, new_norm))
    return tuple(remaps)


def apply_path_remap(value: str | Path, remaps: tuple[tuple[str, str], ...] | None = None) -> str:
    """Apply the first matching path prefix remap to a path-like string."""
    text = _normalize_path_text(str(value))
    active_remaps = parse_path_remaps() if remaps is None else remaps
    text_cmp = text.lower()
    for old, new in active_remaps:
        old_cmp = old.lower()
        if text_cmp == old_cmp:
            return new
        if text_cmp.startswith(old_cmp + "/"):
            suffix = text[len(old) :].lstrip("/")
            return f"{new}/{suffix}" if suffix else new
    return text


def load_dataset(data_yaml: Path, path_remap: str | None = None) -> DatasetInfo:
    """Load a YOLO dataset YAML and resolve relative split paths."""
    data_yaml = data_yaml.resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    remaps = parse_path_remaps(path_remap)
    root = Path(apply_path_remap(cfg.get("path", data_yaml.parent), remaps))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    names_raw = cfg.get("names", {})
    nc = int(cfg.get("nc", len(names_raw)))
    if isinstance(names_raw, dict):
        names = [str(names_raw.get(i, names_raw.get(str(i), i))) for i in range(nc)]
    else:
        names = [str(names_raw[i]) if i < len(names_raw) else str(i) for i in range(nc)]
    return DatasetInfo(yaml_path=data_yaml, root=root, cfg=cfg, names=names, path_remaps=remaps)


def resolve_split_dirs(info: DatasetInfo, split: str) -> list[Path]:
    """Resolve image directories/files for a YOLO split."""
    split_value = info.cfg[split]
    values = split_value if isinstance(split_value, list) else [split_value]
    paths = []
    for value in values:
        path = Path(apply_path_remap(str(value), info.path_remaps))
        paths.append(path if path.is_absolute() else info.root / path)
    return paths


def resolve_split_dir(info: DatasetInfo, split: str) -> Path:
    """Resolve the first image directory for a YOLO split."""
    return resolve_split_dirs(info, split)[0]


def image_to_label_path(image: Path) -> Path:
    """Map an image path under images/ to its YOLO label path under labels/."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return (image.parent.parent / "labels" / image.name).with_suffix(".txt")


def read_labels(path: Path, nc: int) -> tuple[list[LabelBox], list[str]]:
    """Read YOLO labels and return valid boxes plus warnings."""
    warnings: list[str] = []
    boxes: list[LabelBox] = []
    if not path.exists():
        return boxes, [f"missing label: {path}"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) < 5:
            warnings.append(f"{path}:{line_no} has fewer than 5 fields")
            continue
        try:
            cls = int(float(fields[0]))
            x, y, w, h = (float(value) for value in fields[1:5])
        except ValueError:
            warnings.append(f"{path}:{line_no} contains non-numeric values")
            continue
        if not 0 <= cls < nc:
            warnings.append(f"{path}:{line_no} class {cls} outside 0..{nc - 1}")
            continue
        if w <= 0.0 or h <= 0.0:
            warnings.append(f"{path}:{line_no} has non-positive width/height")
            continue
        if any(value < 0.0 or value > 1.0 for value in (x, y, w, h)):
            warnings.append(f"{path}:{line_no} has values outside 0..1")
        boxes.append(LabelBox(cls=cls, x=x, y=y, w=w, h=h))
    return boxes, warnings


def estimate_edge_density(image: Path, resize: int = 96) -> float:
    """Return a lightweight edge-density proxy for background clutter."""
    with Image.open(image) as im:
        gray = im.convert("L")
        gray.thumbnail((resize, resize))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        data_getter = getattr(edges, "get_flattened_data", edges.getdata)
        values = list(data_getter())
    if not values:
        return 0.0
    return sum(1 for value in values if value >= 32) / len(values)


def iter_samples(info: DatasetInfo, splits: Iterable[str]) -> tuple[list[ImageSample], list[str]]:
    """Collect image samples and label metadata for requested splits."""
    samples: list[ImageSample] = []
    warnings: list[str] = []
    for split in splits:
        image_dirs = resolve_split_dirs(info, split)
        for image_dir in image_dirs:
            if not image_dir.exists():
                warnings.append(f"missing split image directory: {image_dir}")
                continue
            if image_dir.is_file() and image_dir.suffix.lower() == ".txt":
                images = []
                for line in image_dir.read_text(encoding="utf-8").splitlines():
                    image_path = Path(line.strip())
                    if image_path and not image_path.is_absolute():
                        image_path = (image_dir.parent / image_path).resolve()
                    if image_path.suffix.lower() in IMAGE_EXTS:
                        images.append(image_path)
            elif image_dir.is_file() and image_dir.suffix.lower() in IMAGE_EXTS:
                images = [image_dir]
            else:
                images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
            for image in images:
                label = image_to_label_path(image)
                boxes, label_warnings = read_labels(label, len(info.names))
                warnings.extend(label_warnings)
                with Image.open(image) as im:
                    width, height = im.size
                samples.append(
                    ImageSample(
                        split=split,
                        image=image,
                        label=label,
                        boxes=boxes,
                        width=width,
                        height=height,
                        edge_density=estimate_edge_density(image),
                    )
                )
    return samples, warnings


def classify_sample(
    sample: ImageSample,
    tail_class_ids: set[int],
    small_area_threshold: float,
    edge_margin: float,
    multi_instance_threshold: int,
    clutter_edge_density_threshold: float,
) -> set[str]:
    """Assign recall-oriented hard-case tags to one image."""
    tags: set[str] = set()
    if any(box.area <= small_area_threshold for box in sample.boxes):
        tags.add("small_edge")
    if any(box.cls in tail_class_ids for box in sample.boxes):
        tags.add("tail_class")
    if len(sample.boxes) >= multi_instance_threshold:
        tags.add("multi_instance")
    if sample.edge_density >= clutter_edge_density_threshold:
        tags.add("clutter_bg")
    for box in sample.boxes:
        left = box.x - box.w / 2
        right = box.x + box.w / 2
        top = box.y - box.h / 2
        bottom = box.y + box.h / 2
        if min(left, top, 1.0 - right, 1.0 - bottom) <= edge_margin:
            tags.add("bad_border")
            break
    return tags


def summarize_box_areas(samples: list[ImageSample]) -> dict[str, float]:
    """Return stable area summary for all boxes."""
    areas = [box.area for sample in samples for box in sample.boxes]
    if not areas:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
    sorted_areas = sorted(areas)

    def pct(q: float) -> float:
        index = min(max(math.ceil(q * len(sorted_areas)) - 1, 0), len(sorted_areas) - 1)
        return sorted_areas[index]

    return {
        "min": min(sorted_areas),
        "p25": pct(0.25),
        "median": statistics.median(sorted_areas),
        "p75": pct(0.75),
        "max": max(sorted_areas),
    }


def write_paper_requirements_report(report: dict, output_dir: Path) -> Path:
    """Write a compact checklist mapped to the yolo.md recall failure modes."""
    hard_counts = report["hard_case_counts"]
    lines = [
        "# yolo.md Recall Requirements Check",
        "",
        "This report maps the dataset scan to the three miss-detection causes in yolo.md.",
        "",
        "## 标注边缘噪声",
        f"- bad_border images: {hard_counts['bad_border']}",
        f"- small_edge images: {hard_counts['small_edge']}",
        "- Suggested action: inspect loose or border-touching labels before changing the model.",
        "",
        "## 复杂背景干扰",
        f"- clutter_bg images: {hard_counts['clutter_bg']}",
        "- Suggested action: compare original, CLAHE/Laplacian/ROI, and SAHI-derived datasets.",
        "",
        "## 多主体遮挡",
        f"- multi_instance images: {hard_counts['multi_instance']}",
        "- Suggested action: run recall sweep with lower conf, higher max_det, and explicit NMS status.",
        "",
        "## 尾部类别与小目标",
        f"- tail_class images: {hard_counts['tail_class']}",
        f"- tail class ids: {report['thresholds']['tail_class_ids']}",
        "- Suggested action: use Copy-Paste for CPD/CR_CD/SM_AB/SM_CD or configured tail classes.",
        "",
        "## Label Quality",
        f"- warning_count: {report['label_quality']['warning_count']}",
        f"- warning_samples: {len(report['label_quality']['warning_samples'])}",
        "",
    ]
    path = output_dir / "paper_requirements_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def sample_display_paths(info: DatasetInfo, sample: ImageSample) -> tuple[str, str]:
    """Return stable report paths even when split entries point outside cfg['path']."""
    try:
        rel_image = sample.image.relative_to(info.root).as_posix()
    except ValueError:
        rel_image = sample.image.as_posix()
        for image_dir in resolve_split_dirs(info, sample.split):
            try:
                rel = sample.image.relative_to(image_dir)
            except ValueError:
                continue
            rel_image = (Path("images") / sample.split / rel).as_posix()
            break

    if sample.label.exists():
        try:
            rel_label = sample.label.relative_to(info.root).as_posix()
        except ValueError:
            if rel_image.startswith("images/"):
                rel_label = (Path("labels") / Path(rel_image).relative_to("images")).with_suffix(".txt").as_posix()
            else:
                rel_label = sample.label.as_posix()
    else:
        rel_label = sample.label.as_posix()
    return rel_image, rel_label


def analyze_dataset(
    data_yaml: Path,
    output_dir: Path,
    splits: Iterable[str] = ("train", "val", "test"),
    tail_class_ids: set[int] | None = None,
    small_area_threshold: float = 0.0025,
    edge_margin: float = 0.03,
    multi_instance_threshold: int = 5,
    clutter_edge_density_threshold: float = 0.35,
) -> dict:
    """Analyze a YOLO dataset and write CSV/JSON reports."""
    info = load_dataset(data_yaml)
    samples, warnings = iter_samples(info, splits)
    tail_class_ids = set() if tail_class_ids is None else set(tail_class_ids)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summary = {
        split: {
            "images": sum(1 for sample in samples if sample.split == split),
            "instances": sum(len(sample.boxes) for sample in samples if sample.split == split),
        }
        for split in splits
    }
    class_counts = {str(i): {"name": name, "count": 0} for i, name in enumerate(info.names)}
    hard_counts = {tag: 0 for tag in HARD_CASE_TAGS}
    hard_case_images = {tag: [] for tag in HARD_CASE_TAGS}
    image_object_count: dict[str, int] = {}

    hard_rows = []
    for sample in samples:
        for box in sample.boxes:
            class_counts[str(box.cls)]["count"] += 1
        rel_image, rel_label = sample_display_paths(info, sample)
        image_object_count[rel_image] = len(sample.boxes)
        tags = classify_sample(
            sample,
            tail_class_ids,
            small_area_threshold,
            edge_margin,
            multi_instance_threshold,
            clutter_edge_density_threshold,
        )
        for tag in tags:
            hard_counts[tag] += 1
            hard_case_images[tag].append(rel_image)
        hard_rows.append(
            {
                "split": sample.split,
                "image": rel_image,
                "label": rel_label,
                "objects": str(len(sample.boxes)),
                "classes": ";".join(sorted({info.names[box.cls] for box in sample.boxes})),
                "min_area": f"{min((box.area for box in sample.boxes), default=0.0):.8f}",
                "edge_density": f"{sample.edge_density:.6f}",
                "hard_case_tags": ";".join(tag for tag in HARD_CASE_TAGS if tag in tags),
            }
        )

    with (output_dir / "hard_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("split", "image", "label", "objects", "classes", "min_area", "edge_density", "hard_case_tags"),
        )
        writer.writeheader()
        writer.writerows(hard_rows)

    report = {
        "data_yaml": str(info.yaml_path),
        "dataset_root": str(info.root),
        "names": info.names,
        "splits": split_summary,
        "class_counts": class_counts,
        "box_area": summarize_box_areas(samples),
        "hard_case_counts": hard_counts,
        "hard_case_images": hard_case_images,
        "image_object_count": image_object_count,
        "label_quality": {
            "warning_count": len(warnings),
            "warning_samples": warnings[:50],
        },
        "warnings": warnings,
        "thresholds": {
            "small_area_threshold": small_area_threshold,
            "edge_margin": edge_margin,
            "multi_instance_threshold": multi_instance_threshold,
            "clutter_edge_density_threshold": clutter_edge_density_threshold,
            "tail_class_ids": sorted(tail_class_ids),
        },
    }
    (output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_paper_requirements_report(report, output_dir)
    return report


def parse_class_ids(raw: str | None) -> set[int]:
    """Parse comma-separated class ids."""
    if raw is None or raw.strip() == "":
        return set()
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML.")
    parser.add_argument("--output", type=Path, default=Path("runs/coffee_v4_diagnostics"), help="Report output folder.")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to scan.")
    parser.add_argument("--tail-classes", default="", help="Comma-separated tail class ids.")
    parser.add_argument("--small-area-threshold", type=float, default=0.0025)
    parser.add_argument("--edge-margin", type=float, default=0.03)
    parser.add_argument("--multi-instance-threshold", type=int, default=5)
    parser.add_argument("--clutter-edge-density-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_dataset(
        args.data,
        args.output,
        splits=tuple(split.strip() for split in args.splits.split(",") if split.strip()),
        tail_class_ids=parse_class_ids(args.tail_classes),
        small_area_threshold=args.small_area_threshold,
        edge_margin=args.edge_margin,
        multi_instance_threshold=args.multi_instance_threshold,
        clutter_edge_density_threshold=args.clutter_edge_density_threshold,
    )
    print(f"Images: {sum(split['images'] for split in report['splits'].values())}")
    print(f"Instances: {sum(split['instances'] for split in report['splits'].values())}")
    print(f"Hard cases: {report['hard_case_counts']}")
    print(f"Report: {Path(args.output).resolve() / 'dataset_report.json'}")


if __name__ == "__main__":
    main()
