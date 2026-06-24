"""Build V4 preprocessing-only dataset variants for coffee leaf disease experiments.

Usage:
    python elteb_experiment/build_preprocessed_dataset.py --variant clahe
    python elteb_experiment/build_preprocessed_dataset.py --variant laplacian
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_YAML = ROOT / "coffee_self_sum" / "coffee_self_sum.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "elteb_experiment" / "preprocessed_data"
IMAGE_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def apply_clahe(bgr):
    """Enhance local contrast on Lab-L while keeping RGB-style three-channel input."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def apply_laplacian(bgr, alpha=0.35):
    """Add a Laplacian high-pass map back to the image as a preprocessing-only edge cue."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    lap_bgr = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(bgr, 1.0, lap_bgr, alpha, 0)


def load_dataset_yaml(path: Path) -> dict:
    """Load a YOLO dataset YAML file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_dir(dataset_root: Path, split_value: str) -> Path:
    """Resolve a YOLO split path relative to the dataset YAML directory."""
    split_path = Path(split_value)
    return split_path if split_path.is_absolute() else dataset_root / split_path


def iter_images(images_dir: Path):
    """Yield image paths under a split image directory."""
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def image_to_label_path(image_path: Path) -> Path:
    """Map a YOLO image path to its matching label path."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return (image_path.parent.parent / "labels" / image_path.name).with_suffix(".txt")


def transform_image(src: Path, dst: Path, variant: str):
    """Read, transform, and write one image."""
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {src}")
    if variant == "clahe":
        image = apply_clahe(image)
    elif variant == "laplacian":
        image = apply_laplacian(image)
    else:
        raise ValueError(f"Unsupported preprocessing variant: {variant}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), image):
        raise ValueError(f"Unable to write image: {dst}")


def copy_label_for_image(src_image: Path, src_images_dir: Path, dst_labels_dir: Path):
    """Copy the matching YOLO label for one transformed image when it exists."""
    rel_label = src_image.relative_to(src_images_dir).with_suffix(".txt")
    src_label = image_to_label_path(src_image)
    dst_label = dst_labels_dir / rel_label
    if src_label.exists():
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_label, dst_label)


def write_dataset_yaml(source_cfg: dict, output_root: Path, output_yaml: Path):
    """Write the dataset YAML for a generated preprocessing variant."""
    generated = {
        "path": str(output_root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": source_cfg["nc"],
        "names": source_cfg["names"],
    }
    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(generated, f, sort_keys=False, allow_unicode=True)


def build_dataset(
    variant: str,
    source_yaml: Path = DEFAULT_SOURCE_YAML,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = False,
    limit: int | None = None,
) -> Path:
    """Build and return a preprocessing-only dataset YAML path."""
    if variant not in {"clahe", "laplacian"}:
        raise ValueError("variant must be 'clahe' or 'laplacian'.")

    source_yaml = source_yaml.resolve()
    cfg = load_dataset_yaml(source_yaml)
    source_root = Path(cfg.get("path", source_yaml.parent))
    source_root = source_root if source_root.is_absolute() else source_yaml.parent / source_root
    base_output_root = output_root.resolve()
    dataset_name = source_yaml.stem
    variant_output_root = (base_output_root / f"{dataset_name}_{variant}").resolve()
    output_yaml = variant_output_root / f"{dataset_name}_{variant}.yaml"

    if output_yaml.exists() and not overwrite:
        return output_yaml

    if base_output_root not in variant_output_root.parents:
        raise ValueError(f"Refusing to write outside {base_output_root}: {variant_output_root}")
    if variant_output_root.exists() and overwrite:
        shutil.rmtree(variant_output_root)
    variant_output_root.mkdir(parents=True, exist_ok=True)

    for split_key in ("train", "val", "test"):
        src_images_dir = resolve_split_dir(source_root, cfg[split_key])
        dst_images_dir = variant_output_root / split_key.replace("val", "valid") / "images"
        dst_labels_dir = variant_output_root / split_key.replace("val", "valid") / "labels"
        images = iter_images(src_images_dir)
        if limit is not None:
            images = images[:limit]
        for src_image in images:
            rel_image = src_image.relative_to(src_images_dir)
            transform_image(src_image, dst_images_dir / rel_image, variant)
            copy_label_for_image(src_image, src_images_dir, dst_labels_dir)

    for meta_name in ("README.dataset.txt", "README.roboflow.txt"):
        src_meta = source_root / meta_name
        if src_meta.exists():
            shutil.copy2(src_meta, variant_output_root / meta_name)
    write_dataset_yaml(cfg, variant_output_root, output_yaml)
    return output_yaml


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("clahe", "laplacian"), required=True)
    parser.add_argument("--source-yaml", type=Path, default=DEFAULT_SOURCE_YAML)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test image limit per split.")
    return parser.parse_args()


def main():
    """Build the requested preprocessing-only dataset."""
    args = parse_args()
    output_yaml = build_dataset(args.variant, args.source_yaml, args.output_root, args.overwrite, args.limit)
    print(f"Generated dataset YAML: {output_yaml}")


if __name__ == "__main__":
    main()
