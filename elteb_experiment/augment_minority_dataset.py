"""Generate augmentation-only samples for minority classes in a YOLO dataset.

The original dataset is left untouched. Generated images and labels are written
to a separate YOLO-style folder with images/train, images/val, images/test and
matching labels folders.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "coffee_self" / "coffee_self.yaml"
DEFAULT_OUTPUT = ROOT / "coffee_self_minority_aug"
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)


@dataclass
class LabelRow:
    cls: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class SourceSample:
    image: Path
    label: Path
    split: str
    labels: list[LabelRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA, help="Source YOLO data YAML.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder for generated samples.")
    parser.add_argument("--target-count", type=int, default=200, help="Target total instances for minority classes.")
    parser.add_argument("--min-class-count", type=int, default=200, help="Only classes below this count are augmented.")
    parser.add_argument("--max-new-per-class", type=int, default=240, help="Safety cap for generated instances per class.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Output JPEG quality.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output before writing.")
    return parser.parse_args()


def load_dataset(data_yaml: Path) -> tuple[Path, dict]:
    data_yaml = data_yaml.resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root, cfg


def class_names(cfg: dict) -> list[str]:
    nc = int(cfg["nc"])
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [str(names.get(i, names.get(str(i), i))) for i in range(nc)]
    return [str(names[i]) if i < len(names) else str(i) for i in range(nc)]


def resolve_split(root: Path, cfg: dict, split: str) -> Path:
    path = Path(cfg[split])
    return path if path.is_absolute() else root / path


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return (image.parent.parent / "labels" / image.name).with_suffix(".txt")


def read_labels(path: Path, nc: int) -> tuple[list[LabelRow], list[str]]:
    labels: list[LabelRow] = []
    errors = []
    if not path.exists():
        return labels, [f"missing label: {path}"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) < 5:
            errors.append(f"{path}:{line_no} has fewer than 5 fields")
            continue
        try:
            cls = int(float(fields[0]))
            x, y, w, h = (float(v) for v in fields[1:5])
        except ValueError:
            errors.append(f"{path}:{line_no} is not numeric: {line}")
            continue
        if not (0 <= cls < nc):
            errors.append(f"{path}:{line_no} class {cls} outside 0..{nc - 1}")
            continue
        if w <= 0 or h <= 0:
            errors.append(f"{path}:{line_no} has non-positive box size")
            continue
        labels.append(LabelRow(cls, x, y, w, h))
    return labels, errors


def collect_sources(root: Path, cfg: dict) -> tuple[list[SourceSample], list[int], list[str]]:
    nc = int(cfg["nc"])
    samples: list[SourceSample] = []
    counts = [0] * nc
    errors: list[str] = []
    for split in ("train", "val", "test"):
        image_dir = resolve_split(root, cfg, split)
        for image in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            label = label_for_image(image)
            labels, label_errors = read_labels(label, nc)
            errors.extend(label_errors)
            if not labels:
                continue
            for row in labels:
                counts[row.cls] += 1
            samples.append(SourceSample(image=image, label=label, split=split, labels=labels))
    return samples, counts, errors


def yolo_to_corners(label: LabelRow, width: int, height: int) -> np.ndarray:
    x1 = (label.x - label.w / 2) * width
    y1 = (label.y - label.h / 2) * height
    x2 = (label.x + label.w / 2) * width
    y2 = (label.y + label.h / 2) * height
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def corners_to_yolo(cls: int, corners: np.ndarray, width: int, height: int) -> LabelRow | None:
    x1 = float(np.clip(corners[:, 0].min(), 0, width - 1))
    y1 = float(np.clip(corners[:, 1].min(), 0, height - 1))
    x2 = float(np.clip(corners[:, 0].max(), 0, width - 1))
    y2 = float(np.clip(corners[:, 1].max(), 0, height - 1))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 2 or bh < 2:
        return None
    return LabelRow(
        cls=cls,
        x=((x1 + x2) / 2) / width,
        y=((y1 + y2) / 2) / height,
        w=bw / width,
        h=bh / height,
    )


def transform_labels(labels: list[LabelRow], matrix: np.ndarray, width: int, height: int) -> list[LabelRow]:
    transformed = []
    m = matrix.astype(np.float32)
    for label in labels:
        corners = yolo_to_corners(label, width, height)
        hom = np.concatenate([corners, np.ones((4, 1), dtype=np.float32)], axis=1)
        new_corners = hom @ m.T
        new_label = corners_to_yolo(label.cls, new_corners, width, height)
        if new_label is not None:
            transformed.append(new_label)
    return transformed


def random_affine(image: np.ndarray, labels: list[LabelRow], rng: random.Random) -> tuple[np.ndarray, list[LabelRow]]:
    height, width = image.shape[:2]
    angle = rng.uniform(-12.0, 12.0)
    scale = rng.uniform(0.90, 1.10)
    tx = rng.uniform(-0.05, 0.05) * width
    ty = rng.uniform(-0.05, 0.05) * height
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += (tx, ty)
    border = tuple(float(x) for x in image.reshape(-1, 3).mean(axis=0))
    warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=border)
    new_labels = transform_labels(labels, matrix, width, height)
    return warped, new_labels


def maybe_flip(image: np.ndarray, labels: list[LabelRow], rng: random.Random) -> tuple[np.ndarray, list[LabelRow]]:
    height, width = image.shape[:2]
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    out = image
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
        matrix = np.array([[-1, 0, width - 1], [0, 1, 0]], dtype=np.float32) @ np.vstack([matrix, [0, 0, 1]])
        matrix = matrix[:2]
    if rng.random() < 0.2:
        out = cv2.flip(out, 0)
        matrix = np.array([[1, 0, 0], [0, -1, height - 1]], dtype=np.float32) @ np.vstack([matrix, [0, 0, 1]])
        matrix = matrix[:2]
    return out, transform_labels(labels, matrix, width, height)


def color_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
    hsv[..., 1] *= rng.uniform(0.75, 1.25)
    hsv[..., 2] *= rng.uniform(0.75, 1.25)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    alpha = rng.uniform(0.85, 1.20)
    beta = rng.uniform(-12, 12)
    return cv2.convertScaleAbs(out, alpha=alpha, beta=beta)


def add_texture_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.astype(np.float32)
    if rng.random() < 0.45:
        noise = np.random.default_rng(rng.randrange(1_000_000_000)).normal(0, rng.uniform(3, 10), out.shape)
        out += noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    roll = rng.random()
    if roll < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    elif roll < 0.55:
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
    return out


def augment_sample(sample: SourceSample, rng: random.Random) -> tuple[np.ndarray, list[LabelRow]]:
    image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {sample.image}")
    image, labels = random_affine(image, sample.labels, rng)
    if not labels:
        image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
        labels = sample.labels
    image, labels = maybe_flip(image, labels, rng)
    image = color_jitter(image, rng)
    image = add_texture_noise(image, rng)
    return image, labels


def choose_output_split(rng: random.Random) -> str:
    r = rng.random()
    if r < DEFAULT_SPLIT_RATIOS[0]:
        return "train"
    if r < DEFAULT_SPLIT_RATIOS[0] + DEFAULT_SPLIT_RATIOS[1]:
        return "val"
    return "test"


def write_label(path: Path, labels: list[LabelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{row.cls} {row.x:.6f} {row.y:.6f} {row.w:.6f} {row.h:.6f}" for row in labels]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists, use --overwrite to replace: {output}")
        import shutil

        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml(output: Path, source_cfg: dict, output_name: str) -> None:
    cfg = {
        "path": str(output.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": int(source_cfg["nc"]),
        "names": source_cfg["names"],
    }
    with (output / f"{output_name}.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def write_classes_txt(output: Path, source_cfg: dict) -> None:
    """Write LabelImg-compatible class names into the generated dataset root."""
    (output / "classes.txt").write_text("\n".join(class_names(source_cfg)) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_root, cfg = load_dataset(args.data_yaml)
    names = class_names(cfg)
    samples, counts, errors = collect_sources(source_root, cfg)

    minority_classes = [i for i, count in enumerate(counts) if 0 < count < args.min_class_count]
    if not minority_classes:
        raise RuntimeError("No minority classes found with the current thresholds.")

    prepare_output(args.output.resolve(), args.overwrite)
    write_yaml(args.output.resolve(), cfg, args.output.resolve().name)
    write_classes_txt(args.output.resolve(), cfg)

    class_to_samples = {
        cls: [sample for sample in samples if any(row.cls == cls for row in sample.labels)] for cls in minority_classes
    }
    generated_instances = [0] * int(cfg["nc"])
    generated_images = 0
    manifest_path = args.output.resolve() / "augmentation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_image", "new_label", "output_split", "source_split", "source_image", "target_class"])
        for cls in minority_classes:
            need = min(max(args.target_count - counts[cls], 0), args.max_new_per_class)
            if need <= 0 or not class_to_samples[cls]:
                continue
            made_for_class = 0
            attempts = 0
            while made_for_class < need and attempts < need * 20:
                attempts += 1
                sample = rng.choice(class_to_samples[cls])
                image, labels = augment_sample(sample, rng)
                produced = sum(1 for row in labels if row.cls == cls)
                if produced <= 0:
                    continue
                split = choose_output_split(rng)
                safe_name = names[cls].replace("/", "_").replace("\\", "_")
                stem = f"aug_{safe_name}_{generated_images:05d}_{sample.image.stem}"
                image_path = args.output.resolve() / "images" / split / f"{stem}.jpg"
                label_path = args.output.resolve() / "labels" / split / f"{stem}.txt"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
                write_label(label_path, labels)
                for row in labels:
                    generated_instances[row.cls] += 1
                made_for_class += produced
                generated_images += 1
                writer.writerow(
                    [
                        image_path.relative_to(args.output.resolve()).as_posix(),
                        label_path.relative_to(args.output.resolve()).as_posix(),
                        split,
                        sample.split,
                        sample.image.as_posix(),
                        names[cls],
                    ]
                )

    summary_path = args.output.resolve() / "augmentation_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"source_yaml: {args.data_yaml.resolve()}\n")
        f.write(f"output: {args.output.resolve()}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"target_count: {args.target_count}\n")
        f.write(f"min_class_count: {args.min_class_count}\n")
        f.write(f"generated_images: {generated_images}\n\n")
        f.write("source_counts:\n")
        for i, count in enumerate(counts):
            f.write(f"  {i}:{names[i]}={count}\n")
        f.write("\ngenerated_instances:\n")
        for i, count in enumerate(generated_instances):
            f.write(f"  {i}:{names[i]}={count}\n")
        if errors:
            f.write("\nlabel_warnings:\n")
            for error in errors[:100]:
                f.write(f"  {error}\n")
            if len(errors) > 100:
                f.write(f"  ... {len(errors) - 100} more\n")

    print(f"Generated images: {generated_images}")
    print(f"Output folder: {args.output.resolve()}")
    print(f"Summary: {summary_path}")
    print(f"Manifest: {manifest_path}")
    print("Minority classes:", ", ".join(f"{i}:{names[i]}={counts[i]}" for i in minority_classes))


if __name__ == "__main__":
    main()
