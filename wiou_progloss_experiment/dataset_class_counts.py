"""Dataset label statistics for ProgLoss configuration."""

from __future__ import annotations

from pathlib import Path

import yaml


IMAGE_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def load_dataset_yaml(data_yaml: Path) -> tuple[Path, dict]:
    """Load a YOLO dataset YAML and return its resolved dataset root."""
    data_yaml = data_yaml.resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root, cfg


def dataset_num_classes(cfg: dict) -> int:
    """Return class count from a YOLO dataset config."""
    if "nc" in cfg:
        return int(cfg["nc"])
    names = cfg.get("names", [])
    return len(names) if isinstance(names, (list, tuple, dict)) else 0


def dataset_class_names(cfg: dict) -> list[str]:
    """Return class names ordered by class index."""
    nc = dataset_num_classes(cfg)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [str(names.get(i, names.get(str(i), i))) for i in range(nc)]
    if isinstance(names, (list, tuple)):
        return [str(names[i]) if i < len(names) else str(i) for i in range(nc)]
    return [str(i) for i in range(nc)]


def resolve_split_paths(root: Path, split_value) -> list[Path]:
    """Resolve train/val/test entries from a YOLO dataset config."""
    if split_value is None:
        return []
    values = split_value if isinstance(split_value, list) else [split_value]
    paths = []
    for item in values:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        paths.append(path.resolve())
    return paths


def image_to_label_path(image_path: Path) -> Path:
    """Map a YOLO image path to its matching label path."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return (image_path.parent.parent / "labels" / image_path.name).with_suffix(".txt")


def iter_split_label_files(split_path: Path) -> list[Path]:
    """Return label files for one resolved YOLO split path."""
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        labels = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            image_path = Path(line)
            if not image_path.is_absolute():
                image_path = (split_path.parent / image_path).resolve()
            labels.append(image_to_label_path(image_path))
        return labels
    if split_path.is_file() and split_path.suffix.lower() in IMAGE_EXTS:
        return [image_to_label_path(split_path)]
    if split_path.exists():
        return [image_to_label_path(p) for p in split_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return []


def count_train_labels(data_yaml: Path, strict: bool = True) -> list[int]:
    """Count training-set target instances per class from YOLO label files."""
    root, cfg = load_dataset_yaml(data_yaml)
    nc = dataset_num_classes(cfg)
    if nc <= 0:
        raise ValueError(f"Unable to determine nc from dataset YAML: {data_yaml}")

    counts = [0] * nc
    invalid = 0
    invalid_examples = []
    missing = 0
    for split_path in resolve_split_paths(root, cfg.get("train")):
        for label_file in iter_split_label_files(split_path):
            if not label_file.exists():
                missing += 1
                continue
            for line_no, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.strip().split()
                if not fields:
                    continue
                try:
                    cls = int(float(fields[0]))
                except ValueError:
                    invalid += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append((label_file, line_no, fields[0]))
                    continue
                if 0 <= cls < nc:
                    counts[cls] += 1
                else:
                    invalid += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append((label_file, line_no, cls))

    if not any(counts):
        raise ValueError(f"No training labels found for ProgLoss class counts in: {data_yaml}")
    if invalid:
        message = f"Found {invalid} label rows outside valid class range 0..{nc - 1} for {data_yaml}"
        if invalid_examples:
            examples = "; ".join(f"{path}:{line_no} -> {value}" for path, line_no, value in invalid_examples)
            message = f"{message}. Examples: {examples}"
        if strict:
            raise ValueError(message)
        print(f"Warning: {message}")
    if missing:
        print(f"Warning: {missing} train images have no matching label file.")
    return counts


def format_class_counts(data_yaml: Path, counts: list[int]) -> str:
    """Format counted class instances with names for training logs."""
    _, cfg = load_dataset_yaml(data_yaml)
    names = dataset_class_names(cfg)
    return ", ".join(f"{i}:{names[i]}={count}" for i, count in enumerate(counts))
