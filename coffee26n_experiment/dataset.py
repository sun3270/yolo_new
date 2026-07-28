"""Dataset YAML parsing, label validation and deterministic data locks."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EMPTY_LABEL_SHA256 = hashlib.sha256(b"").hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names(value: Any) -> list[str]:
    if isinstance(value, list):
        names = [str(item) for item in value]
    elif isinstance(value, dict):
        try:
            ordered = sorted(((int(key), str(name)) for key, name in value.items()), key=lambda item: item[0])
        except Exception as exc:
            raise ValueError(f"Dataset names must be a list or integer-key mapping, got {value!r}.") from exc
        indices = [index for index, _ in ordered]
        if indices != list(range(len(indices))):
            raise ValueError(f"Dataset names mapping keys must be contiguous 0..N-1, got {indices!r}.")
        names = [name for _, name in ordered]
    else:
        raise ValueError("Dataset YAML must define names as a list or integer-key mapping.")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Dataset names must be non-empty and unique, got {names!r}.")
    return names


def _resolve_split(value: Any, root: Path, split: str) -> list[Path]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    paths: list[Path] = []
    for item in values:
        path = Path(str(item))
        path = path if path.is_absolute() else (root / path)
        if path.is_file() and path.suffix.lower() in {".txt", ".list"}:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    child = Path(line)
                    paths.append((child if child.is_absolute() else path.parent / child).resolve())
        elif path.is_dir():
            paths.extend(sorted(p.resolve() for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS))
        elif path.is_file():
            paths.append(path.resolve())
        else:
            raise FileNotFoundError(f"Dataset {split} path does not exist: {path}")
    # pathlib path ordering follows host filesystem semantics; sort POSIX strings for cross-platform data locks.
    paths = sorted(set(paths), key=lambda path: path.as_posix())
    if not paths:
        raise ValueError(f"Dataset {split} split contains no images.")
    return paths


def _label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _validate_labels(images: list[Path], nc: int, split: str) -> tuple[Counter[int], list[dict[str, Any]]]:
    counts: Counter[int] = Counter()
    records: list[dict[str, Any]] = []
    for image in images:
        if image.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension in {split}: {image}")
        label = _label_for_image(image)
        label_exists = label.is_file()
        rows = []
        label_lines = label.read_text(encoding="utf-8").splitlines() if label_exists else []
        for line_no, raw in enumerate(label_lines, 1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid label {label}:{line_no}; expected 5 fields, got {len(fields)}.")
            try:
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric label {label}:{line_no}: {line!r}") from exc
            if not 0 <= class_id < nc:
                raise ValueError(f"Class id {class_id} in {label}:{line_no} outside [0,{nc}).")
            if any(value != value or value < 0.0 or value > 1.0 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                raise ValueError(f"Normalized box in {label}:{line_no} is invalid: {coords!r}.")
            counts[class_id] += 1
            rows.append((class_id, *coords))
        records.append({
            "image": str(image),
            "image_sha256": _sha256(image),
            "label": str(label),
            "label_sha256": _sha256(label) if label_exists else EMPTY_LABEL_SHA256,
            "label_exists": label_exists,
            "boxes": rows,
        })
    return counts, records


def parse_dataset(data_yaml: str | Path) -> dict[str, Any]:
    """Parse and validate a YOLO detect dataset YAML before model construction."""
    path = Path(data_yaml).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid dataset YAML {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset YAML must be a mapping: {path}")
    names = _names(raw.get("names"))
    nc = int(raw.get("nc", len(names)))
    if nc != len(names):
        raise ValueError(f"Dataset nc={nc} does not match names length={len(names)} in {path}.")
    root_value = raw.get("path", path.parent)
    root = Path(str(root_value))
    root = root if root.is_absolute() else (path.parent / root)
    root = root.resolve()
    split_images: dict[str, list[Path]] = {}
    split_records: dict[str, list[dict[str, Any]]] = {}
    class_counts: Counter[int] = Counter()
    for split in ("train", "val", "test"):
        if raw.get(split) is None:
            split_images[split] = []
            split_records[split] = []
            continue
        images = _resolve_split(raw[split], root, split)
        counts, records = _validate_labels(images, nc, split)
        split_images[split] = images
        split_records[split] = records
        if split == "train":
            class_counts.update(counts)
    if not split_images["train"] or not split_images["val"]:
        raise ValueError("Dataset YAML must provide non-empty train and val splits.")
    image_hashes = {record["image"]: record["image_sha256"] for records in split_records.values() for record in records}
    overlap = set(split_images["train"]) & set(split_images["val"])
    if overlap:
        raise ValueError(f"Train/val image overlap detected: {sorted(map(str, overlap))[:3]}")
    hash_splits: dict[str, set[str]] = {}
    hash_paths: dict[str, list[str]] = {}
    for split, records in split_records.items():
        for record in records:
            digest = record["image_sha256"]
            hash_splits.setdefault(digest, set()).add(split)
            hash_paths.setdefault(digest, []).append(record["image"])
    cross_split_duplicates = {digest: hash_paths[digest] for digest, splits in hash_splits.items() if len(splits) > 1}
    if cross_split_duplicates:
        digest, paths = next(iter(cross_split_duplicates.items()))
        raise ValueError(f"Cross-split duplicate image content detected sha256={digest}: {paths}")
    relative_records = {
        split: [
            {
                "image": Path(record["image"]).resolve().relative_to(root).as_posix() if Path(record["image"]).is_relative_to(root) else Path(record["image"]).name,
                "image_sha256": record["image_sha256"],
                "label": Path(record["label"]).resolve().relative_to(root).as_posix() if Path(record["label"]).is_relative_to(root) else Path(record["label"]).name,
                "label_sha256": record["label_sha256"],
                "label_exists": record["label_exists"],
            }
            for record in records
        ]
        for split, records in split_records.items()
    }
    normalized_yaml = {key: value for key, value in raw.items() if key != "path"}
    normalized_yaml_sha256 = hashlib.sha256(
        json.dumps(normalized_yaml, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    class_image_counts: Counter[int] = Counter()
    for record in split_records["train"]:
        class_image_counts.update({int(row[0]) for row in record["boxes"]})
    empty_label_images = {
        split: [record["image"] for record in records if not record["boxes"]]
        for split, records in split_records.items()
    }
    all_boxes = [row for records in split_records.values() for record in records for row in record["boxes"]]
    widths, heights = [float(row[3]) for row in all_boxes], [float(row[4]) for row in all_boxes]
    areas = [width * height for width, height in zip(widths, heights)]

    def stats(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
        }

    box_statistics = {"width": stats(widths), "height": stats(heights), "area": stats(areas)}
    lock_payload = {
        "normalized_dataset_yaml_sha256": normalized_yaml_sha256,
        "nc": nc,
        "names": names,
        "splits": relative_records,
        "class_counts": dict(sorted(class_counts.items())),
        "class_image_counts": dict(sorted(class_image_counts.items())),
        "empty_label_images": empty_label_images,
        "box_statistics": box_statistics,
    }
    lock_id = hashlib.sha256(json.dumps(lock_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "yaml": path,
        "raw": raw,
        "root": root,
        "nc": nc,
        "names": names,
        "splits": split_images,
        "records": split_records,
        "class_counts": dict(sorted(class_counts.items())),
        "class_image_counts": dict(sorted(class_image_counts.items())),
        "empty_label_images": empty_label_images,
        "box_statistics": box_statistics,
        "image_hashes": image_hashes,
        "data_lock_id": lock_id,
        "dataset_yaml_sha256": _sha256(path),
        "normalized_dataset_yaml_sha256": normalized_yaml_sha256,
    }


def write_data_lock(dataset: dict[str, Any], output_dir: Path, pairs: list[tuple[int, int]] | None = None) -> None:
    """Write the audit files for the exact dataset snapshot used by a run."""
    lock_dir = output_dir / "data_lock"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "dataset_yaml_copy.yaml").write_text(yaml.safe_dump(dataset["raw"], sort_keys=False), encoding="utf-8")
    for split, images in dataset["splits"].items():
        rows = [str(path) for path in images]
        (lock_dir / f"{split}_files.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    label_rows = []
    image_rows = []
    for split, records in dataset["records"].items():
        for record in records:
            label_rows.append(f"{split},{record['label']},{record['label_sha256']},{int(record['label_exists'])}")
            image_rows.append(f"{split},{record['image']},{record['image_sha256']}")
    (lock_dir / "label_sha256.csv").write_text("split,label,sha256,exists\n" + "\n".join(sorted(label_rows)) + "\n", encoding="utf-8")
    (lock_dir / "image_sha256.csv").write_text("split,image,sha256\n" + "\n".join(sorted(image_rows)) + "\n", encoding="utf-8")
    (lock_dir / "class_counts.json").write_text(json.dumps(dataset["class_counts"], indent=2, sort_keys=True), encoding="utf-8")
    (lock_dir / "class_image_counts.json").write_text(json.dumps(dataset["class_image_counts"], indent=2, sort_keys=True), encoding="utf-8")
    (lock_dir / "box_statistics.json").write_text(json.dumps(dataset["box_statistics"], indent=2, sort_keys=True), encoding="utf-8")
    (lock_dir / "empty_label_images.json").write_text(json.dumps(dataset["empty_label_images"], indent=2, sort_keys=True), encoding="utf-8")
    pair_rows = [{"true_id": int(true_id), "rival_id": int(rival_id)} for true_id, rival_id in (pairs or [])]
    (lock_dir / "confusion_audit_pairs.json").write_text(json.dumps(pair_rows, indent=2), encoding="utf-8")
    (lock_dir / "dataset_hashes.json").write_text(json.dumps({
        "dataset_yaml_sha256": dataset["dataset_yaml_sha256"],
        "normalized_dataset_yaml_sha256": dataset["normalized_dataset_yaml_sha256"],
    }, indent=2), encoding="utf-8")
    (output_dir / "data_lock_id.txt").write_text(dataset["data_lock_id"] + "\n", encoding="utf-8")
