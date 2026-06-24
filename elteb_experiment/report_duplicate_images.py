"""Report exact duplicate images and label conflicts in a YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "coffee_self" / "coffee_self.yaml"
DEFAULT_OUTPUT = ROOT / "coffee_self_duplicate_report"
IMAGE_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def label_for_image(root: Path, image: Path) -> Path:
    return root / "labels" / image.parent.name / f"{image.stem}.txt"


def file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def read_label_signature(root: Path, image: Path, names: list[str]) -> tuple[str, str, str]:
    label = label_for_image(root, image)
    if not label.exists():
        return "missing", "missing", ""
    rows = []
    classes = []
    for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) < 5:
            rows.append(f"BAD:{line.strip()}")
            classes.append("BAD")
            continue
        try:
            cls = int(float(fields[0]))
            cls_name = names[cls] if 0 <= cls < len(names) else f"OUT_OF_RANGE_{cls}"
            row = " ".join([str(cls), *[f"{float(v):.6f}" for v in fields[1:5]]])
        except ValueError:
            cls_name = f"NON_NUMERIC_{fields[0]}"
            row = f"BAD:{line.strip()}"
        rows.append(row)
        classes.append(cls_name)
    if not rows:
        return "empty", "empty", ""
    return "|".join(sorted(rows)), ",".join(sorted(set(classes))), label.as_posix()


def conflict_type(entries: list[dict]) -> str:
    label_sigs = {e["label_signature"] for e in entries}
    class_sigs = {e["label_classes"] for e in entries}
    splits = {e["split"] for e in entries}
    has_aug = any(e["is_aug"] for e in entries)
    types = []
    if len(splits) > 1:
        types.append("cross_split_duplicate")
    else:
        types.append("same_split_duplicate")
    if has_aug:
        types.append("contains_augmented")
    else:
        types.append("original_only")
    if len(class_sigs) > 1:
        types.append("class_conflict")
    elif len(label_sigs) > 1:
        types.append("box_conflict_same_class")
    else:
        types.append("same_label")
    return "+".join(types)


def main() -> None:
    args = parse_args()
    root, cfg = load_dataset(args.data_yaml)
    names = class_names(cfg)
    args.output.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        for image in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            groups[file_md5(image)].append(image)

    duplicate_groups = {h: paths for h, paths in groups.items() if len(paths) > 1}
    conflict_rows = []
    summary_counter = Counter()
    group_id = 0
    for md5, paths in sorted(duplicate_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        entries = []
        for image in paths:
            label_signature, label_classes, label_path = read_label_signature(root, image, names)
            entries.append(
                {
                    "split": image.parent.name,
                    "image": image.relative_to(root).as_posix(),
                    "label": Path(label_path).relative_to(root).as_posix() if label_path else "",
                    "label_classes": label_classes,
                    "label_signature": label_signature,
                    "is_aug": image.name.startswith("aug_"),
                }
            )
        ctype = conflict_type(entries)
        summary_counter[ctype] += 1
        if "class_conflict" not in ctype and "box_conflict_same_class" not in ctype:
            continue
        group_id += 1
        for entry in entries:
            conflict_rows.append(
                {
                    "group_id": group_id,
                    "md5": md5,
                    "duplicate_count": len(paths),
                    "duplicate_type": ctype,
                    **entry,
                }
            )

    csv_path = args.output / "duplicate_label_conflicts.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "md5",
                "duplicate_count",
                "duplicate_type",
                "split",
                "image",
                "label",
                "label_classes",
                "label_signature",
                "is_aug",
            ],
        )
        writer.writeheader()
        writer.writerows(conflict_rows)

    summary_path = args.output / "duplicate_summary.txt"
    conflict_group_ids = {row["group_id"] for row in conflict_rows}
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"data_yaml: {args.data_yaml.resolve()}\n")
        f.write(f"dataset_root: {root}\n")
        f.write(f"exact_duplicate_groups: {len(duplicate_groups)}\n")
        f.write(f"label_conflict_groups: {len(conflict_group_ids)}\n")
        f.write(f"label_conflict_rows: {len(conflict_rows)}\n\n")
        f.write("duplicate_type_counts:\n")
        for key, count in summary_counter.most_common():
            f.write(f"  {key}: {count}\n")
        f.write("\noutputs:\n")
        f.write(f"  {csv_path}\n")

    print(f"Exact duplicate groups: {len(duplicate_groups)}")
    print(f"Label conflict groups: {len(conflict_group_ids)}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
