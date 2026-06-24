"""Build a YOLO dataset with all exact duplicate image hashes removed.

For every duplicate MD5 group, all images in that group are removed. No
representative is kept. The source dataset is never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "coffee_self" / "coffee_self.yaml"
DEFAULT_OUTPUT = ROOT / "coffee_self_de"
IMAGE_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output folder.")
    return parser.parse_args()


def load_dataset(data_yaml: Path) -> tuple[Path, dict]:
    data_yaml = data_yaml.resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root, cfg


def normalize_names(cfg: dict) -> dict[int, str]:
    nc = int(cfg["nc"])
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return {i: str(names.get(i, names.get(str(i), i))) for i in range(nc)}
    return {i: str(names[i]) if i < len(names) else str(i) for i in range(nc)}


def file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def iter_images(root: Path) -> list[Path]:
    images = []
    for split in SPLITS:
        image_dir = root / "images" / split
        if not image_dir.exists():
            continue
        images.extend(sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS))
    return images


def split_for_image(root: Path, image: Path) -> str:
    return image.relative_to(root / "images").parts[0]


def label_for_image(root: Path, image: Path) -> Path:
    split = split_for_image(root, image)
    rel = image.relative_to(root / "images" / split).with_suffix(".txt")
    return root / "labels" / split / rel


def label_signature(label: Path) -> tuple[str, str]:
    if not label.exists():
        return "missing", ""
    rows = []
    classes = []
    for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        rows.append(" ".join(fields))
        classes.append(fields[0])
    if not rows:
        return "empty", ""
    return "|".join(sorted(rows)), ",".join(sorted(set(classes)))


def parse_label_classes(label: Path, nc: int) -> Counter:
    counts: Counter = Counter()
    if not label.exists():
        return counts
    for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        try:
            cls = int(float(fields[0]))
        except ValueError:
            counts["bad"] += 1
            continue
        if 0 <= cls < nc:
            counts[cls] += 1
        else:
            counts["out_of_range"] += 1
    return counts


def ensure_output(output: Path, source_root: Path, overwrite: bool) -> None:
    output = output.resolve()
    source_root = source_root.resolve()
    if output == source_root:
        raise ValueError("Output folder must be different from the source dataset folder.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
        workspace = ROOT.resolve()
        try:
            output.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"Refusing to overwrite outside workspace: {output}") from exc
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output: Path, cfg: dict, names: dict[int, str]) -> None:
    out_cfg = {
        "path": output.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": int(cfg["nc"]),
        "names": names,
    }
    for yaml_name in ("coffee_self_de.yaml", "coffee_self.yaml"):
        with (output / yaml_name).open("w", encoding="utf-8") as f:
            yaml.safe_dump(out_cfg, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    args = parse_args()
    source_root, cfg = load_dataset(args.data_yaml)
    output = args.output.resolve()
    names = normalize_names(cfg)
    nc = int(cfg["nc"])

    images = iter_images(source_root)
    hash_to_images: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        hash_to_images[file_md5(image)].append(image)

    duplicate_hashes = {h for h, paths in hash_to_images.items() if len(paths) > 1}
    keep_images = [image for paths in hash_to_images.values() if len(paths) == 1 for image in paths]

    ensure_output(output, source_root, args.overwrite)

    copied_by_split: Counter = Counter()
    labels_by_split: Counter = Counter()
    skipped_missing_labels = []
    class_counts: Counter = Counter()

    for image in sorted(keep_images):
        split = split_for_image(source_root, image)
        rel = image.relative_to(source_root / "images" / split)
        label = label_for_image(source_root, image)
        if not label.exists():
            skipped_missing_labels.append(image)
            continue

        out_image = output / "images" / split / rel
        out_label = output / "labels" / split / rel.with_suffix(".txt")
        out_image.parent.mkdir(parents=True, exist_ok=True)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, out_image)
        shutil.copy2(label, out_label)
        copied_by_split[split] += 1
        labels_by_split[split] += 1
        class_counts.update(parse_label_classes(label, nc))

    root_classes = source_root / "classes.txt"
    if root_classes.exists():
        shutil.copy2(root_classes, output / "classes.txt")

    write_dataset_yaml(output, cfg, names)

    removed_csv = output / "removed_duplicate_images.csv"
    with removed_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "md5",
                "duplicate_count",
                "split",
                "image",
                "label",
                "label_classes",
                "label_signature",
            ],
        )
        writer.writeheader()
        group_id = 0
        for md5, paths in sorted(hash_to_images.items(), key=lambda item: (-len(item[1]), item[0])):
            if md5 not in duplicate_hashes:
                continue
            group_id += 1
            for image in sorted(paths):
                label = label_for_image(source_root, image)
                sig, classes = label_signature(label)
                writer.writerow(
                    {
                        "group_id": group_id,
                        "md5": md5,
                        "duplicate_count": len(paths),
                        "split": split_for_image(source_root, image),
                        "image": image.relative_to(source_root).as_posix(),
                        "label": label.relative_to(source_root).as_posix() if label.exists() else "",
                        "label_classes": classes,
                        "label_signature": sig,
                    }
                )

    copied_csv = output / "copied_unique_images.csv"
    with copied_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "label"])
        writer.writeheader()
        for split in SPLITS:
            for image in sorted((output / "images" / split).rglob("*")):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                    rel = image.relative_to(output / "images" / split)
                    writer.writerow(
                        {
                            "split": split,
                            "image": image.relative_to(output).as_posix(),
                            "label": (Path("labels") / split / rel.with_suffix(".txt")).as_posix(),
                        }
                    )

    summary = output / "dedupe_summary.txt"
    removed_count = sum(len(paths) for h, paths in hash_to_images.items() if h in duplicate_hashes)
    with summary.open("w", encoding="utf-8") as f:
        f.write(f"source_dataset: {source_root}\n")
        f.write(f"output_dataset: {output}\n")
        f.write(f"source_images: {len(images)}\n")
        f.write(f"exact_duplicate_groups_removed_completely: {len(duplicate_hashes)}\n")
        f.write(f"duplicate_images_removed: {removed_count}\n")
        f.write(f"unique_images_copied: {sum(copied_by_split.values())}\n")
        f.write(f"skipped_missing_labels: {len(skipped_missing_labels)}\n\n")
        f.write("copied_by_split:\n")
        for split in SPLITS:
            f.write(f"  {split}: images={copied_by_split[split]}, labels={labels_by_split[split]}\n")
        f.write("\nclass_object_counts_after_dedupe:\n")
        for cls in range(nc):
            f.write(f"  {cls} {names[cls]}: {class_counts[cls]}\n")
        if class_counts.get("bad", 0) or class_counts.get("out_of_range", 0):
            f.write(f"  bad: {class_counts.get('bad', 0)}\n")
            f.write(f"  out_of_range: {class_counts.get('out_of_range', 0)}\n")
        if skipped_missing_labels:
            f.write("\nskipped_missing_label_images:\n")
            for image in skipped_missing_labels:
                f.write(f"  {image.relative_to(source_root).as_posix()}\n")
        f.write("\noutputs:\n")
        f.write(f"  {removed_csv}\n")
        f.write(f"  {copied_csv}\n")

    print(f"Source images: {len(images)}")
    print(f"Duplicate groups removed completely: {len(duplicate_hashes)}")
    print(f"Duplicate images removed: {removed_count}")
    print(f"Unique images copied: {sum(copied_by_split.values())}")
    for split in SPLITS:
        print(f"{split}: images={copied_by_split[split]}, labels={labels_by_split[split]}")
    print(f"Output: {output}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
