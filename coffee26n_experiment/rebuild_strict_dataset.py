"""Rebuild the strict five-class split from the grouped source dataset without duplicating image storage."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

EXPECTED_NAMES = ["ANT_AB", "ANT_CD", "BLS", "CR", "SM"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the audited strict five-class dataset from its source split.")
    parser.add_argument("--source", required=True, help="Grouped source dataset directory.")
    parser.add_argument("--manifest", required=True, help="Audited split_manifest.csv path.")
    parser.add_argument("--output", required=True, help="New, empty output dataset directory.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using same-filesystem hard links.")
    return parser


def _link_or_copy(source: Path, target: Path, copy_files: bool) -> None:
    if copy_files:
        shutil.copy2(source, target)
    else:
        os.link(source, target)


def rebuild_dataset(source: Path, manifest: Path, output: Path, *, copy_files: bool = False) -> dict[str, Any]:
    source, manifest, output = source.resolve(), manifest.resolve(), output.resolve()
    source_yaml = source / "coffee_self_sum_5cls_yolo_grouped_701515.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Grouped source YAML not found: {source_yaml}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Strict split manifest not found: {manifest}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be absent or empty: {output}")

    source_config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    names_value = source_config.get("names")
    names = [str(names_value[index]) for index in range(len(names_value))] if isinstance(names_value, dict) else list(names_value)
    if int(source_config.get("nc", len(names))) != 5 or names != EXPECTED_NAMES:
        raise ValueError(f"Unexpected grouped source classes: nc={source_config.get('nc')}, names={names}")

    rows = list(csv.DictReader(manifest.read_text(encoding="utf-8-sig").splitlines()))
    included = [row for row in rows if str(row.get("included", "")).lower() == "true"]
    if not included:
        raise ValueError(f"Manifest contains no included samples: {manifest}")

    targets: set[Path] = set()
    split_counts: Counter[str] = Counter()
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in included:
        filename = row["image_filename"]
        source_split = row["source_final_split"]
        final_split = row["final_split"]
        if source_split not in {"train", "val", "test"} or final_split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split row: {row}")
        source_image = source / "images" / source_split / filename
        source_label = source / "labels" / source_split / Path(filename).with_suffix(".txt")
        target_image = output / "images" / final_split / filename
        target_label = output / "labels" / final_split / Path(filename).with_suffix(".txt")
        for source_path, target_path in ((source_image, target_image), (source_label, target_label)):
            if not source_path.is_file():
                raise FileNotFoundError(f"Manifest source file is missing: {source_path}")
            if target_path in targets or target_path.exists():
                raise ValueError(f"Duplicate manifest target: {target_path}")
            _link_or_copy(source_path, target_path, copy_files)
            targets.add(target_path)
        split_counts[final_split] += 1

    dataset_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 5,
        "names": {index: name for index, name in enumerate(EXPECTED_NAMES)},
    }
    output_yaml = output / "coffee_self_sum_5cls_yolo_grouped_701515_clean.yaml"
    output_yaml.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    shutil.copy2(manifest, output / "split_manifest.csv")
    for filename in ("build_summary.json", "classes.txt"):
        recipe_file = manifest.parent / filename
        if recipe_file.is_file():
            shutil.copy2(recipe_file, output / filename)

    expected = {"train": 2096, "val": 423, "test": 416}
    actual = {split: int(split_counts[split]) for split in expected}
    if actual != expected:
        raise RuntimeError(f"Strict split count mismatch: expected={expected}, actual={actual}")
    return {
        "source": str(source),
        "manifest": str(manifest),
        "output": str(output),
        "mode": "copy" if copy_files else "hardlink",
        "images": sum(actual.values()),
        "splits": actual,
        "yaml": str(output_yaml),
    }


if __name__ == "__main__":
    args = _parser().parse_args()
    report = rebuild_dataset(Path(args.source), Path(args.manifest), Path(args.output), copy_files=args.copy)
    print(json.dumps(report, indent=2, sort_keys=True))
