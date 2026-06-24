"""Scan and repair image files before YOLO training.

This script normalizes readable but malformed JPEG files outside the training
loop, then optionally clears YOLO dataset caches so the next run scans clean
files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "coffee_self" / "coffee_self.yaml"
IMAGE_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan and repair dataset images.")
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA, help="YOLO dataset YAML path.")
    parser.add_argument("--image-root", type=Path, default=None, help="Image directory to scan instead of YAML splits.")
    parser.add_argument("--dry-run", action="store_true", help="Only report files that would be repaired.")
    parser.add_argument("--rewrite-all", action="store_true", help="Rewrite every readable image, not only corrupt files.")
    parser.add_argument("--backup", action="store_true", help="Back up repaired originals before overwriting them.")
    parser.add_argument("--clear-cache", action="store_true", help="Delete YOLO *.cache files under the dataset root.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality used when rewriting images.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to check.")
    return parser.parse_args()


def load_dataset_root(data_yaml: Path) -> tuple[Path, dict]:
    data_yaml = data_yaml.resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root, data


def resolve_split_path(root: Path, value) -> list[Path]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        paths.append(path.resolve())
    return paths


def iter_images_from_file(list_file: Path) -> list[Path]:
    images = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        if not path.is_absolute():
            path = (list_file.parent / path).resolve()
        images.append(path)
    return images


def collect_images(data_yaml: Path, image_root: Path | None) -> tuple[Path, list[Path]]:
    if image_root is not None:
        root = image_root.resolve()
        return root, sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    root, data = load_dataset_root(data_yaml)
    images: list[Path] = []
    for split in ("train", "val", "test"):
        for path in resolve_split_path(root, data.get(split)):
            if path.is_file() and path.suffix.lower() == ".txt":
                images.extend(iter_images_from_file(path))
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
            elif path.exists():
                images.extend(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return root, sorted(set(p.resolve() for p in images))


def is_clean_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def backup_original(path: Path, data_root: Path) -> None:
    backup_root = data_root.parent / f"{data_root.name}_image_repair_backup"
    try:
        rel = path.resolve().relative_to(data_root.resolve())
    except ValueError:
        rel = path.name
    target = backup_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)


def rewrite_image(path: Path, quality: int) -> None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    tmp = path.with_name(f"{path.name}.repair_tmp")
    suffix = path.suffix.lower()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.load()
        if suffix in {".jpg", ".jpeg", ".mpo"}:
            im.convert("RGB").save(tmp, format="JPEG", quality=quality)
        elif suffix == ".png":
            im.save(tmp, format="PNG")
        elif suffix == ".webp":
            im.save(tmp, format="WEBP", quality=quality)
        else:
            im.save(tmp)
    tmp.replace(path)


def clear_cache_files(data_root: Path) -> int:
    count = 0
    for cache_file in data_root.rglob("*.cache"):
        cache_file.unlink()
        count += 1
    return count


def main() -> None:
    args = parse_args()
    data_root, images = collect_images(args.data_yaml, args.image_root)
    if args.limit > 0:
        images = images[: args.limit]

    checked = repaired = failed = dirty = 0
    for image in images:
        checked += 1
        clean = is_clean_image(image)
        needs_rewrite = args.rewrite_all or not clean
        if not clean:
            dirty += 1
        if not needs_rewrite:
            continue
        print(f"{'rewrite' if args.rewrite_all and clean else 'repair'}: {image}")
        if args.dry_run:
            continue
        try:
            if args.backup:
                backup_original(image, data_root)
            rewrite_image(image, args.quality)
            repaired += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"failed: {image} ({exc})")

    cache_deleted = clear_cache_files(data_root) if args.clear_cache and not args.dry_run else 0
    print("\n=== Image repair summary ===")
    print(f"data_root:     {data_root}")
    print(f"checked:       {checked}")
    print(f"dirty_found:   {dirty}")
    print(f"rewritten:     {repaired}")
    print(f"failed:        {failed}")
    print(f"cache_deleted: {cache_deleted}")


if __name__ == "__main__":
    main()
