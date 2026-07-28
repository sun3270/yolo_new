"""SHA256 manifest for the package-local runtime and model YAML sources."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_source_path(package_root: Path, source_path: Path) -> str:
    """Return a repository-relative source reference without publishing local absolute paths."""
    repository_root = package_root.parent.resolve()
    try:
        return source_path.resolve().relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Manifest source escaped repository root: {source_path}") from exc


def manifest_entries(package_root: Path) -> list[dict[str, Any]]:
    """Enumerate every package source/config entry in stable logical order.

    Binary fixtures, checkpoints and generated run outputs are intentionally excluded; their data lock and transfer
    reports carry separate hashes.
    """
    excluded_parts = {"runs", "__pycache__", ".pytest_cache"}
    suffixes = {".py", ".yaml", ".yml", ".slurm", ".sh", ".md", ".csv", ".txt", ".json"}
    files = [
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and path.name != "source_manifest.json"
        and path.suffix.lower() in suffixes
        and not (set(path.relative_to(package_root).parts) & excluded_parts)
    ]
    entries = []
    for path in sorted(files):
        relative = path.relative_to(package_root).as_posix()
        if relative.startswith("local_ultralytics/"):
            logical = relative.removeprefix("local_ultralytics/")
            native_source = package_root.parent / logical
            source_path = native_source if native_source.is_file() else path
        else:
            logical = relative
            source_path = path
        entries.append({
            "logical_path": logical,
            "source_path": _portable_source_path(package_root, source_path),
            "package_path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    return entries


def write_source_manifest(package_root: Path) -> Path:
    """Generate the package manifest and return its path."""
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_root": package_root.name,
        "native_source_root": "ultralytics",
        "native_revision": "ultralytics-8.4.43",
        "files": manifest_entries(package_root),
    }
    output = package_root / "source_manifest.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def verify_source_manifest(package_root: Path, manifest_path: Path | None = None) -> list[str]:
    """Return hash mismatches; callers should fail if the list is non-empty."""
    path = manifest_path or package_root / "source_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mismatches = []
    recorded = {entry["package_path"] for entry in payload.get("files", [])}
    current = {entry["package_path"] for entry in manifest_entries(package_root)}
    mismatches.extend(f"unlisted:{package_root / item}" for item in sorted(current - recorded))
    mismatches.extend(f"stale-entry:{package_root / item}" for item in sorted(recorded - current))
    for entry in payload.get("files", []):
        file_path = package_root / entry["package_path"]
        if not file_path.is_file():
            mismatches.append(f"missing:{file_path}")
            continue
        actual = sha256_file(file_path)
        if actual != entry["sha256"] or file_path.stat().st_size != entry["size_bytes"]:
            mismatches.append(f"changed:{file_path}")
    return mismatches
