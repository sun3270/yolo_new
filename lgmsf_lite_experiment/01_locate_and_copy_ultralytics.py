"""Copy a clean Ultralytics 8.4.14 package into the local experiment area.

The current project package is intentionally not used as a fallback because it
may already include LGReplace changes. Provide a clean source through one of:

1. Environment variable LGMSF_CLEAN_ULTRALYTICS pointing to either a parent
   directory containing `ultralytics/` or the package directory itself.
2. A source archive/wheel under `_sources/`.
3. Network access for `python -m pip download ultralytics==8.4.14 --no-deps`.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path


REQUIRED_VERSION = "8.4.14"
EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_ROOT.parent
SOURCES = EXP_ROOT / "_sources"
DOWNLOADS = SOURCES / "downloads"
EXTRACTED = SOURCES / "extracted"
TARGET_PARENT = EXP_ROOT / "local_ultralytics"
TARGET = TARGET_PARENT / "ultralytics"
REPORT = EXP_ROOT / "reports" / "copy_ultralytics_report.md"


def current_import_path() -> str:
    spec = importlib.util.find_spec("ultralytics")
    if not spec or not spec.origin:
        return "not importable"
    return str(Path(spec.origin).resolve())


def package_dir(candidate: Path) -> Path | None:
    candidate = candidate.resolve()
    if (candidate / "__init__.py").exists() and candidate.name == "ultralytics":
        return candidate
    nested = candidate / "ultralytics"
    if (nested / "__init__.py").exists():
        return nested.resolve()
    return None


def read_version(pkg: Path) -> str | None:
    init_file = pkg / "__init__.py"
    try:
        for line in init_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        return None
    return None


def validate_clean_package(pkg: Path) -> tuple[bool, list[str]]:
    reasons = []
    version = read_version(pkg)
    if version != REQUIRED_VERSION:
        reasons.append(f"version is {version!r}, expected {REQUIRED_VERSION!r}")
    required_files = [
        pkg / "nn" / "modules" / "block.py",
        pkg / "nn" / "modules" / "__init__.py",
        pkg / "nn" / "tasks.py",
        pkg / "cfg" / "models" / "26" / "yolo26.yaml",
    ]
    for path in required_files:
        if not path.exists():
            reasons.append(f"missing {path.relative_to(pkg)}")

    marker_hits = []
    for rel in ("nn/modules/block.py", "nn/modules/__init__.py", "nn/tasks.py"):
        path = pkg / rel
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in ("DualLiteConv", "LGMSFBridge", "LDSConv"):
                if marker in text:
                    marker_hits.append(f"{rel}:{marker}")
    if marker_hits:
        reasons.append("contains experiment markers: " + ", ".join(marker_hits))

    try:
        if pkg.resolve().is_relative_to(PROJECT_ROOT / "ultralytics"):
            reasons.append("source resolves inside the current modified project ultralytics package")
    except AttributeError:
        project_pkg = (PROJECT_ROOT / "ultralytics").resolve()
        if str(pkg.resolve()).startswith(str(project_pkg)):
            reasons.append("source resolves inside the current modified project ultralytics package")
    return not reasons, reasons


def archive_candidates() -> list[Path]:
    patterns = ("ultralytics-8.4.14*.zip", "ultralytics-8.4.14*.whl", "ultralytics-8.4.14*.tar.gz", "ultralytics-8.4.14*.tgz")
    found: list[Path] = []
    search_roots = [SOURCES, DOWNLOADS, PROJECT_ROOT.parent, Path.home() / "Downloads"]
    for pattern in patterns:
        for root in search_roots:
            if root.exists():
                found.extend(root.glob(pattern))
    return sorted(set(found))


def extract_archive(archive: Path) -> Path:
    dest = EXTRACTED / archive.stem.replace(".tar", "")
    if dest.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes)
    if archive.suffix in {".zip", ".whl"}:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise RuntimeError(f"Unsupported archive type: {archive}")
    return dest


def try_pip_download() -> list[Path]:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", f"ultralytics=={REQUIRED_VERSION}", "--no-deps", "--dest", str(DOWNLOADS)]
    subprocess.run(cmd, check=True)
    return archive_candidates()


def find_clean_source() -> tuple[Path, str, list[str]]:
    attempts: list[str] = []

    env_path = os.environ.get("LGMSF_CLEAN_ULTRALYTICS")
    if env_path:
        pkg = package_dir(Path(env_path))
        if pkg:
            ok, reasons = validate_clean_package(pkg)
            if ok:
                return pkg, "LGMSF_CLEAN_ULTRALYTICS", attempts
            attempts.append(f"env source rejected: {pkg} ({'; '.join(reasons)})")
        else:
            attempts.append(f"env source does not contain an ultralytics package: {env_path}")

    for root in (SOURCES, EXTRACTED):
        for candidate in sorted(root.glob("**/ultralytics")) if root.exists() else []:
            pkg = package_dir(candidate)
            if not pkg:
                continue
            ok, reasons = validate_clean_package(pkg)
            if ok:
                return pkg, f"local source under {root}", attempts
            attempts.append(f"local source rejected: {pkg} ({'; '.join(reasons)})")

    for archive in archive_candidates():
        extracted = extract_archive(archive)
        pkg = package_dir(extracted)
        if not pkg:
            candidates = list(extracted.glob("**/ultralytics"))
            pkg = package_dir(candidates[0]) if candidates else None
        if pkg:
            ok, reasons = validate_clean_package(pkg)
            if ok:
                return pkg, f"archive {archive}", attempts
            attempts.append(f"archive source rejected: {pkg} ({'; '.join(reasons)})")

    try:
        for archive in try_pip_download():
            extracted = extract_archive(archive)
            candidates = [package_dir(extracted)] + [package_dir(p) for p in extracted.glob("**/ultralytics")]
            for pkg in [p for p in candidates if p is not None]:
                ok, reasons = validate_clean_package(pkg)
                if ok:
                    return pkg, f"pip download {archive.name}", attempts
                attempts.append(f"download source rejected: {pkg} ({'; '.join(reasons)})")
    except Exception as exc:  # noqa: BLE001 - report all acquisition failures clearly
        attempts.append(f"pip download failed: {exc}")

    raise RuntimeError("No clean ultralytics 8.4.14 source with YOLO26 was found.\n" + "\n".join(attempts))


def ignore_patterns(_dir: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".git", ".pytest_cache"}
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def copy_package(source: Path) -> None:
    TARGET_PARENT.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = TARGET_PARENT / f"ultralytics.backup_{stamp}"
        shutil.move(str(TARGET), str(backup))
    shutil.copytree(source, TARGET, ignore=ignore_patterns)


def write_report(source: Path, source_kind: str, attempts: list[str]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Copy Ultralytics Report",
        "",
        f"- Time: `{now}`",
        f"- Current Python import path: `{current_import_path()}`",
        f"- Required clean version: `{REQUIRED_VERSION}`",
        f"- Clean source kind: `{source_kind}`",
        f"- Clean source path: `{source}`",
        f"- Copied target: `{TARGET}`",
        f"- block.py exists: `{(TARGET / 'nn/modules/block.py').exists()}`",
        f"- modules __init__.py exists: `{(TARGET / 'nn/modules/__init__.py').exists()}`",
        f"- tasks.py exists: `{(TARGET / 'nn/tasks.py').exists()}`",
        "",
        "## Rejected Attempts",
        "",
    ]
    lines.extend(f"- {item}" for item in attempts) if attempts else lines.append("- None")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source, source_kind, attempts = find_clean_source()
    copy_package(source)
    write_report(source, source_kind, attempts)
    print(f"Copied clean ultralytics from: {source}")
    print(f"Target: {TARGET}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
