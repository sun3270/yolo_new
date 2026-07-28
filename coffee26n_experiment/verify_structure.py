"""Offline structure, budget and forward verification for Coffee26n variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from coffee26n_experiment.config import (  # noqa: E402
    MODEL_FILES,
    VARIANTS,
    parse_imgsz,
    resolve_profile,
    resolve_profile_runtime,
    validate_profile_dataset,
)
from coffee26n_experiment.dataset import parse_dataset  # noqa: E402
from coffee26n_experiment.manifest import verify_source_manifest, write_source_manifest  # noqa: E402
from coffee26n_experiment.model_utils import build_model, detect_metadata, finite_forward, parameter_budget, transfer_weights  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Coffee26n model structure without network access.")
    parser.add_argument("--data", default=None, help="Optional explicit dataset YAML.")
    parser.add_argument("--variant", choices=VARIANTS, default="native")
    parser.add_argument("--profile", default="generic")
    parser.add_argument("--imgsz", default="640")
    parser.add_argument("--weights", default="none")
    parser.add_argument("--matrix", action="store_true", help="Verify all variants at nc=1,6,80 and imgsz=640,960.")
    parser.add_argument("--output", default=None)
    return parser


def _weights(value: str) -> Path | None:
    if str(value).lower() in {"none", "", "null"}:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Weights path does not exist; refusing download: {path}")
    return path


def _one(variant: str, nc: int, imgsz: tuple[int, int], weights: Path | None, *, data: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = PACKAGE_ROOT / "configs" / "models" / MODEL_FILES[variant]
    model = build_model(cfg, nc, loss_name="native")
    result: dict[str, Any] = {"variant": variant, "nc": nc, "imgsz": list(imgsz), "model": detect_metadata(model)}
    result["forward"] = finite_forward(model, imgsz)
    budget = parameter_budget(cfg, nc, model=model, variant=variant, imgsz=imgsz)
    if not budget["within_hard_limit"]:
        raise RuntimeError(f"{variant}/nc={nc} exceeds YOLO26s hard limit: {budget}")
    result["budget"] = budget
    if weights is not None:
        result["transfer"] = transfer_weights(model, weights, variant, target_nc=nc)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = PACKAGE_ROOT / "source_manifest.json"
    if not manifest.is_file():
        write_source_manifest(PACKAGE_ROOT)
    mismatches = verify_source_manifest(PACKAGE_ROOT)
    if mismatches:
        raise RuntimeError(f"Source manifest mismatch: {mismatches[:5]}")
    weights = _weights(args.weights)
    dataset = parse_dataset(args.data) if args.data else None
    if args.matrix:
        rows = []
        for nc in (1, 6, 80):
            for size in (640, 960):
                for variant in VARIANTS:
                    rows.append(_one(variant, nc, (size, size), weights))
        report = {"schema_version": 1, "mode": "matrix", "rows": rows}
    else:
        if dataset is None:
            raise ValueError("--data is required unless --matrix is used.")
        profile, profile_path = resolve_profile(args.profile)
        validate_profile_dataset(profile, dataset["yaml"], dataset)
        runtime_profile = resolve_profile_runtime(profile, dataset)
        row = _one(args.variant, dataset["nc"], parse_imgsz(args.imgsz), weights, data=dataset)
        row["data"] = str(dataset["yaml"])
        row["data_lock_id"] = dataset["data_lock_id"]
        row["profile"] = str(profile_path)
        row["profile_runtime"] = runtime_profile
        report = {"schema_version": 1, "mode": "single", "row": row}
    output = Path(args.output).resolve() if args.output else PACKAGE_ROOT / "structure_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run(_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
