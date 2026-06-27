"""Summarize Coffee V4 training runs and optional recall-sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / "runs" / "train"


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def as_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "") or 0))
    except ValueError:
        return 0


def read_results_csv(path: Path) -> list[dict[str, str]]:
    """Read an Ultralytics results.csv file."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def best_row(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    """Return the row with the highest metric."""
    return max(rows, key=lambda row: as_float(row, metric))


def find_recall_summaries(run_dir: Path) -> list[Path]:
    """Find recall sweep summaries below a run directory."""
    return sorted(run_dir.rglob("recall_sweep_summary.json"))


def summarize_recall_sweep(path: Path) -> dict[str, Any]:
    """Summarize the best recall config in a recall sweep JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    configs = data.get("configs", {})
    if not configs:
        return {"path": str(path), "available": False}
    best_name, best_cfg = max(configs.items(), key=lambda item: float(item[1].get("recall", 0.0) or 0.0))
    classes = best_cfg.get("classes", {})
    return {
        "path": str(path),
        "available": True,
        "best_config": best_name,
        "precision": best_cfg.get("precision"),
        "recall": best_cfg.get("recall"),
        "missed": best_cfg.get("missed"),
        "false_positives": best_cfg.get("false_positives"),
        "classes": {
            name: {
                "precision": cls.get("precision"),
                "recall": cls.get("recall"),
                "missed": cls.get("missed"),
                "low_conf_candidates": cls.get("low_conf_candidates"),
            }
            for name, cls in classes.items()
        },
    }


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    """Summarize one Ultralytics train directory."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None
    rows = read_results_csv(results_csv)
    if not rows:
        return None
    best_map = best_row(rows, "metrics/mAP50-95(B)")
    best_recall = best_row(rows, "metrics/recall(B)")
    last = rows[-1]
    recall_summaries = [summarize_recall_sweep(path) for path in find_recall_summaries(run_dir)]
    return {
        "run": run_dir.name,
        "path": str(run_dir.resolve()),
        "last_epoch": as_int(last, "epoch"),
        "best_map_epoch": as_int(best_map, "epoch"),
        "best_map_precision": as_float(best_map, "metrics/precision(B)"),
        "best_map_recall": as_float(best_map, "metrics/recall(B)"),
        "best_map50": as_float(best_map, "metrics/mAP50(B)"),
        "best_map5095": as_float(best_map, "metrics/mAP50-95(B)"),
        "max_recall_epoch": as_int(best_recall, "epoch"),
        "max_recall_precision": as_float(best_recall, "metrics/precision(B)"),
        "max_recall": as_float(best_recall, "metrics/recall(B)"),
        "max_recall_map50": as_float(best_recall, "metrics/mAP50(B)"),
        "max_recall_map5095": as_float(best_recall, "metrics/mAP50-95(B)"),
        "has_pr_curve": (run_dir / "BoxPR_curve.png").exists(),
        "has_confusion_matrix": (run_dir / "confusion_matrix_normalized.png").exists(),
        "best_weights": str((run_dir / "weights" / "best.pt").resolve()) if (run_dir / "weights" / "best.pt").exists() else "",
        "recall_sweeps": recall_summaries,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a compact CSV summary."""
    fields = [
        "run",
        "last_epoch",
        "best_map_epoch",
        "best_map_precision",
        "best_map_recall",
        "best_map50",
        "best_map5095",
        "max_recall_epoch",
        "max_recall_precision",
        "max_recall",
        "max_recall_map50",
        "max_recall_map5095",
        "has_pr_curve",
        "has_confusion_matrix",
        "best_weights",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--runs", default=None, help="Comma-separated run names. Defaults to all runs with results.csv.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "training_summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_names = {name.strip() for name in args.runs.split(",") if name.strip()} if args.runs else None
    run_dirs = sorted((path for path in args.runs_dir.iterdir() if path.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for run_dir in run_dirs:
        if run_names and run_dir.name not in run_names:
            continue
        summary = summarize_run(run_dir)
        if summary:
            rows.append(summary)
    json_path = args.output / "training_summary.json"
    csv_path = args.output / "training_summary.csv"
    json_path.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows, csv_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    for row in rows[:10]:
        print(
            f"{row['run']}: best mAP50-95={row['best_map5095']:.4f} "
            f"(epoch {row['best_map_epoch']}), max recall={row['max_recall']:.4f} "
            f"(epoch {row['max_recall_epoch']})"
        )


if __name__ == "__main__":
    main()
