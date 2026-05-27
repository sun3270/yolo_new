from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "android_phone_deploy"
EXPORT_DIR = DEPLOY_DIR / "exports"
LOCAL_ULTRALYTICS = ROOT / "edgelite_experiment" / "local_ultralytics"

sys.path.insert(0, str(LOCAL_ULTRALYTICS))

import torch  # noqa: E402
from ultralytics.engine.exporter import Exporter  # noqa: E402


MODELS = [
    {
        "slug": "01_yolo26n_base",
        "name": "YOLO26n base",
        "source": ROOT / "run_down" / "yolo26n_coffee3000_laptop-4" / "weights" / "best.pt",
    },
    {
        "slug": "02_yolo26n_edgelite",
        "name": "YOLO26n EdgeLite",
        "source": ROOT / "run_down" / "yolo26n_edgelite_coffee3000" / "weights" / "best.pt",
    },
    {
        "slug": "03_yolo26n_edgelite_simam",
        "name": "YOLO26n EdgeLite + SimAM",
        "source": ROOT / "run_down" / "yolo26n_edgelite_simam_coffee3000" / "weights" / "best.pt",
    },
    {
        "slug": "04_yolo26n_edgelite_wiou_progloss",
        "name": "YOLO26n EdgeLite + WIoU/ProgLoss",
        "source": ROOT / "run_down" / "yolo26n_edgelite_wiou_progloss_coffee30003" / "weights" / "best.pt",
    },
]


def file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() and path.is_file() else None


def dir_size(path: Path) -> int | None:
    if not path.exists() or not path.is_dir():
        return None
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def export_one(model_info: dict, formats: tuple[str, ...]) -> dict:
    slug = model_info["slug"]
    source = Path(model_info["source"])
    out_dir = EXPORT_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    local_pt = out_dir / f"{slug}.pt"
    shutil.copy2(source, local_pt)

    result = {
        "slug": slug,
        "name": model_info["name"],
        "source": str(source),
        "pt": str(local_pt),
        "pt_size": file_size(local_pt),
        "formats": {},
    }

    for fmt in formats:
        try:
            ckpt = torch.load(local_pt, map_location="cpu", weights_only=False)
            model = ckpt.get("ema") or ckpt.get("model")
            if model is None:
                raise RuntimeError(f"No model found in checkpoint: {local_pt}")
            model = model.float()
            model.pt_path = str(local_pt)
            model.task = getattr(model, "task", "detect")

            exporter = Exporter(
                overrides={
                    "format": fmt,
                    "imgsz": 640,
                    "batch": 1,
                    "device": "cpu",
                    "half": False,
                    "int8": False,
                    "dynamic": False,
                    "simplify": False,
                    "nms": False,
                    "verbose": False,
                }
            )
            exported = exporter(
                model=model,
            )
            exported_path = Path(exported)
            result["formats"][fmt] = {
                "status": "ok",
                "path": str(exported_path),
                "size": dir_size(exported_path) if exported_path.is_dir() else file_size(exported_path),
            }
        except Exception as exc:
            result["formats"][fmt] = {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

    return result


def write_reports(results: list[dict]) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_size": "640x640",
        "class_names": ["algal_spot", "brown_eye_spot", "healthy", "miner", "phoma", "powdery_mildew"],
        "results": results,
    }
    (EXPORT_DIR / "export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Export Results",
        "",
        f"- Generated at: {manifest['generated_at']}",
        "- Input size: 640x640",
        "- Classes: algal_spot, brown_eye_spot, healthy, miner, phoma, powdery_mildew",
        "",
        "| Model | PT | ONNX | NCNN |",
        "|---|---:|---:|---:|",
    ]

    def status_text(entry: dict | None) -> str:
        if not entry:
            return "missing"
        if entry["status"] == "ok":
            size = entry.get("size")
            if size is None:
                return "ok"
            return f"ok ({size / 1024 / 1024:.2f} MB)"
        return "failed"

    for item in results:
        lines.append(
            f"| {item['slug']} | {item['pt_size'] / 1024 / 1024:.2f} MB | "
            f"{status_text(item['formats'].get('onnx'))} | {status_text(item['formats'].get('ncnn'))} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- WIoU/ProgLoss only affects training. Phone-side inference only needs the exported model and YOLO postprocess.",
            "- EdgeLite models require the custom Ultralytics modules during export, but the Android app should use the exported NCNN/TFLite/ONNX files.",
            "- For Android local testing, prefer NCNN when available.",
        ]
    )
    (EXPORT_DIR / "EXPORT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all Android deployment candidate models.")
    parser.add_argument(
        "--formats",
        default="onnx,ncnn",
        help="Comma-separated export formats, for example: onnx,ncnn",
    )
    parser.add_argument("--only", default="", help="Export only one model slug.")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    formats = tuple(x.strip().lower() for x in cli.formats.split(",") if x.strip())
    selected = [m for m in MODELS if not cli.only or m["slug"] == cli.only]
    if not selected:
        raise SystemExit(f"No matching model slug: {cli.only}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for model_info in selected:
        print(f"=== exporting {model_info['slug']} ===", flush=True)
        results.append(export_one(model_info, formats))
    if not cli.only:
        write_reports(results)

    failed = [
        f"{item['slug']}:{fmt}"
        for item in results
        for fmt, info in item["formats"].items()
        if info["status"] != "ok"
    ]
    if failed:
        print("Failed exports:", ", ".join(failed), flush=True)
        return 1
    print("All exports completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
