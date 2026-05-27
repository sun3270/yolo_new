from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
LOCAL_ULTRALYTICS = ROOT / "edgelite_experiment" / "local_ultralytics"
sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO  # noqa: E402


DEPLOY_DIR = ROOT / "android_phone_deploy"
EXPORT_DIR = DEPLOY_DIR / "exports"
VALIDATION_DIR = DEPLOY_DIR / "consistency_validation"
SAMPLES_DIR = VALIDATION_DIR / "samples"
REPORTS_DIR = VALIDATION_DIR / "reports"
VISUALS_DIR = VALIDATION_DIR / "visuals"

CLASS_NAMES = ["algal_spot", "brown_eye_spot", "healthy", "miner", "phoma", "powdery_mildew"]

MODEL_SPECS = [
    ("01_yolo26n_base", "YOLO26n base"),
    ("02_yolo26n_edgelite", "YOLO26n EdgeLite"),
    ("03_yolo26n_edgelite_simam", "YOLO26n EdgeLite + SimAM"),
    ("04_yolo26n_edgelite_wiou_progloss", "YOLO26n EdgeLite + WIoU/ProgLoss"),
]

THRESHOLDS = {
    "onnx": {"min_iou": 0.995, "max_conf_diff": 0.005, "max_box_diff_px": 2.0, "max_count_diff": 0},
    "ncnn": {"min_iou": 0.95, "max_conf_diff": 0.08, "max_box_diff_px": 12.0, "max_count_diff": 0},
}


@dataclass
class Det:
    cls: int
    conf: float
    xyxy: list[float]


def detections_from_result(result) -> list[Det]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    dets = [Det(int(c), float(s), [float(v) for v in box]) for box, s, c in zip(xyxy, conf, cls)]
    return final_nms(sorted(dets, key=lambda d: d.conf, reverse=True), iou_thr=0.7)


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def final_nms(dets: list[Det], iou_thr: float) -> list[Det]:
    kept: list[Det] = []
    for det in sorted(dets, key=lambda d: d.conf, reverse=True):
        duplicate = any(det.cls == old.cls and iou(det.xyxy, old.xyxy) >= iou_thr for old in kept)
        if not duplicate:
            kept.append(det)
    return kept


def compare_dets(reference: list[Det], candidate: list[Det], fmt: str) -> dict:
    used: set[int] = set()
    matches = []
    for ref in reference:
        best_idx = None
        best_iou = -1.0
        for i, cand in enumerate(candidate):
            if i in used:
                continue
            score = iou(ref.xyxy, cand.xyxy)
            if cand.cls == ref.cls:
                score += 1.0
            if score > best_iou:
                best_iou = score
                best_idx = i
        if best_idx is None:
            continue
        used.add(best_idx)
        cand = candidate[best_idx]
        real_iou = iou(ref.xyxy, cand.xyxy)
        box_diff = max(abs(x - y) for x, y in zip(ref.xyxy, cand.xyxy))
        matches.append(
            {
                "ref_cls": ref.cls,
                "cand_cls": cand.cls,
                "class_match": ref.cls == cand.cls,
                "iou": real_iou,
                "conf_diff": abs(ref.conf - cand.conf),
                "box_diff_px": box_diff,
            }
        )

    count_diff = abs(len(reference) - len(candidate))
    class_mismatches = sum(1 for m in matches if not m["class_match"])
    min_iou = min((m["iou"] for m in matches), default=1.0 if not reference and not candidate else 0.0)
    mean_iou = float(np.mean([m["iou"] for m in matches])) if matches else min_iou
    max_conf_diff = max((m["conf_diff"] for m in matches), default=0.0)
    max_box_diff = max((m["box_diff_px"] for m in matches), default=0.0)
    threshold = THRESHOLDS[fmt]
    passed = (
        count_diff <= threshold["max_count_diff"]
        and class_mismatches == 0
        and min_iou >= threshold["min_iou"]
        and max_conf_diff <= threshold["max_conf_diff"]
        and max_box_diff <= threshold["max_box_diff_px"]
    )
    return {
        "passed": passed,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "count_diff": count_diff,
        "matched": len(matches),
        "class_mismatches": class_mismatches,
        "min_iou": min_iou,
        "mean_iou": mean_iou,
        "max_conf_diff": max_conf_diff,
        "max_box_diff_px": max_box_diff,
        "matches": matches,
    }


def draw_panel(image_path: Path, title: str, dets: list[Det]) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 24], fill=(20, 20, 20))
    draw.text((6, 5), title, fill=(255, 255, 255))
    for det in dets:
        x1, y1, x2, y2 = det.xyxy
        color = (30, 180, 60) if det.cls == 2 else (230, 80, 40)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{CLASS_NAMES[det.cls] if 0 <= det.cls < len(CLASS_NAMES) else det.cls} {det.conf:.3f}"
        tw = max(80, int(len(label) * 7.2))
        y_text = max(24, int(y1) - 20)
        draw.rectangle([x1, y_text, x1 + tw, y_text + 18], fill=color)
        draw.text((x1 + 3, y_text + 3), label, fill=(255, 255, 255))
    return image


def save_visual(image_path: Path, slug: str, outputs: dict[str, list[Det]]) -> None:
    panels = [
        draw_panel(image_path, "PyTorch", outputs["pt"]),
        draw_panel(image_path, "ONNX", outputs["onnx"]),
        draw_panel(image_path, "NCNN", outputs["ncnn"]),
    ]
    width = sum(p.width for p in panels)
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    out = VISUALS_DIR / f"{slug}__{image_path.stem}.jpg"
    canvas.save(out, quality=92)


def predict(model_path: Path, images: list[Path]):
    model = YOLO(str(model_path), task="detect")
    results = []
    for image in images:
        results.append(model.predict(str(image), imgsz=640, conf=0.25, iou=0.7, batch=1, verbose=False)[0])
    return results


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    sample_images = sorted(SAMPLES_DIR.glob("*.jpg"))
    if not sample_images:
        source_images = sorted((ROOT / "coffee3000" / "test" / "images").glob("*.jpg"))[:8]
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        for img in source_images:
            shutil.copy2(img, SAMPLES_DIR / img.name)
        sample_images = sorted(SAMPLES_DIR.glob("*.jpg"))

    all_records = []
    all_predictions = {}

    for slug, display_name in MODEL_SPECS:
        print(f"Checking {slug}", flush=True)
        base = EXPORT_DIR / slug
        paths = {
            "pt": base / f"{slug}.pt",
            "onnx": base / f"{slug}.onnx",
            "ncnn": base / f"{slug}_ncnn_model",
        }
        outputs = {fmt: [detections_from_result(r) for r in predict(path, sample_images)] for fmt, path in paths.items()}
        all_predictions[slug] = {
            "display_name": display_name,
            "images": {},
        }

        for image_path, pt_dets, onnx_dets, ncnn_dets in zip(
            sample_images, outputs["pt"], outputs["onnx"], outputs["ncnn"]
        ):
            comparisons = {
                "onnx": compare_dets(pt_dets, onnx_dets, "onnx"),
                "ncnn": compare_dets(pt_dets, ncnn_dets, "ncnn"),
            }
            all_predictions[slug]["images"][image_path.name] = {
                "pt": [det.__dict__ for det in pt_dets],
                "onnx": [det.__dict__ for det in onnx_dets],
                "ncnn": [det.__dict__ for det in ncnn_dets],
                "comparisons": comparisons,
            }
            save_visual(image_path, slug, {"pt": pt_dets, "onnx": onnx_dets, "ncnn": ncnn_dets})
            for fmt, cmp_result in comparisons.items():
                all_records.append(
                    {
                        "model": slug,
                        "image": image_path.name,
                        "format": fmt,
                        "passed": cmp_result["passed"],
                        "pt_count": cmp_result["reference_count"],
                        "candidate_count": cmp_result["candidate_count"],
                        "count_diff": cmp_result["count_diff"],
                        "matched": cmp_result["matched"],
                        "class_mismatches": cmp_result["class_mismatches"],
                        "min_iou": cmp_result["min_iou"],
                        "mean_iou": cmp_result["mean_iou"],
                        "max_conf_diff": cmp_result["max_conf_diff"],
                        "max_box_diff_px": cmp_result["max_box_diff_px"],
                    }
                )

    summary = {}
    for slug, _display_name in MODEL_SPECS:
        model_rows = [r for r in all_records if r["model"] == slug]
        summary[slug] = {}
        for fmt in ("onnx", "ncnn"):
            rows = [r for r in model_rows if r["format"] == fmt]
            summary[slug][fmt] = {
                "passed": all(r["passed"] for r in rows),
                "images": len(rows),
                "min_iou": min(r["min_iou"] for r in rows),
                "mean_iou": float(np.mean([r["mean_iou"] for r in rows])),
                "max_conf_diff": max(r["max_conf_diff"] for r in rows),
                "max_box_diff_px": max(r["max_box_diff_px"] for r in rows),
                "max_count_diff": max(r["count_diff"] for r in rows),
                "class_mismatches": sum(r["class_mismatches"] for r in rows),
            }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": sys.executable,
            "ultralytics_path": str(LOCAL_ULTRALYTICS),
        },
        "thresholds": THRESHOLDS,
        "sample_count": len(sample_images),
        "summary": summary,
        "predictions": all_predictions,
    }
    (REPORTS_DIR / "consistency_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (REPORTS_DIR / "consistency_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    lines = [
        "# Consistency Validation Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Python: `{sys.executable}`",
        f"- Sample images: {len(sample_images)}",
        "- Reference: PyTorch `.pt` output",
        "- Compared formats: ONNX and NCNN",
        "",
        "| Model | ONNX | ONNX min IoU | ONNX max conf diff | NCNN | NCNN min IoU | NCNN max conf diff | NCNN max box diff |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for slug, display_name in MODEL_SPECS:
        onnx = summary[slug]["onnx"]
        ncnn = summary[slug]["ncnn"]
        lines.append(
            f"| {display_name} | {'PASS' if onnx['passed'] else 'FAIL'} | {onnx['min_iou']:.6f} | "
            f"{onnx['max_conf_diff']:.6f} | {'PASS' if ncnn['passed'] else 'FAIL'} | "
            f"{ncnn['min_iou']:.6f} | {ncnn['max_conf_diff']:.6f} | {ncnn['max_box_diff_px']:.3f}px |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ONNX should be nearly identical to PyTorch.",
            "- NCNN may have small numeric differences because it uses a mobile-oriented graph and runtime.",
            "- A final same-class NMS pass is applied to all three outputs before comparison, matching the Android app behavior.",
            "- PASS means class IDs match, detection counts match, and box/confidence differences are within the thresholds recorded in `consistency_report.json`.",
        ]
    )
    (REPORTS_DIR / "CONSISTENCY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    passed = all(fmt_summary["passed"] for model in summary.values() for fmt_summary in model.values())
    print("PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
