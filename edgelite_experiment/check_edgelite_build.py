"""Build-check native YOLO26n, EdgeLite, and EdgeLite-SimAM."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch


EXP_ROOT = Path(__file__).resolve().parent
ROOT = EXP_ROOT.parent
LOCAL_ULTRALYTICS = EXP_ROOT / "local_ultralytics"
REPORT = EXP_ROOT / "reports" / "build_compare_report.md"


def summarize_output(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [summarize_output(v) for v in value]
    if isinstance(value, dict):
        return {str(k): summarize_output(v) for k, v in value.items()}
    return str(type(value).__name__)


def check_model(name: str, cfg: Path, DetectionModel):
    model = DetectionModel(str(cfg), ch=3, nc=6, verbose=False)
    info = model.info(verbose=True)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = info[3] if isinstance(info, tuple) and len(info) > 3 else None
    module_names = [m.__class__.__name__ for m in model.modules()]
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 640, 640))
    return {
        "name": name,
        "cfg": str(cfg),
        "layers": info[0] if isinstance(info, tuple) else None,
        "params": params,
        "trainable": trainable,
        "flops": flops,
        "has_ldsconv": "LDSConv" in module_names,
        "has_texture_stream_p3": "TextureStreamP3" in module_names,
        "has_semantic_stream_p5": "SemanticStreamP5" in module_names,
        "has_fast_norm_fuse2": "FastNormFuse2" in module_names,
        "has_simam_module": "SimAM" in module_names,
        "has_edge_bridge": "EdgeLGMSFBridge" in module_names,
        "output": summarize_output(output),
    }


def main() -> None:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
    lines = ["# YOLO26n-EdgeLite Build Compare Report", ""]
    try:
        from ultralytics.nn.tasks import DetectionModel  # noqa: PLC0415

        configs = [
            ("native", EXP_ROOT / "configs" / "yolo26n_original_copy.yaml"),
            ("edgelite", EXP_ROOT / "configs" / "yolo26n_edgelite.yaml"),
            ("edgelite_simam", EXP_ROOT / "configs" / "yolo26n_edgelite_simam.yaml"),
        ]
        results = [check_model(name, cfg, DetectionModel) for name, cfg in configs]
        lines.extend(
            [
                "| model | params | GFLOPs | layers | EdgeBridge | SimAM | dummy forward |",
                "|---|---:|---:|---:|---|---|---|",
            ]
        )
        for item in results:
            lines.append(
                f"| {item['name']} | {item['params']} | {item['flops']} | {item['layers']} | "
                f"{item['has_edge_bridge']} | {item['has_simam_module']} | passed |"
            )
        lines.extend(["", "## JSON", "", "```json", json.dumps(results, indent=2), "```"])
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Build compare passed: {REPORT}")
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        lines.extend(["- Status: failed", "", "```text", err, "```"])
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Build compare failed: {REPORT}")
        raise


if __name__ == "__main__":
    main()
