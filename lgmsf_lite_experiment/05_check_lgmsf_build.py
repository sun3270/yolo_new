"""Build-check YOLO26n-LGMSF-Lite before any training."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch


EXP_ROOT = Path(__file__).resolve().parent
LOCAL_PARENT = EXP_ROOT / "local_ultralytics"
CFG = EXP_ROOT / "configs" / "yolo26n_lgmsf_lite.yaml"
REPORT = EXP_ROOT / "reports" / "build_check_report.md"


def summarize_output(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [summarize_output(v) for v in value]
    if isinstance(value, dict):
        return {str(k): summarize_output(v) for k, v in value.items()}
    return str(type(value).__name__)


def main() -> None:
    sys.path.insert(0, str(LOCAL_PARENT))
    lines = ["# LGMSF-Lite Build Check Report", ""]
    try:
        from ultralytics.nn.tasks import DetectionModel  # noqa: PLC0415
        from ultralytics.nn.modules import LDSConv, LGMSFBridge  # noqa: F401, PLC0415

        model = DetectionModel(str(CFG), ch=3, nc=6, verbose=False)
        info_result = model.info(verbose=True)
        flops = None
        if isinstance(info_result, tuple) and len(info_result) >= 4:
            flops = info_result[3]
        module_names = [m.__class__.__name__ for m in model.modules()]
        has_lds = "LDSConv" in module_names
        has_bridge = "LGMSFBridge" in module_names
        params = sum(p.numel() for p in model.parameters())

        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 640, 640))
        output_summary = summarize_output(output)

        if not has_lds or not has_bridge:
            raise RuntimeError(f"Missing custom modules in built model: LDSConv={has_lds}, LGMSFBridge={has_bridge}")
        if params <= 0:
            raise RuntimeError("Parameter count is zero.")
        if flops is not None and flops <= 0:
            raise RuntimeError(f"FLOPs must be non-zero, got {flops}.")

        lines.extend(
            [
                "- Status: `passed`",
                f"- Config: `{CFG}`",
                f"- Parameters: `{params}`",
                f"- FLOPs: `{flops if flops is not None else 'reported by model.info()'}`",
                f"- Contains LDSConv: `{has_lds}`",
                f"- Contains LGMSFBridge: `{has_bridge}`",
                "- Dummy forward: `passed`",
                "",
                "## Output Summary",
                "",
                "```json",
                json.dumps(output_summary, indent=2),
                "```",
            ]
        )
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Build check passed: {REPORT}")
    except Exception:  # noqa: BLE001 - report full build failure for debugging
        err = traceback.format_exc()
        lines.extend(["- Status: `failed`", "", "## Error", "", "```text", err, "```"])
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Build check failed: {REPORT}")
        raise


if __name__ == "__main__":
    main()
