"""Create the YOLO26n-LGMSF-Lite YAML from inspected YOLO26n metadata."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parent
STRUCTURE_JSON = EXP_ROOT / "reports" / "yolo26n_structure.json"
ORIGINAL_YAML = EXP_ROOT / "configs" / "yolo26n_original_copy.yaml"
OUTPUT_YAML = EXP_ROOT / "configs" / "yolo26n_lgmsf_lite.yaml"
REPORT = EXP_ROOT / "reports" / "lgmsf_yaml_report.md"


def remap_from(f, insert_at: int):
    def one(x):
        if isinstance(x, int) and x >= insert_at:
            return x + 1
        return x

    if isinstance(f, list):
        return [one(x) for x in f]
    return one(f)


def main() -> None:
    if not STRUCTURE_JSON.exists():
        raise FileNotFoundError(f"Run 02_inspect_yolo26n.py first: {STRUCTURE_JSON}")
    info = json.loads(STRUCTURE_JSON.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(ORIGINAL_YAML.read_text(encoding="utf-8", errors="ignore"))
    cfg = copy.deepcopy(cfg)
    cfg["nc"] = 6

    backbone = cfg["backbone"]
    head = cfg["head"]
    original_backbone_len = int(info["backbone_len"])
    replaced = []
    for index in info["downsample_indices"]:
        if index >= len(backbone):
            raise RuntimeError(f"Downsample index {index} is not inside backbone length {len(backbone)}")
        layer = backbone[index]
        if layer[2] != "Conv" or len(layer[3]) < 3 or layer[3][2] != 2:
            raise RuntimeError(f"Layer {index} is not a stride-2 Conv: {layer}")
        old = copy.deepcopy(layer)
        layer[2] = "LDSConv"
        replaced.append({"index": index, "old": old, "new": copy.deepcopy(layer)})

    bridge_index = original_backbone_len
    bridge = [
        [int(info["p3_index"]), int(info["p5_index"])],
        1,
        "LGMSFBridge",
        [int(info["p5_channels"]), True],
    ]
    backbone.append(bridge)

    remap_table = []
    for layer in head:
        old_from = copy.deepcopy(layer[0])
        layer[0] = remap_from(layer[0], original_backbone_len)
        if layer[0] != old_from:
            remap_table.append({"old": old_from, "new": copy.deepcopy(layer[0])})

    OUTPUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    detect_from = cfg["head"][-1][0]
    lines = [
        "# LGMSF-Lite YAML Report",
        "",
        f"- Source YAML: `{ORIGINAL_YAML}`",
        f"- Output YAML: `{OUTPUT_YAML}`",
        f"- Replaced downsample Conv indices: `{[x['index'] for x in replaced]}`",
        f"- LGMSFBridge index: `{bridge_index}`",
        f"- LGMSFBridge from: `{bridge[0]}`",
        f"- LGMSFBridge output channels: `{bridge[3][0]}`",
        f"- Final Detect from: `{detect_from}`",
        "",
        "## Head Remap Table",
        "",
    ]
    if remap_table:
        lines.extend(f"- `{item['old']}` -> `{item['new']}`" for item in remap_table)
    else:
        lines.append("- No positive head indices required remapping.")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"LGMSF YAML: {OUTPUT_YAML}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
