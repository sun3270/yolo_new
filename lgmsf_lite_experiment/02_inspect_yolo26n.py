"""Inspect the true YOLO26n structure from the local Ultralytics copy."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml


EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_ROOT.parent
LOCAL_PARENT = EXP_ROOT / "local_ultralytics"
LOCAL_PKG = LOCAL_PARENT / "ultralytics"
CONFIG_COPY = EXP_ROOT / "configs" / "yolo26n_original_copy.yaml"
REPORT = EXP_ROOT / "reports" / "yolo26n_structure_report.md"
JSON_REPORT = EXP_ROOT / "reports" / "yolo26n_structure.json"


def import_local_ultralytics():
    sys.path.insert(0, str(LOCAL_PARENT))
    from ultralytics.nn.tasks import DetectionModel  # noqa: PLC0415

    return DetectionModel


def find_yolo26_yaml() -> Path:
    candidates = [
        LOCAL_PKG / "cfg" / "models" / "26" / "yolo26.yaml",
        PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find yolo26.yaml in local copy or project ultralytics cfg.")


def yaml_load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore"))


def normalize_shape(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [normalize_shape(v) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_shape(v) for k, v in value.items()}
    return str(type(value).__name__)


def record_shapes(model, imgsz: int = 640) -> dict[int, list[int] | str | list | dict]:
    shapes: dict[int, list[int] | str | list | dict] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            shapes[index] = normalize_shape(output)

        return hook

    for i, layer in enumerate(model.model):
        handles.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, imgsz, imgsz))
    for handle in handles:
        handle.remove()
    return shapes


def shape_stride(shape, imgsz: int = 640) -> int | None:
    if isinstance(shape, list) and len(shape) == 4 and isinstance(shape[-1], int) and shape[-1] > 0:
        return imgsz // int(shape[-1])
    return None


def infer_structure(cfg: dict, layers: list[dict], shapes: dict[int, object]) -> dict:
    backbone = cfg["backbone"]
    all_defs = cfg["backbone"] + cfg["head"]
    backbone_len = len(backbone)

    stride_by_index = {i: shape_stride(shapes.get(i)) for i in range(len(all_defs))}
    p3_down, p4_down, p5_down = None, None, None
    for i, layer_def in enumerate(all_defs[:backbone_len]):
        _f, _n, module, args = layer_def
        if module == "Conv" and len(args) >= 3 and args[2] == 2:
            stride = stride_by_index.get(i)
            if stride == 8:
                p3_down = i
            elif stride == 16:
                p4_down = i
            elif stride == 32:
                p5_down = i

    head_backbone_refs: list[int] = []
    for f, _n, module, _args in cfg["head"]:
        refs = f if isinstance(f, list) else [f]
        if module == "Concat":
            for ref in refs:
                if isinstance(ref, int) and 0 <= ref < backbone_len:
                    head_backbone_refs.append(ref)

    p3_index = next((i for i in head_backbone_refs if stride_by_index.get(i) == 8), None)
    p4_index = next((i for i in head_backbone_refs if stride_by_index.get(i) == 16), None)
    p5_candidates = [i for i in range(backbone_len) if stride_by_index.get(i) == 32]
    p5_index = p5_candidates[-1] if p5_candidates else None

    detect_index = None
    detect_from = None
    for i, (_f, _n, module, _args) in enumerate(all_defs):
        if isinstance(module, str) and "Detect" in module:
            detect_index = i
            detect_from = _f

    if None in (p3_index, p4_index, p5_index, p3_down, p4_down, p5_down, detect_index):
        raise RuntimeError(
            "Could not confidently infer P3/P4/P5 or downsample layers. "
            f"p3={p3_index}, p4={p4_index}, p5={p5_index}, "
            f"down={p3_down, p4_down, p5_down}, detect={detect_index}"
        )

    p5_shape = shapes.get(p5_index)
    p5_channels = p5_shape[1] if isinstance(p5_shape, list) and len(p5_shape) == 4 else None
    if p5_channels is None:
        raise RuntimeError(f"Could not infer P5 channels from shape: {p5_shape}")

    return {
        "backbone_len": backbone_len,
        "p3_index": p3_index,
        "p4_index": p4_index,
        "p5_index": p5_index,
        "p3_downsample_index": p3_down,
        "p4_downsample_index": p4_down,
        "p5_downsample_index": p5_down,
        "downsample_indices": [p3_down, p4_down, p5_down],
        "detect_index": detect_index,
        "detect_from": detect_from,
        "p5_channels": p5_channels,
        "stride_by_index": stride_by_index,
    }


def layer_rows(model, cfg: dict, shapes: dict[int, object]) -> list[dict]:
    defs = cfg["backbone"] + cfg["head"]
    rows = []
    for i, layer in enumerate(model.model):
        rows.append(
            {
                "index": i,
                "from": getattr(layer, "f", defs[i][0]),
                "module": layer.__class__.__name__,
                "type": getattr(layer, "type", ""),
                "params": int(getattr(layer, "np", 0)),
                "shape": shapes.get(i),
                "stride": shape_stride(shapes.get(i)),
            }
        )
    return rows


def write_reports(yaml_path: Path, rows: list[dict], info: dict) -> None:
    serializable = copy.deepcopy(info)
    serializable["layers"] = rows
    serializable["yaml_path"] = str(yaml_path)
    JSON_REPORT.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# YOLO26n Structure Report",
        "",
        f"- YAML path: `{yaml_path}`",
        f"- Backbone length: `{info['backbone_len']}`",
        f"- P3 feature index: `{info['p3_index']}`",
        f"- P4 feature index: `{info['p4_index']}`",
        f"- P5 feature index: `{info['p5_index']}`",
        f"- P3/P4/P5 downsample Conv indices: `{info['downsample_indices']}`",
        f"- P5 channels: `{info['p5_channels']}`",
        f"- Detect index: `{info['detect_index']}`",
        f"- Detect from: `{info['detect_from']}`",
        "",
        "| idx | from | module | params | shape | stride |",
        "|---:|---|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | `{row['from']}` | `{row['module']}` | {row['params']} | `{row['shape']}` | `{row['stride']}` |"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DetectionModel = import_local_ultralytics()
    yaml_path = find_yolo26_yaml()
    CONFIG_COPY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(yaml_path, CONFIG_COPY)
    cfg = yaml_load(CONFIG_COPY)
    cfg["nc"] = 6

    model = DetectionModel(cfg, ch=3, nc=6, verbose=False)
    shapes = record_shapes(model)
    rows = layer_rows(model, cfg, shapes)
    info = infer_structure(cfg, rows, shapes)
    write_reports(yaml_path, rows, info)

    print(f"Structure report: {REPORT}")
    print(f"JSON report: {JSON_REPORT}")
    print(f"P3/P4/P5: {info['p3_index']}, {info['p4_index']}, {info['p5_index']}")
    print(f"Downsample Conv indices: {info['downsample_indices']}")


if __name__ == "__main__":
    main()
