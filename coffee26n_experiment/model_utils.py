"""Model construction, import isolation, budgets and auditable weight transfer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS = PACKAGE_ROOT / "local_ultralytics"


def setup_local_imports() -> dict[str, str]:
    """Make package-local Ultralytics the only runtime implementation in this process."""
    local = LOCAL_ULTRALYTICS.resolve()
    package = PACKAGE_ROOT.resolve()
    repo = package.parent.resolve()
    filtered = []
    for entry in sys.path:
        try:
            resolved = Path(entry or Path.cwd()).resolve()
        except OSError:
            filtered.append(entry)
            continue
        if resolved in {repo, package, local}:
            continue
        filtered.append(entry)
    sys.path[:] = [str(local), str(package), *filtered]
    for name, module in list(sys.modules.items()):
        if name == "ultralytics" or name.startswith("ultralytics."):
            file = getattr(module, "__file__", "")
            if file and str(local) not in str(Path(file).resolve()):
                del sys.modules[name]
    import ultralytics  # noqa: PLC0415
    from ultralytics.nn import modules  # noqa: PLC0415
    from ultralytics.nn import tasks  # noqa: PLC0415

    paths = {
        "ultralytics": str(Path(ultralytics.__file__).resolve()),
        "tasks": str(Path(tasks.__file__).resolve()),
        "block": str(Path(sys.modules["ultralytics.nn.modules.block"].__file__).resolve()),
    }
    if any(str(local) not in path for path in paths.values()):
        raise RuntimeError(f"Import isolation failed: {paths}")
    return paths


def _model_dict(path: Path, scale: str, nc: int) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["scale"] = scale
    payload["nc"] = int(nc)
    return payload


def build_model(model_cfg: Path, nc: int, *, scale: str = "n", loss_name: str = "native",
                profile: dict[str, Any] | None = None, loss_config: dict[str, Any] | None = None):
    """Build a package-local DetectionModel with runtime nc and optional criterion metadata."""
    setup_local_imports()
    from ultralytics.nn.tasks import DetectionModel  # noqa: PLC0415

    # Passing a dictionary preserves the requested scale; yaml_model_load() derives scale from a filename.
    model = DetectionModel(_model_dict(model_cfg, scale, nc), ch=3, nc=nc, verbose=False)
    # DetectionModel constructed directly from YAML has no trainer namespace; losses and dry-run need native gains.
    from ultralytics.cfg import get_cfg  # noqa: PLC0415
    model.args = get_cfg()
    model.coffee_loss_name = loss_name
    model.coffee_profile = profile or {}
    model.coffee_loss_config = loss_config or {}
    return model


def detect_metadata(model) -> dict[str, Any]:
    """Return runtime Detect sources and stride, independent of YAML indices."""
    detect = next((module for module in model.modules() if type(module).__name__ == "Detect"), None)
    if detect is None:
        raise RuntimeError("Model contains no Detect module.")
    stride = [float(value) for value in detect.stride.detach().cpu().tolist()]
    sources = list(detect.f)
    if len(stride) != 3 or any(abs(value - expected) > 1e-4 for value, expected in zip(stride, (8.0, 16.0, 32.0))):
        raise RuntimeError(f"Detect must have P3/P4/P5 stride [8,16,32], got {stride}.")
    return {"sources": sources, "stride": stride, "nc": int(detect.nc), "reg_max": int(detect.reg_max), "end2end": bool(getattr(model, "end2end", False))}


def parameter_budget(model_cfg: Path, nc: int, *, scale: str = "n", model=None,
                     variant: str | None = None, imgsz: tuple[int, int] = (640, 640)) -> dict[str, Any]:
    """Build same-nc n/s controls and calculate the dynamic YOLO26s hard limit."""
    if model is None:
        model = build_model(model_cfg, nc, scale=scale)
    native_cfg = PACKAGE_ROOT / "configs" / "models" / "yolo26n_native_control.yaml"
    native_n = build_model(native_cfg, nc, scale="n")
    native_s = build_model(native_cfg, nc, scale="s")
    model_params = sum(parameter.numel() for parameter in model.parameters())
    native_n_params = sum(parameter.numel() for parameter in native_n.parameters())
    native_s_params = sum(parameter.numel() for parameter in native_s.parameters())
    return {
        "schema_version": 1,
        "nc": int(nc),
        "scale": scale,
        "variant": variant or model_cfg.stem,
        "source_revision": "ultralytics-8.4.43",
        "comparison_family": "yolo26s",
        "comparison_nc": int(nc),
        "comparison_scales": {"n": scale, "s": "s"},
        "model_params": model_params,
        "native_n_params": native_n_params,
        "native_s_params": native_s_params,
        "hard_limit": native_s_params,
        "headroom_params": native_s_params - model_params,
        "within_hard_limit": model_params < native_s_params,
        "recommended_limit": int(native_n_params * 2.0),
        "within_recommended_limit": model_params <= native_n_params * 2.0,
        "gflops": None,
        "input_shape": [1, 3, int(imgsz[0]), int(imgsz[1])],
    }


def _variant_layer_map(variant: str) -> tuple[dict[int, int], dict[int, int]]:
    """Map target layers and MixDown.base branches to native YOLO26n layers."""
    aliases = {
        "c0": "r3",
        "d0": "c3",
        "d2": "core",
        "k0": "core",
        "k1": "p4mid",
        "k2": "p5lk7",
        "k3": "p5lk13",
        "k4": "full",
    }
    variant = aliases.get(variant, variant)
    if variant in {"native", "b0"}:
        return {index: index for index in range(24)}, {}
    common = {index: index for index in range(11)}
    r_map = {14: 13, 17: 16, 18: 17, 20: 19, 21: 20, 23: 22, 24: 23}
    c_new_down_map = {14: 13, 17: 16, 20: 19, 23: 22, 24: 23}
    maps = {
        "r0": r_map,
        "r1": r_map,
        "r2": r_map,
        "r3": r_map,
        "c1": c_new_down_map,
        "c2": c_new_down_map,
        "c3": c_new_down_map,
        "d1": c_new_down_map,
        "core": {14: 13, 17: 16, 21: 19, 24: 22, 25: 23},
        "x3": {14: 13, 17: 16, 21: 19, 24: 22, 25: 23},
        "x5": {14: 13, 17: 16, 21: 19, 24: 22, 25: 23},
        "x7": {14: 13, 17: 16, 21: 19, 24: 22, 25: 23},
        "p4mid": {14: 13, 17: 16, 21: 19, 25: 22, 26: 23},
        "p5lk7": {15: 13, 18: 16, 22: 19, 25: 22, 26: 23},
        "p5lk13": {15: 13, 18: 16, 22: 19, 25: 22, 26: 23},
        "full": {15: 13, 18: 16, 22: 19, 26: 22, 27: 23},
    }
    mixdown_bases = {
        "r0": {},
        "r1": {},
        "r2": {},
        "r3": {},
        "c1": {},
        "c2": {},
        "c3": {18: 17, 21: 20},
        "d1": {18: 17, 21: 20},
        "core": {19: 17, 22: 20},
        "x3": {3: 3, 19: 17, 22: 20},
        "x5": {5: 5, 19: 17, 22: 20},
        "x7": {7: 7, 19: 17, 22: 20},
        "p4mid": {19: 17, 23: 20},
        "p5lk7": {20: 17, 23: 20},
        "p5lk13": {20: 17, 23: 20},
        "full": {20: 17, 24: 20},
    }
    return {**common, **maps[variant]}, mixdown_bases[variant]


def transfer_weights(model, weights: Path, variant: str, *, target_nc: int, min_coverage: float = 99.0) -> dict[str, Any]:
    """Transfer shape-compatible native tensors and emit a detailed coverage report."""
    if not weights.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {weights}")
    setup_local_imports()
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    source_model = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if source_model is None or not hasattr(source_model, "state_dict"):
        raise ValueError(f"Checkpoint does not contain a model state_dict: {weights}")
    source_state = source_model.float().state_dict()
    target_state = model.float().state_dict()
    mapping, mixdown_bases = _variant_layer_map(variant)
    transferred: dict[str, torch.Tensor] = {}
    exact, remapped, skipped, missing, expected_head, new_keys = [], [], [], [], [], []
    eligible = 0
    detect_target = max(mapping)
    for target_key, target_value in target_state.items():
        if not target_key.startswith("model."):
            continue
        parts = target_key.split(".", 2)
        if len(parts) < 3:
            continue
        try:
            target_layer = int(parts[1])
        except ValueError:
            continue
        source_layer = mapping.get(target_layer)
        suffix = parts[2]
        if target_layer in mixdown_bases:
            if not suffix.startswith("base."):
                new_keys.append(target_key)
                continue
            source_layer = mixdown_bases[target_layer]
            suffix = suffix.removeprefix("base.")
        if source_layer is None:
            new_keys.append(target_key)
            continue
        source_key = f"model.{source_layer}.{suffix}"
        source_value = source_state.get(source_key)
        if source_value is None:
            if ".texture_enhance." in target_key:
                new_keys.append(target_key)
                continue
            eligible += target_value.numel()
            missing.append(target_key)
            continue
        is_detect_head = target_layer == detect_target and ("cv3" in target_key or "one2one_cv3" in target_key)
        if source_value.shape != target_value.shape:
            if is_detect_head:
                expected_head.append({"source": source_key, "target": target_key, "source_shape": list(source_value.shape), "target_shape": list(target_value.shape)})
            else:
                eligible += target_value.numel()
                skipped.append({"source": source_key, "target": target_key, "reason": "shape", "source_shape": list(source_value.shape), "target_shape": list(target_value.shape)})
            continue
        eligible += target_value.numel()
        transferred[target_key] = source_value.detach().clone()
        entry = {"source": source_key, "target": target_key, "numel": target_value.numel()}
        if target_layer == source_layer and target_layer not in mixdown_bases:
            exact.append(target_key)
        else:
            remapped.append(entry)
    model.load_state_dict(transferred, strict=False)
    loaded_state = model.state_dict()
    copy_mismatches = [key for key, value in transferred.items() if not torch.equal(loaded_state[key].detach().cpu(), value.detach().cpu())]
    if copy_mismatches:
        raise RuntimeError(f"Pretrained tensor copy verification failed: {copy_mismatches[:5]}")
    loaded_numel = sum(value.numel() for value in transferred.values())
    gates = {}
    expected_gates = {"ResBridge": torch.sigmoid(torch.tensor(-3.0)).item(), "ContextBlock": torch.sigmoid(torch.tensor(-3.0)).item(),
                      "MixDown": torch.sigmoid(torch.tensor(-4.0)).item(), "DetailBlock": 0.0, "MidBlock": 0.0}
    for name, module in model.named_modules():
        if hasattr(module, "effective_gate"):
            try:
                module_type = type(module).__name__
                value = float(module.effective_gate().detach().cpu().item())
                gates[name or module_type] = {"type": module_type, "value": value}
                if module_type in expected_gates and abs(value - expected_gates[module_type]) > 1e-6:
                    raise RuntimeError(f"Unexpected {module_type} gate at {name}: {value} != {expected_gates[module_type]}")
            except Exception:
                raise
    coverage = 100.0 * loaded_numel / max(eligible, 1)
    if missing or skipped or coverage < float(min_coverage):
        raise RuntimeError(
            f"Pretrained coverage gate failed for {variant}: coverage={coverage:.4f}% < {min_coverage}%, "
            f"missing={len(missing)}, skipped={len(skipped)}."
        )
    total_target = sum(value.numel() for value in target_state.values())
    return {
        "source_weights": str(weights.resolve()),
        "source_model": "yolo26n",
        "source_nc": 80,
        "target_nc": int(target_nc),
        "exact_matched_keys": exact,
        "remapped_keys": remapped,
        "skipped_shape_keys": skipped,
        "missing_target_keys": missing,
        "new_parameter_keys": new_keys,
        "expected_head_mismatches": expected_head,
        "source_numel_considered": sum(value.numel() for value in source_state.values()),
        "target_numel_loaded": loaded_numel,
        "eligible_native_target_numel": eligible,
        "coverage_percent": coverage,
        "minimum_coverage_percent": float(min_coverage),
        "copy_mismatches": copy_mismatches,
        "loaded_over_total_target_percent": 100.0 * loaded_numel / max(total_target, 1),
        "gate_initialization": gates,
    }


def finite_forward(model, imgsz: tuple[int, int]) -> dict[str, Any]:
    """Run one CPU forward and verify every returned tensor is finite."""
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, imgsz[0], imgsz[1]))
    tensors = []
    def collect(value):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
    collect(output)
    if not tensors or any(not torch.isfinite(tensor).all().item() for tensor in tensors):
        raise RuntimeError("Model forward produced no finite tensors.")
    return {"tensor_count": len(tensors), "shapes": [list(tensor.shape) for tensor in tensors]}
