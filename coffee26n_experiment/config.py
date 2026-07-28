"""Strict profile and CLI configuration handling for the Coffee26n package."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = PACKAGE_ROOT / "configs" / "profiles"
MODEL_ROOT = PACKAGE_ROOT / "configs" / "models"

MODEL_FILES = {
    "native": "yolo26n_native_control.yaml",
    "b0": "yolo26n_b0_control.yaml",
    "r0": "coffee26n_r0.yaml",
    "r1": "coffee26n_r1.yaml",
    "r2": "coffee26n_r2.yaml",
    "r3": "coffee26n_r3.yaml",
    "c0": "coffee26n_r3.yaml",
    "c1": "coffee26n_c1_ldsconv.yaml",
    "c2": "coffee26n_c2_scdown.yaml",
    "c3": "coffee26n_c3_mixdown.yaml",
    "d0": "coffee26n_c3_mixdown.yaml",
    "d1": "coffee26n_d1_bounded_elteb.yaml",
    "d2": "coffee26n_core.yaml",
    "core": "coffee26n_core.yaml",
    "x3": "coffee26n_x3.yaml",
    "x5": "coffee26n_x5.yaml",
    "x7": "coffee26n_x7.yaml",
    "k0": "coffee26n_core.yaml",
    "k1": "coffee26n_p4mid.yaml",
    "k2": "coffee26n_p5lk7.yaml",
    "k3": "coffee26n_p5lk13.yaml",
    "k4": "coffee26n_full.yaml",
    "p4mid": "coffee26n_p4mid.yaml",
    "p5lk7": "coffee26n_p5lk7.yaml",
    "p5lk13": "coffee26n_p5lk13.yaml",
    "full": "coffee26n_full.yaml",
}
VARIANTS = tuple(MODEL_FILES)
LOSS_NAMES = ("native", "coffee_l1", "coffee_l2", "coffee_l3", "coffee_l4")

GENERIC_DEFAULTS = {
    "schema_version": 1,
    "name": "generic",
    "dataset_yaml": None,
    "expected_nc": None,
    "expected_names": None,
    "tail_bce": {"enabled": False, "counts": None, "cap": 1.8, "exponent": 0.5, "ramp_start": 0.10, "ramp_end": 0.60},
    "pair_margin": {
        "enabled": False,
        "pairs": [],
        "auto_from_counts": False,
        "auto_topk": 1,
        "auto_min_ratio": 1.25,
        "quality_threshold": 0.30,
        "margin_min": 0.15,
        "margin_max": 0.30,
        "margin_base": 0.15,
        "margin_kappa": 0.05,
        "softplus_beta": 10.0,
        "lambda_max": 0.20,
        "ramp_start": 0.10,
        "ramp_end": 0.60,
    },
    "bgmix": {"enabled": False, "p": 0.0, "box_expand": 0.02, "feather_px": 16, "blur_kernel": 21, "no_box": "skip"},
    "structure": {"context": False, "context_kernel": 13, "mid": False, "mid_kernel": 9, "mixdown": True},
    "budget": {"compare_family": "yolo26s", "recommended_n_multiplier": 2.0},
    "loss": {
        "wiou_enabled": True,
        "wiou_ramp_start": 0.10,
        "wiou_ramp_end": 0.60,
        "wiou_alpha": 1.7,
        "wiou_delta": 2.7,
        "wiou_momentum": 0.0001,
        "wiou_focus_min": 0.5,
        "wiou_focus_max": 3.0,
        "wiou_distance_gain_max": 1.8,
    },
}

_TOP_KEYS = set(GENERIC_DEFAULTS)
_NESTED_KEYS = {key: set(value) for key, value in GENERIC_DEFAULTS.items() if isinstance(value, dict)}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries without accepting unknown keys."""
    result = deepcopy(base)
    for key, value in override.items():
        if key not in _TOP_KEYS:
            raise ValueError(f"Unknown profile/config key {key!r}; expected one of {sorted(_TOP_KEYS)}.")
        if isinstance(value, dict):
            if not isinstance(result.get(key), dict):
                raise ValueError(f"Config key {key!r} must be a mapping.")
            unknown = set(value) - _NESTED_KEYS[key]
            if unknown:
                raise ValueError(f"Unknown keys under {key!r}: {sorted(unknown)}.")
            result[key].update(deepcopy(value))
        else:
            result[key] = deepcopy(value)
    return result


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a mapping YAML file with a useful path in errors."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid YAML at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"YAML at {path} must contain a mapping, got {type(value).__name__}.")
    return value


def resolve_profile(reference: str | Path | None) -> tuple[dict[str, Any], Path]:
    """Load generic or a named/path profile and reject unknown keys."""
    if reference is None or str(reference).lower() == "generic":
        path = PROFILE_ROOT / "generic.yaml"
    else:
        candidate = Path(reference)
        path = candidate if candidate.is_file() else PROFILE_ROOT / f"{candidate.name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Profile not found: {reference!r} (resolved {path})")
    raw = read_yaml(path)
    merged = deep_merge(GENERIC_DEFAULTS, raw)
    if merged["name"] != raw.get("name", merged["name"]):
        raise ValueError(f"Profile name mismatch in {path}: {raw.get('name')!r}")
    return merged, path.resolve()


def parse_imgsz(value: int | str | tuple[int, int] | list[int] | None, default: int = 640) -> tuple[int, int]:
    """Parse a square integer or H,W image size without a dataset-specific default."""
    if value is None:
        return default, default
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        parts = [part.strip() for part in str(value).split(",")]
        if len(parts) == 1:
            result = (int(parts[0]), int(parts[0]))
        elif len(parts) == 2:
            result = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError(f"imgsz must be INT or H,W, got {value!r}.")
    if min(result) <= 0:
        raise ValueError(f"imgsz dimensions must be positive, got {result}.")
    return result


def make_resolved_config(profile: dict[str, Any], *, data_yaml: Path, dataset: dict[str, Any], variant: str,
                         loss: str, bgmix: str | float, imgsz: tuple[int, int], seed: int, weights: str,
                         device: str = "cpu", cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply CLI overrides after profile and dataset facts, then validate feature requests."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected {VARIANTS}.")
    if loss not in LOSS_NAMES:
        raise ValueError(f"Unknown loss {loss!r}; expected {LOSS_NAMES}.")
    result = deepcopy(profile)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if key in {"loss", "variant", "bgmix", "imgsz", "seed", "weights", "device"}:
                continue
            if key not in result:
                raise ValueError(f"Unknown CLI config key {key!r}.")
            result[key] = deepcopy(value)
    result["dataset"] = {
        "yaml": str(data_yaml.resolve()),
        "nc": int(dataset["nc"]),
        "names": list(dataset["names"]),
        "data_lock_id": dataset.get("data_lock_id"),
    }
    result["variant"] = variant
    result["loss_name"] = loss
    result["bgmix_request"] = bgmix
    result["imgsz"] = list(imgsz)
    result["seed"] = int(seed)
    result["weights"] = str(weights)
    result["device"] = str(device)
    validate_profile_dataset(profile, data_yaml, dataset)
    if loss == "coffee_l4" and not profile["pair_margin"].get("enabled"):
        raise ValueError("coffee_l4 requires profile pair_margin.enabled=true.")
    if isinstance(bgmix, str) and bgmix.lower() == "off":
        result["bgmix_request"] = "off"
    elif isinstance(bgmix, str) and bgmix.lower() == "auto":
        if not profile["bgmix"].get("enabled"):
            raise ValueError("--bgmix auto requires a profile with bgmix.enabled=true.")
        result["bgmix_request"] = float(profile["bgmix"].get("p", 0.0))
    else:
        probability = float(bgmix)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"--bgmix must be off or a probability in [0,1], got {bgmix!r}.")
        if probability and not profile["bgmix"].get("enabled"):
            raise ValueError("BgMix was requested but the selected profile has bgmix.enabled=false.")
        result["bgmix_request"] = probability
    return result


def validate_profile_dataset(profile: dict[str, Any], data_yaml: str | Path, dataset: dict[str, Any]) -> None:
    """Fail when an opt-in profile does not describe the explicitly supplied dataset."""
    actual_path = Path(data_yaml).resolve()
    declared_path = profile.get("dataset_yaml")
    if declared_path:
        expected_path = Path(str(declared_path))
        expected_path = expected_path if expected_path.is_absolute() else PACKAGE_ROOT.parent / expected_path
        expected_path = expected_path.resolve()
        if actual_path != expected_path:
            raise ValueError(
                f"Profile dataset_yaml={declared_path!r} resolves to {expected_path}, but --data resolves to {actual_path}."
            )
    if profile.get("expected_nc") is not None and int(profile["expected_nc"]) != int(dataset["nc"]):
        raise ValueError(f"Profile expected_nc={profile['expected_nc']} but dataset nc={dataset['nc']} from {actual_path}.")
    expected_names = profile.get("expected_names")
    if expected_names is not None and list(expected_names) != list(dataset["names"]):
        raise ValueError(
            f"Profile expected_names={expected_names!r} but dataset names={dataset['names']!r} from {actual_path}."
        )


def resolve_profile_runtime(profile: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Resolve optional profile class names to IDs using the supplied dataset only.

    Profiles remain readable with class names, while loss code receives a stable integer-key mapping.
    """
    names = list(dataset["names"])

    def class_id(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(f"Boolean is not a valid class identifier: {value!r}.")
        if isinstance(value, int):
            index = value
        else:
            text = str(value)
            if text not in names:
                raise ValueError(f"Profile class {text!r} is absent from dataset names {names!r}.")
            index = names.index(text)
        if not 0 <= index < len(names):
            raise ValueError(f"Profile class id {index} outside dataset range [0,{len(names)}).")
        return index

    runtime = deepcopy(profile)
    counts = profile.get("tail_bce", {}).get("counts")
    if isinstance(counts, dict):
        runtime["tail_bce"]["counts"] = {class_id(key): float(value) for key, value in counts.items()}
    elif counts is not None:
        runtime["tail_bce"]["counts"] = [float(value) for value in counts]
    elif runtime.get("tail_bce", {}).get("enabled"):
        observed = dataset.get("class_counts") or {}
        runtime["tail_bce"]["counts"] = {index: float(observed.get(index, 0)) for index in range(len(names))}

    pairs = []
    for pair in profile.get("pair_margin", {}).get("pairs", []):
        if isinstance(pair, dict):
            if "true_name" not in pair or "rival_name" not in pair:
                raise ValueError(f"PairMargin pair must define true_name and rival_name: {pair!r}.")
            true_id, rival_id = class_id(pair["true_name"]), class_id(pair["rival_name"])
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            true_id, rival_id = class_id(pair[0]), class_id(pair[1])
        else:
            raise ValueError(f"Invalid PairMargin pair: {pair!r}.")
        if true_id == rival_id:
            raise ValueError(f"PairMargin true and rival classes must differ: {pair!r}.")
        if (true_id, rival_id) in pairs:
            raise ValueError(f"Duplicate PairMargin pair: {pair!r}.")
        pairs.append((true_id, rival_id))
    pair_cfg = runtime.get("pair_margin", {})
    if not pairs and pair_cfg.get("enabled") and pair_cfg.get("auto_from_counts"):
        observed = dataset.get("class_counts") or {}
        topk = max(int(pair_cfg.get("auto_topk", 1)), 0)
        min_ratio = max(float(pair_cfg.get("auto_min_ratio", 1.25)), 1.0)
        for true_id in range(len(names)):
            true_count = float(observed.get(true_id, 0))
            candidates = [
                rival_id
                for rival_id in range(len(names))
                if rival_id != true_id
                and float(observed.get(rival_id, 0)) >= max(true_count * min_ratio, true_count + 1.0)
            ]
            candidates.sort(key=lambda rival_id: (-float(observed.get(rival_id, 0)), rival_id))
            pairs.extend((true_id, rival_id) for rival_id in candidates[:topk])
    runtime["pair_ids"] = pairs
    runtime["class_counts"] = runtime.get("tail_bce", {}).get("counts")
    observed = dataset.get("class_counts")
    if runtime["class_counts"] is not None and observed is not None:
        if isinstance(runtime["class_counts"], dict):
            declared = {index: float(runtime["class_counts"].get(index, 0)) for index in range(len(names))}
        else:
            declared = {
                index: float(runtime["class_counts"][index]) if index < len(runtime["class_counts"]) else 0.0
                for index in range(len(names))
            }
        actual = {index: float(observed.get(index, 0)) for index in range(len(names))}
        if declared != actual:
            raise ValueError(f"Profile TailBCE counts do not match current train labels: declared={declared}, actual={actual}.")
    return runtime
