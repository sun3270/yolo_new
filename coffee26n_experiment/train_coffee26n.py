"""Unified Coffee26n training and offline dry-run entry point."""

from __future__ import annotations

import argparse
import json
import platform
import random
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from coffee26n_experiment.config import (  # noqa: E402
    MODEL_FILES,
    make_resolved_config,
    parse_imgsz,
    resolve_profile,
    resolve_profile_runtime,
)
from coffee26n_experiment.augmentations import BatchCoffeeBgMix  # noqa: E402
from coffee26n_experiment.dataset import parse_dataset, write_data_lock  # noqa: E402
from coffee26n_experiment.manifest import sha256_file, verify_source_manifest, write_source_manifest  # noqa: E402
from coffee26n_experiment.model_utils import (  # noqa: E402
    build_model,
    detect_metadata,
    finite_forward,
    parameter_budget,
    setup_local_imports,
    transfer_weights,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or dry-run a self-contained Coffee26n variant.")
    parser.add_argument("--data", required=True, help="Explicit YOLO dataset YAML path.")
    parser.add_argument("--variant", choices=tuple(MODEL_FILES), default="full")
    parser.add_argument("--profile", default="auto_full", help="generic, auto_full, a named profile, or a profile YAML path.")
    parser.add_argument("--loss", choices=("native", "coffee_l1", "coffee_l2", "coffee_l3", "coffee_l4"), default="coffee_l4")
    parser.add_argument("--bgmix", default="auto", help="auto, off, or probability in [0,1].")
    parser.add_argument("--bgmix-close-epoch", type=int, default=150, help="Last epoch using BgMix; 0 keeps it enabled.")
    parser.add_argument("--imgsz", default="960", help="INT or H,W.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default="yolo26n.pt", help="Local checkpoint path, or none. No download is attempted.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="MuSGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=4.0)
    parser.add_argument("--warmup-momentum", type=float, default=0.8)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument(
        "--fitness-metric",
        choices=("map50-95", "map50"),
        default="map50-95",
        help="Metric used by EarlyStopping and best.pt selection.",
    )
    parser.add_argument("--nbs", type=int, default=64)
    parser.add_argument("--mosaic", type=float, default=0.7)
    parser.add_argument("--close-mosaic", type=int, default=150)
    parser.add_argument("--hsv-h", type=float, default=0.005)
    parser.add_argument("--hsv-s", type=float, default=0.20)
    parser.add_argument("--hsv-v", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--erasing", type=float, default=0.0)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", default=None, help="Run output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Build, forward and (for CoffeeLoss) evaluate one synthetic batch.")
    return parser


FITNESS_METRIC_KEYS = {
    "map50-95": "metrics/mAP50-95(B)",
    "map50": "metrics/mAP50(B)",
}


def _select_fitness(metrics: dict[str, Any], metric: str) -> float:
    """Select the run-level fitness without changing the validator's recorded metrics."""
    key = FITNESS_METRIC_KEYS[metric]
    if key not in metrics:
        raise KeyError(f"Validation did not return the requested fitness metric {key!r}.")
    fitness = float(metrics[key])
    if not np.isfinite(fitness):
        raise ValueError(f"Validation returned a non-finite {key}: {fitness}.")
    return fitness


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _synthetic_batch(nc: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "img": torch.zeros(1, 3, 64, 64, device=device),
        "batch_idx": torch.zeros(1, dtype=torch.long, device=device),
        "cls": torch.zeros(1, dtype=torch.float32, device=device),
        "bboxes": torch.tensor([[0.50, 0.50, 0.30, 0.30]], dtype=torch.float32, device=device),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_monitor_headers(output: Path) -> None:
    headers = {
        "per_class_metrics.csv": "epoch,class_id,class_name,precision,recall,map50,map50_95\n",
        "confusion_matrix.csv": "epoch,true_id,pred_id,count\n",
        "confusion_pairs.csv": "epoch,true_id,true_name,rival_id,rival_name,count,rate\n",
        "class_false_positives.csv": "epoch,class_id,class_name,false_positives\n",
        "module_monitor.csv": "epoch,module_path,module_type,gate,p3_weight,p5_weight,entropy,steps\n",
        "class_weight_monitor.csv": "epoch,class_id,class_name,count,base_weight,effective_weight\n",
        "pair_monitor.csv": "epoch,branch,true_id,true_name,rival_id,rival_name,eligible_count,violations,pair_loss\n",
        "loss_monitor.csv": "epoch,branch,base_bce,legacy_prog,tail_bce,pair_margin,ciou,wiou,wiou_blend,norm_l1\n",
        "bgmix_monitor.csv": "epoch,status,transform,preserve_ratio,applied,skipped_no_boxes,skipped_probability\n",
    }
    for filename, header in headers.items():
        (output / filename).write_text(header, encoding="utf-8")


def _append_csv(path: Path, row: list[Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(",".join(str(value) for value in row) + "\n")


def _model_summary(model, metadata: dict[str, Any]) -> str:
    rows = [
        f"parameters: {sum(parameter.numel() for parameter in model.parameters())}",
        f"detect_sources: {metadata['sources']}",
        f"detect_stride: {metadata['stride']}",
        f"nc: {metadata['nc']}",
        f"reg_max: {metadata['reg_max']}",
        "layers:",
    ]
    for module in model.model:
        rows.append(f"{module.i:>3} from={module.f!s:<14} params={module.np:<10} type={module.type}")
    return "\n".join(rows) + "\n"


def _append_module_snapshot(path: Path, model, epoch: int) -> None:
    rows = []
    for name, module in model.named_modules():
        gate = ""
        if hasattr(module, "effective_gate"):
            gate = f"{float(module.effective_gate().detach().cpu().item()):.10g}"
        summary = module.monitor_summary() if hasattr(module, "monitor_summary") else None
        if gate or summary is not None:
            summary = summary or {}
            rows.append(
                f"{epoch},{name},{type(module).__name__},{gate},"
                f"{summary.get('p3_weight', '')},{summary.get('p5_weight', '')},"
                f"{summary.get('entropy', '')},{summary.get('steps', '')}"
            )
        if summary is not None and hasattr(module, "reset_monitor"):
            module.reset_monitor()
    if rows:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")


def _reset_module_monitors(model) -> None:
    for module in model.modules():
        if hasattr(module, "reset_monitor"):
            module.reset_monitor()


def _resolve_weights(value: str) -> Path | None:
    if str(value).lower() in {"none", "", "null"}:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and not candidate.is_file():
        candidate = PACKAGE_ROOT / candidate
    path = candidate.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--weights must point to an existing local file; refusing download: {path}")
    return path


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": platform.platform(),
    }


def _write_static_artifacts(output: Path, args: argparse.Namespace, profile: dict[str, Any], data: dict[str, Any],
                            manifest_path: Path, model_cfg: Path, model, metadata: dict[str, Any],
                            budget: dict[str, Any], forward: dict[str, Any], latency_ms: float) -> None:
    (output / "cli_args.yaml").write_text(yaml.safe_dump(vars(args), sort_keys=False), encoding="utf-8")
    (output / "args.yaml").write_text(yaml.safe_dump(vars(args), sort_keys=False), encoding="utf-8")
    (output / "profile_snapshot.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    (output / "dataset_yaml_copy.yaml").write_text(yaml.safe_dump(data["raw"], sort_keys=False), encoding="utf-8")
    (output / "source_manifest.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    model_yaml = yaml.safe_load(model_cfg.read_text(encoding="utf-8"))
    model_yaml["nc"] = int(data["nc"])
    model_yaml["scale"] = "n"
    (output / "model.yaml").write_text(yaml.safe_dump(model_yaml, sort_keys=False), encoding="utf-8")
    (output / "model_summary.txt").write_text(_model_summary(model, metadata), encoding="utf-8")
    _write_json(output / "param_budget.json", budget)
    _write_json(output / "pretrain_transfer.json", {"status": "pending"})
    _write_json(output / "latency.json", {"forward_wall_ms": latency_ms, "imgsz": list(args.imgsz) if isinstance(args.imgsz, (list, tuple)) else str(args.imgsz), "forward": forward})
    _write_monitor_headers(output)


def _record_training_diagnostics(output: Path, trainer, class_names: list[str], epoch: int) -> None:
    model = getattr(trainer.model, "module", trainer.model)
    _append_module_snapshot(output / "module_monitor.csv", model, epoch)
    criterion = getattr(model, "criterion", None)
    branches = [("one2many", getattr(criterion, "one2many", criterion)), ("one2one", getattr(criterion, "one2one", None))]
    from coffee26n_experiment.loss.loss_extensions import tail_ramp  # noqa: PLC0415
    for branch_name, branch in branches:
        if branch is None or not hasattr(branch, "last_terms"):
            continue
        terms = branch.last_terms
        _append_csv(output / "loss_monitor.csv", [epoch, branch_name, terms.get("base_bce", 0.0), terms.get("legacy_prog", 0.0),
                                                     terms.get("tail_bce", 0.0), terms.get("pair_margin", 0.0), terms.get("ciou", 0.0),
                                                     terms.get("wiou", 0.0), terms.get("wiou_blend", 0.0), terms.get("norm_l1", 0.0)])
        counts = getattr(branch, "class_counts", None)
        weights = getattr(branch, "tail_weights", None)
        if weights is not None:
            ramp = tail_ramp(getattr(branch, "progress", 0.0), float(getattr(branch, "tail_cfg", {}).get("ramp_start", 0.10)),
                             float(getattr(branch, "tail_cfg", {}).get("ramp_end", 0.60)))
            for class_id, value in enumerate(weights.detach().cpu().tolist()):
                count = counts.get(class_id, 0) if isinstance(counts, dict) else (counts[class_id] if counts else 0)
                effective = 1.0 + ramp * (float(value) - 1.0)
                _append_csv(output / "class_weight_monitor.csv", [epoch, class_id, class_names[class_id], count, 1.0, effective])
        diagnostics = getattr(branch, "last_pair_diagnostics", {})
        per_pair = diagnostics.get("pairs", {})
        for true_id, rival_id in getattr(branch, "pairs", []):
            pair_diag = per_pair.get(f"{true_id}:{rival_id}", {})
            _append_csv(output / "pair_monitor.csv", [epoch, branch_name, true_id, class_names[true_id], rival_id, class_names[rival_id],
                                                         pair_diag.get("eligible", 0), pair_diag.get("violations", 0), pair_diag.get("loss", 0.0)])


def _record_bgmix_stats(output: Path, trainer, epoch: int) -> None:
    batch_bgmix = getattr(trainer, "batch_bgmix", None)
    if batch_bgmix is not None:
        stats = batch_bgmix.stats
        close_epoch = int(getattr(trainer.args, "coffee_bgmix_close_epoch", 0))
        active = not close_epoch or epoch <= close_epoch
        _append_csv(output / "bgmix_monitor.csv", [epoch, "batch_gpu" if active else "disabled", stats.get("last_transform"),
                                                     stats.get("preserve_ratio", 0.0), stats.get("applied", 0),
                                                     stats.get("skipped_no_boxes", 0), stats.get("skipped_probability", 0)])
        batch_bgmix.reset()
        return
    dataset = getattr(trainer, "train_loader", None)
    dataset = getattr(dataset, "dataset", None)
    transforms = getattr(dataset, "transforms", None)
    for transform in getattr(transforms, "transforms", transforms or []):
        if type(transform).__name__ != "CoffeeBgMix":
            continue
        stats = transform.snapshot_stats(reset=True) if hasattr(transform, "snapshot_stats") else getattr(transform, "stats", {})
        _append_csv(output / "bgmix_monitor.csv", [epoch, "runtime", stats.get("last_transform"), stats.get("preserve_ratio", 0.0),
                                                     stats.get("applied", 0), stats.get("skipped_no_boxes", 0), stats.get("skipped_probability", 0)])
        return


def _record_validation_diagnostics(output: Path, trainer, class_names: list[str], pairs: list[tuple[int, int]], epoch: int) -> None:
    validator = getattr(trainer, "validator", None)
    metrics = getattr(validator, "metrics", None)
    if metrics is None:
        return
    class_rows = {int(class_id): index for index, class_id in enumerate(getattr(metrics, "ap_class_index", []))}
    for class_id, class_name in enumerate(class_names):
        values = metrics.class_result(class_rows[class_id]) if class_id in class_rows else (0.0, 0.0, 0.0, 0.0)
        _append_csv(output / "per_class_metrics.csv", [epoch, class_id, class_name, *[float(value) for value in values]])
    confusion = getattr(getattr(validator, "confusion_matrix", None), "matrix", None)
    if confusion is None:
        return
    nc = len(class_names)
    for true_id in range(nc + 1):
        for pred_id in range(nc + 1):
            count = float(confusion[pred_id, true_id])
            if count:
                _append_csv(output / "confusion_matrix.csv", [epoch, true_id, pred_id, count])
    for class_id, class_name in enumerate(class_names):
        _append_csv(output / "class_false_positives.csv", [epoch, class_id, class_name, float(confusion[class_id, nc])])
    for true_id, rival_id in pairs:
        count = float(confusion[rival_id, true_id])
        denominator = float(confusion[:, true_id].sum())
        _append_csv(
            output / "confusion_pairs.csv",
            [epoch, true_id, class_names[true_id], rival_id, class_names[rival_id], count, count / denominator if denominator else 0.0],
        )


def _training_overrides(args: argparse.Namespace, imgsz: tuple[int, int]) -> dict[str, Any]:
    return {
        "epochs": int(args.epochs),
        "imgsz": imgsz[0] if imgsz[0] == imgsz[1] else list(imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": str(args.device),
        "seed": int(args.seed),
        "optimizer": str(args.optimizer),
        "lr0": float(args.lr0),
        "lrf": float(args.lrf),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": float(args.warmup_epochs),
        "warmup_momentum": float(args.warmup_momentum),
        "warmup_bias_lr": float(args.warmup_bias_lr),
        "patience": int(args.patience),
        "nbs": int(args.nbs),
        "mosaic": float(args.mosaic),
        "close_mosaic": int(args.close_mosaic),
        "hsv_h": float(args.hsv_h),
        "hsv_s": float(args.hsv_s),
        "hsv_v": float(args.hsv_v),
        "scale": float(args.scale),
        "erasing": float(args.erasing),
        "cache": bool(args.cache),
        "amp": bool(args.amp),
        "deterministic": bool(args.deterministic),
        "cos_lr": bool(args.cos_lr),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch <= 0 or args.workers < 0 or args.nbs <= 0:
        raise ValueError("--epochs/--batch/--nbs must be positive and --workers must be nonnegative.")
    if args.close_mosaic < 0 or args.bgmix_close_epoch < 0 or args.patience < 0 or args.warmup_epochs < 0:
        raise ValueError("--close-mosaic/--bgmix-close-epoch/--patience/--warmup-epochs must be nonnegative.")
    for name in ("mosaic", "hsv_h", "hsv_s", "hsv_v", "scale", "erasing"):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative, got {value}.")
    _seed_everything(args.seed)
    data = parse_dataset(args.data)
    profile, profile_path = resolve_profile(args.profile)
    runtime_profile = resolve_profile_runtime(profile, data)
    effective_counts = runtime_profile.get("class_counts")
    if effective_counts is None:
        effective_counts = {index: float(data["class_counts"].get(index, 0)) for index in range(data["nc"])}
    imgsz = parse_imgsz(args.imgsz)
    training = _training_overrides(args, imgsz)
    resolved = make_resolved_config(
        runtime_profile,
        data_yaml=Path(args.data).resolve(),
        dataset=data,
        variant=args.variant,
        loss=args.loss,
        bgmix=args.bgmix,
        imgsz=imgsz,
        seed=args.seed,
        weights=args.weights,
        device=args.device,
    )
    resolved["profile_path"] = str(profile_path)
    resolved["class_counts"] = effective_counts
    resolved["pair_ids"] = runtime_profile.get("pair_ids", [])
    resolved["loss"] = runtime_profile.get("loss", {})
    resolved["profile_runtime"] = runtime_profile
    resolved["fitness_metric"] = args.fitness_metric
    run_name = args.name or f"{args.variant}_{args.loss}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    project = Path(args.project).resolve() if args.project else PACKAGE_ROOT / "runs"
    output = Path(args.output).resolve() if args.output else project / run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run output already exists and is non-empty; choose a new --name/--output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    source_manifest_path = PACKAGE_ROOT / "source_manifest.json"
    if not source_manifest_path.is_file():
        write_source_manifest(PACKAGE_ROOT)
    mismatches = verify_source_manifest(PACKAGE_ROOT)
    if mismatches:
        raise RuntimeError(f"Source manifest mismatch before run: {mismatches[:5]}")
    import_paths: dict[str, str] = {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "status": "running",
        "mode": "dry_run" if args.dry_run else "train",
        "command": shlex.join([str(value) for value in sys.argv]),
        "argv": list(sys.argv),
        "output": str(output),
        "variant": args.variant,
        "profile": profile["name"],
        "profile_path": str(profile_path),
        "data_yaml": str(Path(args.data).resolve()),
        "data_lock_id": data["data_lock_id"],
        "nc": int(data["nc"]),
        "names": list(data["names"]),
        "imgsz": list(imgsz),
        "seed": int(args.seed),
        "device": str(args.device),
        "fitness_metric": args.fitness_metric,
        "environment": _environment(),
        "imports": import_paths,
        "started_at": started_at,
        "ended_at": None,
        "git": "not-a-git-repo",
        "failure": None,
    }
    try:
        resolved["training"] = {**training, "project": str(project), "name": run_name}
        write_data_lock(data, output, runtime_profile.get("pair_ids", []))
        runtime_data_yaml = output / "runtime_data.yaml"
        runtime_data = dict(data["raw"])
        runtime_data["path"] = str(data["root"])
        runtime_data_yaml.write_text(yaml.safe_dump(runtime_data, sort_keys=False), encoding="utf-8")
        import_paths = setup_local_imports()
        report["imports"] = import_paths
        manifest_sha = sha256_file(source_manifest_path)
        report["source_manifest_sha256"] = manifest_sha
        model_cfg = PACKAGE_ROOT / "configs" / "models" / MODEL_FILES[args.variant]
        loss_config = {**resolved, "loss": resolved["loss"]}
        model = build_model(model_cfg, data["nc"], loss_name=args.loss, profile=runtime_profile, loss_config=loss_config)
        for key, value in training.items():
            setattr(model.args, key, value)
        report["model"] = detect_metadata(model)
        resolved["model_structure"] = {
            "variant": args.variant,
            "modules": [type(module).__name__ for module in model.model],
            "detect_sources": report["model"]["sources"],
            "detect_stride": report["model"]["stride"],
        }
        budget = parameter_budget(model_cfg, data["nc"], model=model, variant=args.variant, imgsz=imgsz)
        if not budget["within_hard_limit"]:
            raise RuntimeError(f"{args.variant}/nc={data['nc']} exceeds dynamic YOLO26s hard limit: {budget}")
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        report["resolved_config_sha256"] = sha256_file(output / "resolved_config.yaml")
        forward_start = time.perf_counter()
        report["forward"] = finite_forward(model, imgsz)
        latency_ms = (time.perf_counter() - forward_start) * 1000.0
        _write_static_artifacts(output, args, profile, data, source_manifest_path, model_cfg, model, report["model"], budget, report["forward"], latency_ms)
        weights = _resolve_weights(args.weights)
        if weights is not None:
            report["weights"] = transfer_weights(model, weights, args.variant, target_nc=data["nc"])
        else:
            report["weights"] = {"status": "skipped", "reason": "--weights none"}
        _write_json(output / "pretrain_transfer.json", report["weights"])
        if args.loss != "native":
            criterion = model.init_criterion()
            coffee_loss_file = str(Path(sys.modules[criterion.one2many.__class__.__module__].__file__).resolve()) if hasattr(criterion, "one2many") else str(Path(sys.modules[criterion.__class__.__module__].__file__).resolve())
            if str(PACKAGE_ROOT / "loss") not in coffee_loss_file:
                raise RuntimeError(f"Coffee loss import escaped experiment package: {coffee_loss_file}")
            report["imports"]["coffee_loss"] = coffee_loss_file
            report["criterion"] = {"class": type(criterion).__name__, "branches": [type(criterion.one2many).__name__, type(criterion.one2one).__name__] if hasattr(criterion, "one2many") else []}
            if args.dry_run:
                model.train()
                preds = model(torch.zeros(1, 3, imgsz[0], imgsz[1]))
                total, items = criterion(preds, _synthetic_batch(data["nc"], next(model.parameters()).device))
                if not torch.isfinite(total).all() or not torch.isfinite(items).all():
                    raise RuntimeError("CoffeeLoss dry-run produced non-finite values.")
                report["criterion"]["loss_items"] = [float(value) for value in items.detach().cpu().reshape(-1)]
        # Persist the running state before entering the native trainer so a process-level
        # failure still leaves an auditable manifest instead of an unexplained directory.
        _write_json(output / "run_manifest.json", report)
        if args.dry_run:
            report["status"] = "dry_run_passed"
        else:
            from ultralytics import YOLO  # noqa: PLC0415
            from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: PLC0415

            prebuilt_model = model
            bgmix_cfg = runtime_profile.get("bgmix", {})
            bgmix_p = 0.0 if resolved["bgmix_request"] == "off" else float(resolved["bgmix_request"])

            class CoffeeDetectionTrainer(DetectionTrainer):
                def __init__(self, *trainer_args, **trainer_kwargs):
                    super().__init__(*trainer_args, **trainer_kwargs)
                    self.batch_bgmix = BatchCoffeeBgMix(
                        p=bgmix_p,
                        box_expand=float(bgmix_cfg.get("box_expand", 0.02)),
                        feather_px=int(bgmix_cfg.get("feather_px", 16)),
                        blur_kernel=int(bgmix_cfg.get("blur_kernel", 21)),
                        no_box=str(bgmix_cfg.get("no_box", "skip")),
                    ) if bgmix_p else None

                def get_model(self, cfg=None, weights=None, verbose=True):
                    prebuilt_model.args = self.args
                    prebuilt_model.coffee_loss_name = args.loss
                    prebuilt_model.coffee_profile = runtime_profile
                    prebuilt_model.coffee_loss_config = loss_config
                    return prebuilt_model

                def preprocess_batch(self, batch):
                    cpu_batch_idx = batch.get("batch_idx")
                    cpu_bboxes = batch.get("bboxes")
                    batch = super().preprocess_batch(batch)
                    close_epoch = int(getattr(self.args, "coffee_bgmix_close_epoch", 0))
                    active = not close_epoch or self.epoch + 1 <= close_epoch
                    if self.batch_bgmix is not None and active:
                        batch["img"] = self.batch_bgmix(batch["img"], cpu_batch_idx, cpu_bboxes)
                    return batch

                def validate(self):
                    previous_best = self.best_fitness
                    metrics, native_fitness = super().validate()
                    if metrics is None:
                        return None, None
                    if args.fitness_metric == "map50-95":
                        return metrics, native_fitness
                    fitness = _select_fitness(metrics, args.fitness_metric)
                    self.best_fitness = fitness if previous_best is None else max(float(previous_best), fitness)
                    return metrics, fitness

            runner = YOLO(str(model_cfg), task="detect")
            runner.model = model
            runner.overrides["model"] = str(model_cfg)
            validation_epochs: set[int] = set()

            def record_validation(trainer) -> None:
                epoch = trainer.epoch + 1
                if epoch in validation_epochs:
                    return
                validation_epochs.add(epoch)
                _record_validation_diagnostics(output, trainer, data["names"], runtime_profile.get("pair_ids", []), epoch)

            runner.add_callback("on_train_start", lambda trainer: _reset_module_monitors(getattr(trainer.model, "module", trainer.model)))
            runner.add_callback("on_train_epoch_end", lambda trainer: (_record_training_diagnostics(output, trainer, data["names"], trainer.epoch + 1), _record_bgmix_stats(output, trainer, trainer.epoch + 1)))
            runner.add_callback("on_fit_epoch_end", record_validation)
            results = runner.train(
                trainer=CoffeeDetectionTrainer,
                data=str(runtime_data_yaml),
                pretrained=False,
                project=str(output.parent),
                name=output.name,
                exist_ok=True,
                coffee_bgmix=bgmix_p,
                coffee_bgmix_mode="batch_gpu",
                coffee_bgmix_close_epoch=int(args.bgmix_close_epoch),
                coffee_bgmix_box_expand=float(bgmix_cfg.get("box_expand", 0.02)),
                coffee_bgmix_feather_px=int(bgmix_cfg.get("feather_px", 16)),
                coffee_bgmix_blur_kernel=int(bgmix_cfg.get("blur_kernel", 21)),
                coffee_bgmix_no_box=str(bgmix_cfg.get("no_box", "skip")),
                **training,
            )
            report["status"] = "completed"
            report["results"] = str(results)
        report["ended_at"] = _utc_now()
        _write_json(output / "run_manifest.json", report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["ended_at"] = _utc_now()
        _write_json(output / "run_manifest.json", report)
        raise


if __name__ == "__main__":
    namespace = _parser().parse_args()
    result = run(namespace)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
