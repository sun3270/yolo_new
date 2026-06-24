"""Build and forward-check all coffee V4 model YAMLs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coffee_v4_training.train import EXPERIMENTS, setup_import_paths  # noqa: E402


def summarize_output(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [summarize_output(v) for v in value]
    if isinstance(value, dict):
        return {str(k): summarize_output(v) for k, v in value.items()}
    return type(value).__name__


def selected_configs(exp_ids: list[str] | None) -> dict[str, Path]:
    items = EXPERIMENTS.items() if exp_ids is None else [(exp_id, EXPERIMENTS[exp_id]) for exp_id in exp_ids]
    configs: dict[str, Path] = {}
    for exp_id, exp in items:
        configs.setdefault(exp.cfg.stem, exp.cfg)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exps", default=None, help="Comma-separated experiment ids. Defaults to all unique configs.")
    parser.add_argument("--imgsz", type=int, default=320, help="Dummy forward image size.")
    parser.add_argument("--nc", type=int, default=9, help="Number of classes for model construction.")
    args = parser.parse_args()

    setup_import_paths()
    from ultralytics.nn.tasks import DetectionModel  # noqa: PLC0415

    exp_ids = None
    if args.exps:
        exp_ids = [x.strip() for x in args.exps.split(",") if x.strip()]
        unknown = [x for x in exp_ids if x not in EXPERIMENTS]
        if unknown:
            raise ValueError(f"Unknown experiment ids: {unknown}. Valid ids: {sorted(EXPERIMENTS)}")

    for name, cfg in selected_configs(exp_ids).items():
        if not cfg.exists():
            raise FileNotFoundError(f"Model yaml not found: {cfg}")
        model = DetectionModel(str(cfg), ch=3, nc=args.nc, verbose=False)
        params = sum(p.numel() for p in model.parameters())
        module_names = [m.__class__.__name__ for m in model.modules()]
        detect_from = next((layer.f for layer in model.model if layer.__class__.__name__ == "Detect"), None)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, args.imgsz, args.imgsz))
        print(
            f"{name}: params={params}, DetectFrom={detect_from}, "
            f"EdgeBridge={'EdgeLGMSFBridge' in module_names}, "
            f"P5ToP3={'P5ToP3SemanticFuse' in module_names}, "
            f"ELTEB={'ELTEB' in module_names or 'ELTEBLite' in module_names}, "
            f"output={summarize_output(output)}"
        )


if __name__ == "__main__":
    main()

