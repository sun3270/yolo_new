"""Run selected coffee V4 ablation experiments as isolated subprocesses."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "coffee_v4_training" / "train.py"
sys.path.insert(0, str(ROOT))

from coffee_v4_training.train import (  # noqa: E402
    CORE_EXPERIMENTS,
    ELTEB_CONTROL_EXPERIMENTS,
    EXPERIMENTS,
    MAIN_EXPERIMENTS,
    RECALL_EXPERIMENTS,
    STRONG_EXPERIMENTS,
)


GROUPS = {
    "core": CORE_EXPERIMENTS,
    "main": MAIN_EXPERIMENTS,
    "elteb": ELTEB_CONTROL_EXPERIMENTS,
    "recall": RECALL_EXPERIMENTS,
    "strong": STRONG_EXPERIMENTS,
    "full": tuple(EXPERIMENTS),
}


def parse_exps(args: argparse.Namespace) -> list[str]:
    if args.exps:
        exps = [x.strip() for x in args.exps.split(",") if x.strip()]
    else:
        exps = list(GROUPS[args.group])
    unknown = [x for x in exps if x not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiment ids: {unknown}. Valid ids: {sorted(EXPERIMENTS)}")
    return exps


def append_common_args(command: list[str], args: argparse.Namespace) -> None:
    options = (
        ("--data", args.data),
        ("--weights", args.weights),
        ("--project", args.project),
        ("--device", args.device),
        ("--epochs", args.epochs),
        ("--imgsz", args.imgsz),
        ("--batch", args.batch),
        ("--workers", args.workers),
        ("--cache", args.cache),
        ("--preprocess-output", args.preprocess_output),
        ("--path-remap", args.path_remap),
    )
    for flag, value in options:
        if value is not None:
            command.extend([flag, str(value)])
    if args.no_weights:
        command.append("--no-weights")
    if args.overwrite_preprocess:
        command.append("--overwrite-preprocess")
    if args.dry_run:
        command.append("--dry-run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--group", choices=sorted(GROUPS), default="core", help="Experiment group to run.")
    parser.add_argument("--exps", default=None, help="Comma-separated explicit experiment ids.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running after a failed experiment.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without training.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument("--project", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--preprocess-output", default=None)
    parser.add_argument("--overwrite-preprocess", action="store_true")
    parser.add_argument("--path-remap", default=None)
    args = parser.parse_args()

    failures: list[tuple[str, int]] = []
    for exp_id in parse_exps(args):
        command = [sys.executable, str(TRAIN_SCRIPT), "--exp", exp_id]
        append_common_args(command, args)
        print("\n" + "=" * 88)
        print(f"Running: {exp_id}")
        print("Command:", " ".join(command))
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            failures.append((exp_id, result.returncode))
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    if failures:
        print("\nFailed experiments:")
        for exp_id, code in failures:
            print(f"  {exp_id}: exit code {code}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

