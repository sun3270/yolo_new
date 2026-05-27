"""Train YOLO26n-LGMSF-Lite on a PC using the local experiment package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_ROOT.parent
LOCAL_PARENT = EXP_ROOT / "local_ultralytics"
CFG = EXP_ROOT / "configs" / "yolo26n_lgmsf_lite.yaml"
DEFAULT_DATA = PROJECT_ROOT / "coffee3000" / "coffee3000.yaml"
REPORT = EXP_ROOT / "reports" / "training_command_report.md"


def parse_batch(value: str):
    return value if value.lower() == "auto" else int(value)


def git_commit() -> str:
    candidates = [
        "git",
        r"E:\Git\bin\git.exe",
        r"E:\Git\cmd\git.exe",
    ]
    for exe in candidates:
        try:
            return subprocess.check_output([exe, "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        except Exception:
            continue
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=parse_batch, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(EXP_ROOT / "runs_lgmsf_pc"))
    parser.add_argument("--name", default="yolo26n_lgmsf_lite")
    parser.add_argument("--pretrained", default="", help="Optional pretrained weights, e.g. yolo26n.pt.")
    return parser.parse_args()


def write_training_report(args: argparse.Namespace) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# LGMSF-Lite Training Command Report",
        "",
        f"- Time: `{now}`",
        f"- Config YAML: `{CFG}`",
        f"- Data YAML: `{args.data}`",
        f"- Epochs: `{args.epochs}`",
        f"- Image size: `{args.imgsz}`",
        f"- Batch: `{args.batch}`",
        f"- Device: `{args.device}`",
        f"- Workers: `{args.workers}`",
        f"- Project: `{args.project}`",
        f"- Name: `{args.name}`",
        f"- Pretrained: `{args.pretrained or 'none'}`",
        f"- Git commit: `{git_commit()}`",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_training_report(args)
    sys.path.insert(0, str(LOCAL_PARENT))

    from ultralytics import YOLO

    model = YOLO(str(CFG))
    if args.pretrained:
        model.load(args.pretrained)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
