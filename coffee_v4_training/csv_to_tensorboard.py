"""Mirror an Ultralytics results.csv file into TensorBoard event scalars.

This is useful for training runs where the project-local Ultralytics fork writes
``results.csv`` but does not create ``events.out.tfevents`` files.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def numeric_or_none(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def write_rows(csv_path: Path, writer: SummaryWriter, seen_epochs: set[int]) -> int:
    if not csv_path.exists():
        return 0

    written = 0
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            epoch_value = numeric_or_none(row.get("epoch", ""))
            if epoch_value is None:
                continue
            epoch = int(epoch_value)
            if epoch in seen_epochs:
                continue

            for key, value in row.items():
                tag = (key or "").strip()
                if not tag or tag == "epoch" or value is None:
                    continue
                metric = numeric_or_none(value)
                if metric is not None:
                    writer.add_scalar(tag, metric, epoch)

            seen_epochs.add(epoch)
            written += 1

    if written:
        writer.flush()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing results.csv.")
    parser.add_argument("--interval", type=float, default=10.0, help="Polling interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Convert current rows and exit.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    csv_path = run_dir / "results.csv"
    seen_epochs: set[int] = set()

    print(f"Watching {csv_path}", flush=True)
    with SummaryWriter(log_dir=str(run_dir)) as writer:
        while True:
            written = write_rows(csv_path, writer, seen_epochs)
            if written:
                print(f"Wrote {written} epoch(s); latest={max(seen_epochs)}", flush=True)
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
