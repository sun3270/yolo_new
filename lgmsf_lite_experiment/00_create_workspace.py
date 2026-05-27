"""Create the isolated LGMSF-Lite experiment workspace."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRS = (
    "configs",
    "patches",
    "reports",
    "local_ultralytics",
    "_sources",
)
README = ROOT / "README_LGMSF_Lite_PC_Training.md"
LOG = ROOT / "reports" / "workspace_log.md"


README_TEXT = """# LGMSF-Lite PC Training Workspace

This directory is an isolated experiment workspace for YOLO26n-LGMSF-Lite.

Core route:

```text
LGMSF-Lite = LDSConv downsample + P3 texture stream + P5 semantic stream + FastNormFuse2 + SimAM
```

Rules:

- Do not modify the project root `ultralytics/` package.
- Do not modify the original YOLO26 YAML.
- Modify only the copied package under `local_ultralytics/`.
- Keep reports under `reports/` and patch notes under `patches/`.
- This stage is PC training only: no deployment, export, quantization, or ablation scripts.

Default dataset:

```text
coffee3000/coffee3000.yaml
```
"""


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    created = []
    for rel in DIRS:
        path = ROOT / rel
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        created.append((rel, existed))

    if not README.exists():
        README.write_text(README_TEXT, encoding="utf-8")

    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        f"## {now}",
        "",
        f"- Workspace: `{ROOT}`",
        f"- README existed before run: `{README.exists()}`",
    ]
    for rel, existed in created:
        lines.append(f"- `{rel}/`: {'already existed' if existed else 'created'}")
    lines.append("")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Workspace ready: {ROOT}")
    print(f"README: {README}")
    print(f"Log: {LOG}")


if __name__ == "__main__":
    main()
