"""Patch the local Ultralytics copy with LGMSF-Lite modules."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
LOCAL_PKG = EXP_ROOT / "local_ultralytics" / "ultralytics"
PATCHES = EXP_ROOT / "patches"
REPORT = PATCHES / "patch_report.md"
BLOCK = LOCAL_PKG / "nn" / "modules" / "block.py"
MODULES_INIT = LOCAL_PKG / "nn" / "modules" / "__init__.py"
TASKS = LOCAL_PKG / "nn" / "tasks.py"

MODULE_NAMES = ["LDSConv", "LGMSFBridge", "TextureBranch", "SemanticBranch", "SimAM", "FastNormFuse2"]

LGMSF_BLOCK = r'''

# ---- LGMSF-Lite modules: begin ----
class ConvBNAct(nn.Module):
    """Small Conv-BN-activation helper used by LGMSF-Lite."""

    def __init__(self, c1, c2, k=1, s=1, p=None, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWSeparableConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw = ConvBNAct(c1, c1, k=k, s=s, groups=c1, act=act)
        self.pw = ConvBNAct(c1, c2, k=1, s=1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class LDSConv(nn.Module):
    """Learnable depthwise-separable downsample convolution."""

    def __init__(self, c1, c2, k=3, s=2, act=True):
        super().__init__()
        self.conv = DWSeparableConv(c1, c2, k=k, s=s, act=act)

    def forward(self, x):
        return self.conv(x)


class GhostConvLite(nn.Module):
    """Lightweight Ghost-style convolution without external dependencies."""

    def __init__(self, c1, c2, k=1, s=1, ratio=2, act=True):
        super().__init__()
        c_primary = max(1, int((c2 + ratio - 1) // ratio))
        c_cheap = c2 - c_primary
        self.primary = ConvBNAct(c1, c_primary, k=k, s=s, act=act)
        self.cheap = ConvBNAct(c_primary, c_cheap, k=3, s=1, groups=c_primary, act=act) if c_cheap > 0 else None
        self.c2 = c2

    def forward(self, x):
        y = self.primary(x)
        if self.cheap is None:
            return y
        return torch.cat((y, self.cheap(y)), dim=1)[:, : self.c2]


class TextureBranch(nn.Module):
    """P3 texture stream for small disease spots and edge detail."""

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()
        self.conv1 = DWSeparableConv(c2, c2, k=3, s=1)
        self.conv2 = DWSeparableConv(c2, c2, k=3, s=1)

    def forward(self, x):
        x = self.proj(x)
        return x + self.conv2(self.conv1(x))


class SemanticBranch(nn.Module):
    """P5 semantic stream with low-cost receptive field expansion."""

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = ConvBNAct(c1, c2, k=1, s=1) if c1 != c2 else nn.Identity()
        self.conv = nn.Sequential(
            GhostConvLite(c2, c2, k=1, s=1),
            ConvBNAct(c2, c2, k=5, s=1, groups=c2),
            GhostConvLite(c2, c2, k=1, s=1),
        )

    def forward(self, x):
        x = self.proj(x)
        return x + self.conv(x)


class FastNormFuse2(nn.Module):
    """BiFPN-style fast normalized weighted fusion for two inputs."""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = eps

    def forward(self, x1, x2):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return w[0] * x1 + w[1] * x2


class SimAM(nn.Module):
    """Parameter-free SimAM attention."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        _b, _c, h, w = x.size()
        n = h * w - 1
        if n <= 0:
            return x
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * torch.sigmoid(y)


class LGMSFBridge(nn.Module):
    """Fuse P3 texture and P5 semantic features into an enhanced P5 feature."""

    def __init__(self, c3, c5, c_out, use_simam=True):
        super().__init__()
        self.texture = TextureBranch(c3, c_out)
        self.texture_down = nn.Sequential(
            LDSConv(c_out, c_out, k=3, s=2),
            LDSConv(c_out, c_out, k=3, s=2),
        )
        self.semantic = SemanticBranch(c5, c_out)
        self.fuse = FastNormFuse2()
        self.attn = SimAM() if use_simam else nn.Identity()

    def forward(self, xs):
        p3, p5 = xs
        texture = self.texture_down(self.texture(p3))
        semantic = self.semantic(p5)
        if texture.shape[-2:] != semantic.shape[-2:]:
            texture = F.interpolate(texture, size=semantic.shape[-2:], mode="nearest")
        return self.attn(self.fuse(texture, semantic))
# ---- LGMSF-Lite modules: end ----
'''


def require_files() -> None:
    missing = [path for path in (BLOCK, MODULES_INIT, TASKS) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing local Ultralytics files: " + ", ".join(str(p) for p in missing))


def backup_files() -> None:
    PATCHES.mkdir(parents=True, exist_ok=True)
    backups = {
        BLOCK: PATCHES / "block.py.before_lgmsf.bak",
        MODULES_INIT: PATCHES / "__init__.py.before_lgmsf.bak",
        TASKS: PATCHES / "tasks.py.before_lgmsf.bak",
    }
    for src, dst in backups.items():
        if not dst.exists():
            shutil.copy2(src, dst)


def append_block_modules() -> bool:
    text = BLOCK.read_text(encoding="utf-8", errors="ignore")
    if "class LGMSFBridge" in text:
        return False
    if "__all__ = (" in text:
        text = add_to_string_tuple(
            text, "__all__ = (", ["ConvBNAct", "DWSeparableConv", "GhostConvLite", *MODULE_NAMES]
        )
    BLOCK.write_text(text.rstrip() + LGMSF_BLOCK + "\n", encoding="utf-8")
    (PATCHES / "lgmsf_modules_block_append.py").write_text(LGMSF_BLOCK.lstrip(), encoding="utf-8")
    return True


def add_to_parenthesized_import(text: str, start_marker: str, names: list[str]) -> str:
    start = text.find(start_marker)
    if start == -1:
        raise RuntimeError(f"Could not find import marker: {start_marker}")
    end = text.find("\n)", start)
    if end == -1:
        raise RuntimeError(f"Could not find import close for: {start_marker}")
    block = text[start:end]
    missing = [name for name in names if f"    {name}," not in block and f"{name}," not in block]
    if not missing:
        return text
    insert = "".join(f"    {name},\n" for name in missing)
    return text[:end] + "\n" + insert.rstrip("\n") + text[end:]


def add_to_string_tuple(text: str, start_marker: str, names: list[str]) -> str:
    start = text.find(start_marker)
    if start == -1:
        raise RuntimeError(f"Could not find tuple marker: {start_marker}")
    end = text.find("\n)", start)
    if end == -1:
        raise RuntimeError(f"Could not find tuple close for: {start_marker}")
    block = text[start:end]
    missing = [name for name in names if f'"{name}",' not in block and f"'{name}'," not in block]
    if not missing:
        return text
    insert = "".join(f'    "{name}",\n' for name in missing)
    return text[:end] + "\n" + insert.rstrip("\n") + text[end:]


def patch_modules_init() -> bool:
    original = MODULES_INIT.read_text(encoding="utf-8", errors="ignore")
    text = add_to_parenthesized_import(original, "from .block import (", MODULE_NAMES)
    if "__all__ = (" in text:
        text = add_to_string_tuple(text, "__all__ = (", MODULE_NAMES)
    MODULES_INIT.write_text(text, encoding="utf-8")
    return text != original


def patch_tasks_import(text: str) -> str:
    return add_to_parenthesized_import(text, "from ultralytics.nn.modules import (", ["LDSConv", "LGMSFBridge"])


def patch_tasks_base_modules(text: str) -> str:
    if "            LDSConv,\n" in text:
        return text
    marker = "            DWConv,\n"
    if marker not in text:
        raise RuntimeError("Could not find DWConv entry in base_modules.")
    return text.replace(marker, marker + "            LDSConv,\n", 1)


def patch_tasks_lgmsf_bridge(text: str) -> str:
    if "elif m is LGMSFBridge:" in text:
        return text
    marker = "        elif m is AIFI:\n"
    if marker not in text:
        raise RuntimeError("Could not find parse_model AIFI branch insertion point.")
    block = (
        "        elif m is LGMSFBridge:\n"
        "            if not isinstance(f, list) or len(f) != 2:\n"
        "                raise ValueError('LGMSFBridge expects from=[P3_INDEX, P5_INDEX].')\n"
        "            c3, c5 = ch[f[0]], ch[f[1]]\n"
        "            c2 = args[0] if len(args) > 0 else c5\n"
        "            args = [c3, c5, c2, *args[1:]]\n"
    )
    return text.replace(marker, block + marker, 1)


def patch_tasks() -> bool:
    original = TASKS.read_text(encoding="utf-8", errors="ignore")
    text = patch_tasks_import(original)
    text = patch_tasks_base_modules(text)
    text = patch_tasks_lgmsf_bridge(text)
    TASKS.write_text(text, encoding="utf-8")
    (PATCHES / "tasks_parse_model_notes.md").write_text(
        "# tasks.py LGMSF-Lite Notes\n\n"
        "- Imported `LDSConv` and `LGMSFBridge` from `ultralytics.nn.modules`.\n"
        "- Added `LDSConv` to `base_modules` so it receives standard `c1, c2` parsing.\n"
        "- Added a dedicated `LGMSFBridge` parse branch for two-input channel handling.\n",
        encoding="utf-8",
    )
    return text != original


def write_report(changed: dict[str, bool]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# LGMSF-Lite Patch Report",
        "",
        f"- Time: `{now}`",
        f"- Local package: `{LOCAL_PKG}`",
        "- Original project package was not modified by this script.",
        "",
        "## Files",
        "",
    ]
    for name, did_change in changed.items():
        status = "patched this run" if did_change else "already present; no rewrite needed"
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(
        [
            "",
            "## Added Modules",
            "",
            "- `ConvBNAct`",
            "- `DWSeparableConv`",
            "- `LDSConv`",
            "- `GhostConvLite`",
            "- `TextureBranch`",
            "- `SemanticBranch`",
            "- `FastNormFuse2`",
            "- `SimAM`",
            "- `LGMSFBridge`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clear_local_pycache() -> None:
    for path in LOCAL_PKG.rglob("__pycache__"):
        shutil.rmtree(path)


def main() -> None:
    require_files()
    backup_files()
    changed = {
        "nn/modules/block.py": append_block_modules(),
        "nn/modules/__init__.py": patch_modules_init(),
        "nn/tasks.py": patch_tasks(),
    }
    clear_local_pycache()
    write_report(changed)
    print(f"Patched local Ultralytics package: {LOCAL_PKG}")
    print(f"Patch report: {REPORT}")


if __name__ == "__main__":
    main()
