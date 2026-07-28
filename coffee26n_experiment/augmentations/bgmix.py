"""BgMix: transform only pixels outside the union of expanded GT boxes."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _preserve_mask(height: int, width: int, boxes: torch.Tensor, expand: float, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    if boxes.numel() == 0:
        return mask
    dx, dy = int(round(float(expand) * width)), int(round(float(expand) * height))
    for box in boxes.reshape(-1, 4):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        left = max(0, math.floor(x1) - dx)
        top = max(0, math.floor(y1) - dy)
        right = min(width, math.ceil(x2) + dx)
        bottom = min(height, math.ceil(y2) + dy)
        if right > left and bottom > top:
            mask[top:bottom, left:right] = True
    return mask


def _gaussian_blur(image: torch.Tensor, kernel: int) -> torch.Tensor:
    sigma = max(float(kernel) / 6.0, 0.5)
    coordinates = torch.arange(kernel, device=image.device, dtype=image.dtype) - kernel // 2
    weights = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    weights = weights / weights.sum()
    channels = image.shape[0]
    horizontal = weights.view(1, 1, 1, kernel).expand(channels, 1, 1, kernel)
    vertical = weights.view(1, 1, kernel, 1).expand(channels, 1, kernel, 1)
    pad = kernel // 2
    value = F.pad(image.unsqueeze(0), (pad, pad, 0, 0), mode="replicate")
    value = F.conv2d(value, horizontal, groups=channels)
    value = F.pad(value, (0, 0, pad, pad), mode="replicate")
    return F.conv2d(value, vertical, groups=channels).squeeze(0)


def apply_bgmix(image: torch.Tensor, boxes_xyxy: torch.Tensor, *, p: float = 0.25, box_expand: float = 0.02,
                feather_px: int = 16, blur_kernel: int = 21, training: bool = True,
                force: bool = False, generator: torch.Generator | None = None,
                transform: str | None = None) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    """Apply one allowed external transform outside the preserve mask.

    The returned binary mask is suitable for a bitwise-preservation assertion. ``feather_px`` is accepted as part of
    the profile contract; the blend remains exactly one inside the binary preserve region.
    """
    if image.ndim != 3:
        raise ValueError(f"BgMix expects CHW image tensor, got {tuple(image.shape)}.")
    if not training:
        return image, {"status": "skipped_not_training", "transform": None, "preserve_ratio": 0.0}, torch.zeros(image.shape[-2:], dtype=torch.bool, device=image.device)
    boxes_xyxy = torch.as_tensor(boxes_xyxy, device=image.device, dtype=torch.float32).reshape(-1, 4)
    preserve = _preserve_mask(image.shape[-2], image.shape[-1], boxes_xyxy, float(box_expand), image.device)
    if boxes_xyxy.numel() == 0:
        return image, {"status": "skipped_no_boxes", "transform": None, "preserve_ratio": 0.0}, preserve
    random_value = torch.rand((), generator=generator, device=image.device).item() if generator is not None else torch.rand(()).item()
    if not force and random_value >= float(p):
        return image, {"status": "skipped_probability", "transform": None, "preserve_ratio": float(preserve.float().mean())}, preserve
    kernel = int(blur_kernel)
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError(f"BgMix blur_kernel must be odd and >=3, got {blur_kernel}.")
    allowed = ("blur", "desaturate", "contrast")
    if transform is None:
        transform = allowed[int(torch.randint(len(allowed), (), generator=generator).item())]
    if transform not in allowed:
        raise ValueError(f"BgMix transform must be one of {allowed}, got {transform!r}.")
    if transform == "blur":
        transformed = _gaussian_blur(image, kernel)
    elif transform == "desaturate":
        gray = image.mean(0, keepdim=True)
        transformed = image * 0.75 + gray * 0.25
    else:
        mean = image.mean(dim=(-2, -1), keepdim=True)
        transformed = mean + (image - mean) * 0.85
    outside = (~preserve).to(dtype=image.dtype).unsqueeze(0).unsqueeze(0)
    feather = max(int(feather_px), 0)
    if feather:
        outside = F.avg_pool2d(F.pad(outside, (feather,) * 4, mode="replicate"), 2 * feather + 1, stride=1)
        outside[..., preserve] = 0.0
    blend = outside.squeeze(0)
    output = image * (1.0 - blend) + transformed * blend
    if not torch.equal(output[:, preserve], image[:, preserve]):
        raise RuntimeError("BgMix preserve mask was modified; refusing to continue.")
    return output, {
        "status": "applied",
        "transform": transform,
        "preserve_ratio": float(preserve.float().mean()),
        "feather_px": int(feather_px),
    }, preserve


class BatchCoffeeBgMix:
    """Apply the same box-preserving background transform to a GPU batch.

    Decisions are sampled on CPU so deterministic training remains reproducible and no GPU-to-CPU
    synchronization is needed inside the batch loop. Images are expected to be BCHW float tensors in [0, 1],
    while boxes remain CPU tensors in normalized xywh format from the YOLO dataloader.
    """

    def __init__(self, p: float, box_expand: float, feather_px: int, blur_kernel: int, no_box: str = "skip"):
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"coffee_bgmix must be in [0,1], got {p!r}.")
        if no_box not in {"skip", "fail"}:
            raise ValueError(f"coffee_bgmix_no_box must be skip or fail, got {no_box!r}.")
        if int(blur_kernel) < 3 or int(blur_kernel) % 2 == 0:
            raise ValueError(f"coffee_bgmix_blur_kernel must be odd and >=3, got {blur_kernel!r}.")
        self.p = float(p)
        self.box_expand = float(box_expand)
        self.feather_px = int(feather_px)
        self.blur_kernel = int(blur_kernel)
        self.no_box = no_box
        self.reset()

    def reset(self) -> None:
        self._applied = 0
        self._skipped_no_boxes = 0
        self._skipped_probability = 0
        self._preserve_sum: torch.Tensor | None = None
        self._transform_counts = [0, 0, 0]

    @property
    def stats(self) -> dict[str, Any]:
        total = max(self._applied, 1)
        names = ("blur", "desaturate", "contrast")
        last = max(range(3), key=lambda index: self._transform_counts[index])
        return {
            "applied": self._applied,
            "skipped_no_boxes": self._skipped_no_boxes,
            "skipped_probability": self._skipped_probability,
            "preserve_ratio": float(self._preserve_sum.item()) / total if self._preserve_sum is not None else 0.0,
            "last_transform": names[last] if any(self._transform_counts) else None,
        }

    @staticmethod
    def _gaussian_blur_batch(images: torch.Tensor, kernel: int) -> torch.Tensor:
        sigma = max(float(kernel) / 6.0, 0.5)
        coordinates = torch.arange(kernel, device=images.device, dtype=images.dtype) - kernel // 2
        weights = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
        weights = weights / weights.sum()
        channels = images.shape[1]
        horizontal = weights.view(1, 1, 1, kernel).expand(channels, 1, 1, kernel)
        vertical = weights.view(1, 1, kernel, 1).expand(channels, 1, kernel, 1)
        pad = kernel // 2
        value = F.pad(images, (pad, pad, 0, 0), mode="replicate")
        value = F.conv2d(value, horizontal, groups=channels)
        value = F.pad(value, (0, 0, pad, pad), mode="replicate")
        return F.conv2d(value, vertical, groups=channels)

    def __call__(self, images: torch.Tensor, batch_idx: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"BatchCoffeeBgMix expects BCHW RGB images, got {tuple(images.shape)}.")
        if not self.p or not images.shape[0]:
            return images
        if batch_idx is None or boxes is None:
            raise ValueError("BatchCoffeeBgMix requires batch_idx and bboxes from the YOLO batch.")
        batch_size, _, height, width = images.shape
        batch_idx = batch_idx.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        boxes = boxes.detach().to(device="cpu", dtype=torch.float32).reshape(-1, 4)
        counts = torch.bincount(batch_idx, minlength=batch_size) if batch_idx.numel() else torch.zeros(batch_size, dtype=torch.long)
        has_boxes = counts > 0
        if self.no_box == "fail" and (~has_boxes).any():
            raise ValueError("BatchCoffeeBgMix received an empty-label training image and no_box=fail.")
        random_values = torch.rand(batch_size)
        eligible = has_boxes & (random_values < self.p)
        apply_ids = eligible.nonzero(as_tuple=False).flatten().tolist()
        self._skipped_no_boxes += int((~has_boxes).sum())
        self._skipped_probability += int((has_boxes & ~eligible).sum())
        if not apply_ids:
            return images

        position = {image_id: offset for offset, image_id in enumerate(apply_ids)}
        preserve = torch.zeros((len(apply_ids), 1, height, width), dtype=torch.bool, device=images.device)
        for box, image_id in zip(boxes, batch_idx.tolist()):
            if image_id not in position:
                continue
            cx, cy, bw, bh = box.tolist()
            cx, bw = cx * width, bw * width
            cy, bh = cy * height, bh * height
            dx, dy = round(self.box_expand * width), round(self.box_expand * height)
            left = max(0, math.floor(cx - bw / 2) - dx)
            top = max(0, math.floor(cy - bh / 2) - dy)
            right = min(width, math.ceil(cx + bw / 2) + dx)
            bottom = min(height, math.ceil(cy + bh / 2) + dy)
            if right > left and bottom > top:
                preserve[position[image_id], :, top:bottom, left:right] = True

        index = torch.tensor(apply_ids, dtype=torch.long, device=images.device)
        selected = images.index_select(0, index)
        transformed = selected.clone()
        transform_ids = torch.randint(3, (len(apply_ids),))
        for transform_id in range(3):
            group = (transform_ids == transform_id).nonzero(as_tuple=False).flatten()
            if not group.numel():
                continue
            group_gpu = group.to(device=images.device)
            group_images = selected.index_select(0, group_gpu)
            if transform_id == 0:
                group_images = self._gaussian_blur_batch(group_images, self.blur_kernel)
            elif transform_id == 1:
                group_images = group_images * 0.75 + group_images.mean(1, keepdim=True) * 0.25
            else:
                mean = group_images.mean(dim=(-2, -1), keepdim=True)
                group_images = mean + (group_images - mean) * 0.85
            # AMP convolution may return float16/bfloat16 even when the preprocessed batch is float32.
            transformed.index_copy_(0, group_gpu, group_images.to(dtype=transformed.dtype))
            self._transform_counts[transform_id] += int(group.numel())

        outside = (~preserve).to(dtype=selected.dtype)
        feather = max(self.feather_px, 0)
        if feather:
            outside = F.avg_pool2d(F.pad(outside, (feather,) * 4, mode="replicate"), 2 * feather + 1, stride=1)
            outside = outside.masked_fill(preserve, 0.0)
        output = selected * (1.0 - outside) + transformed * outside
        images.index_copy_(0, index, output.to(dtype=images.dtype))
        self._applied += len(apply_ids)
        preserve_sum = preserve.sum(dtype=torch.float32).detach() / float(height * width)
        self._preserve_sum = preserve_sum if self._preserve_sum is None else self._preserve_sum + preserve_sum
        return images
