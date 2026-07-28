"""Training-only background augmentation."""

from .bgmix import BatchCoffeeBgMix, apply_bgmix

__all__ = ["BatchCoffeeBgMix", "apply_bgmix"]
