"""Frozen foundation encoders. None of these are ever trained.

Star-topology anchor (CLIP) + text foundation (GTE) + vision foundation (V-JEPA-2).
All three use non-autoregressive pretraining objectives.
"""
from .gte import FrozenGTE, GTE_NATIVE_DIM
from .clip import FrozenCLIP, CLIP_NATIVE_DIM
from .vjepa2 import FrozenVJEPA2, VJEPA2_NATIVE_DIM

__all__ = [
    "FrozenGTE", "GTE_NATIVE_DIM",
    "FrozenCLIP", "CLIP_NATIVE_DIM",
    "FrozenVJEPA2", "VJEPA2_NATIVE_DIM",
]
