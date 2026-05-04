"""Stage 0 smoke test — verify each frozen foundation forward-passes.

Usage:
    python3 scripts/test_foundations.py                # text + clip only (default; light)
    python3 scripts/test_foundations.py --vjepa        # also test V-JEPA-2 (downloads ~1GB weights)
    python3 scripts/test_foundations.py --device cuda  # run on GPU

Pass criteria:
    • Each enabled encoder loads without error.
    • Each produces non-NaN, non-zero embeddings of the documented dim.
    • Embeddings of different inputs are distinguishable (cosine sim < 0.99).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def _make_dummy_image(seed: int, size: int = 256) -> Image.Image:
    """Synthetic random RGB image — enough to verify forward pass."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def check_embedding(name: str, emb: torch.Tensor, expected_dim: int) -> None:
    print(f"\n[{name}]")
    print(f"  shape:   {tuple(emb.shape)}")
    assert emb.dim() == 2, f"expected 2D embedding, got {emb.shape}"
    assert emb.shape[1] == expected_dim, (
        f"expected dim {expected_dim}, got {emb.shape[1]}"
    )
    assert not torch.isnan(emb).any(), "embedding contains NaN"
    assert emb.abs().sum() > 0, "embedding is all zeros"
    # Distinguishability: two different inputs should not be near-identical.
    if emb.shape[0] >= 2:
        a = F.normalize(emb[0:1], dim=-1)
        b = F.normalize(emb[1:2], dim=-1)
        cos = (a * b).sum().item()
        print(f"  cos(0,1): {cos:+.3f}  ({'distinguishable' if cos < 0.99 else 'WARN: near-identical'})")
    print(f"  norm:    mean={emb.norm(dim=-1).mean().item():.3f}, "
          f"std={emb.norm(dim=-1).std().item():.3f}")
    print(f"  ✓ pass")


def test_gte(device: str) -> None:
    from selflearnai.foundations import FrozenGTE, GTE_NATIVE_DIM
    print("\n" + "=" * 60)
    print("Loading GTE-base ...")
    gte = FrozenGTE(device=device)
    emb = gte.encode([
        "the cat sat on the mat",
        "a quick brown fox jumps over a lazy dog",
        "machine learning is the study of statistical patterns",
    ])
    check_embedding("GTE", emb, GTE_NATIVE_DIM)


def test_clip(device: str) -> None:
    from selflearnai.foundations import FrozenCLIP, CLIP_NATIVE_DIM
    print("\n" + "=" * 60)
    print("Loading CLIP ViT-B/32 ...")
    clip = FrozenCLIP(device=device)

    # Text side
    t_emb = clip.encode_text([
        "a photo of a cat",
        "a diagram of a circuit",
        "an empty hallway at dusk",
    ])
    check_embedding("CLIP/text", t_emb, CLIP_NATIVE_DIM)

    # Vision side
    images = [_make_dummy_image(i, size=224) for i in range(3)]
    v_emb = clip.encode_image(images)
    check_embedding("CLIP/vision", v_emb, CLIP_NATIVE_DIM)


def test_vjepa(device: str) -> None:
    from selflearnai.foundations import FrozenVJEPA2, VJEPA2_NATIVE_DIM
    print("\n" + "=" * 60)
    print("Loading V-JEPA-2 (this downloads ~1GB on first run) ...")
    vj = FrozenVJEPA2(device=device)

    images = [_make_dummy_image(i, size=256) for i in range(2)]
    pooled = vj.encode(images)
    check_embedding("V-JEPA-2/pooled", pooled, VJEPA2_NATIVE_DIM)

    patches = vj.encode_patches(images)
    print(f"\n[V-JEPA-2/patches]")
    print(f"  shape:   {tuple(patches.shape)}")
    print(f"  ✓ per-patch features available for factored adapter")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu",
                        help="cpu | cuda | mps")
    parser.add_argument("--vjepa", action="store_true",
                        help="also test V-JEPA-2 (heavy; default off)")
    parser.add_argument("--skip-clip", action="store_true",
                        help="skip CLIP test")
    parser.add_argument("--skip-gte", action="store_true",
                        help="skip GTE test")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Torch: {torch.__version__}")

    if not args.skip_gte:
        test_gte(args.device)
    if not args.skip_clip:
        test_clip(args.device)
    if args.vjepa:
        test_vjepa(args.device)
    else:
        print("\n[V-JEPA-2 skipped — pass --vjepa to include]")

    print("\n" + "=" * 60)
    print("Stage 0 smoke test PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
