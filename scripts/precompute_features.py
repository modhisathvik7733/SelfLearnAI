"""Pre-compute frozen-foundation features for Stage 1.

Stage 1 was bottlenecked by CPU-side image loading + per-step V-JEPA-2 forward
passes. Both are deterministic functions of the input — running them every
epoch wastes compute.

This script does the work ONCE:
  • Encodes every (text) sample through GTE and CLIP-text.
  • Encodes every UNIQUE image through CLIP-vision and V-JEPA-2.
  • Saves the results to disk.

Stage 1 trainer can then load the cache and skip foundation forward passes
entirely. Each step becomes ~ms (a few tensor slices + tiny adapter MLPs)
instead of ~seconds (PIL decode + V-JEPA-2 ViT-L forward).

Usage:
    python3 scripts/precompute_features.py \\
        --coco-root /workspace/coco \\
        --split val2017 \\
        --max-samples 30000 \\
        --out-dir /workspace/coco/precomputed \\
        --device cuda \\
        --fp16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from selflearnai.foundations import FrozenCLIP, FrozenGTE, FrozenVJEPA2
from selflearnai.grounding import load_coco_samples


def _maybe_fp16(t: torch.Tensor, fp16: bool) -> torch.Tensor:
    return t.half() if fp16 else t


def encode_texts(samples, gte, clip, batch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every sample's caption through GTE and CLIP-text."""
    gte_out, clip_out = [], []
    n = len(samples)
    for i in range(0, n, batch_size):
        texts = [s.text for s in samples[i:i + batch_size]]
        gte_out.append(gte.encode(texts).cpu())
        clip_out.append(clip.encode_text(texts).cpu())
        if (i // batch_size) % 20 == 0:
            print(f"  text {i}/{n}")
    return torch.cat(gte_out, dim=0), torch.cat(clip_out, dim=0)


def encode_images(unique_image_ids, image_paths, clip, vjepa, batch_size: int = 16):
    """Encode each unique image through CLIP-vision and V-JEPA-2.

    Per-image batch size is small (16) because V-JEPA-2 patches are heavy in
    memory: (B, 256_patches, 1024_dim) at fp32 is ~64 MB per image.
    """
    clip_out, vjepa_out = [], []
    n = len(unique_image_ids)
    for i in range(0, n, batch_size):
        chunk_ids = unique_image_ids[i:i + batch_size]
        images = [Image.open(image_paths[iid]).convert("RGB") for iid in chunk_ids]
        clip_out.append(clip.encode_image(images).cpu())
        vjepa_out.append(vjepa.encode_patches(images).cpu())
        if (i // batch_size) % 20 == 0:
            print(f"  image {i}/{n}")
    return torch.cat(clip_out, dim=0), torch.cat(vjepa_out, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-batch", type=int, default=128)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument(
        "--fp16", action="store_true",
        help="store features in fp16 (halves disk + RAM at <1% accuracy cost)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading COCO samples ...")
    samples, image_paths = load_coco_samples(
        args.coco_root, args.split, args.max_samples,
    )
    n_samples = len(samples)
    unique_image_ids = sorted(image_paths.keys())
    image_id_to_idx = {iid: i for i, iid in enumerate(unique_image_ids)}
    sample_to_image_idx = torch.tensor(
        [image_id_to_idx[s.image_id] for s in samples], dtype=torch.long,
    )
    print(f"  {n_samples} samples covering {len(unique_image_ids)} unique images")

    print("\nLoading frozen foundations (will download ~3GB if not cached) ...")
    gte = FrozenGTE(device=args.device)
    clip = FrozenCLIP(device=args.device)
    vjepa = FrozenVJEPA2(device=args.device)

    # ---------- Text features ----------
    print(f"\nEncoding {n_samples} texts (GTE + CLIP-text, batched x{args.text_batch}) ...")
    gte_feats, clip_text_feats = encode_texts(samples, gte, clip, args.text_batch)
    gte_feats = _maybe_fp16(gte_feats, args.fp16)
    clip_text_feats = _maybe_fp16(clip_text_feats, args.fp16)
    print(f"  gte_text:  {tuple(gte_feats.shape)}    "
          f"({gte_feats.element_size() * gte_feats.numel() / 1e9:.2f} GB)")
    print(f"  clip_text: {tuple(clip_text_feats.shape)}    "
          f"({clip_text_feats.element_size() * clip_text_feats.numel() / 1e9:.2f} GB)")

    text_path = out_dir / "text_features.pt"
    torch.save({
        "gte": gte_feats,
        "clip_text": clip_text_feats,
        "sample_to_image_idx": sample_to_image_idx,
        "samples_meta": [
            {
                "text": s.text,
                "image_id": s.image_id,
                "category": s.category,
                "count": s.count,
                "idx": s.idx,
            }
            for s in samples
        ],
        "fp16": args.fp16,
    }, text_path)
    print(f"  → wrote {text_path}")

    # Free GTE before doing the vision pass — V-JEPA-2 is the big GPU consumer.
    del gte
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---------- Image features ----------
    print(f"\nEncoding {len(unique_image_ids)} unique images "
          f"(CLIP-vision + V-JEPA-2 patches, batched x{args.image_batch}) ...")
    clip_vis_feats, vjepa_feats = encode_images(
        unique_image_ids, image_paths, clip, vjepa, args.image_batch,
    )
    clip_vis_feats = _maybe_fp16(clip_vis_feats, args.fp16)
    vjepa_feats = _maybe_fp16(vjepa_feats, args.fp16)
    print(f"  clip_vision:   {tuple(clip_vis_feats.shape)}    "
          f"({clip_vis_feats.element_size() * clip_vis_feats.numel() / 1e9:.2f} GB)")
    print(f"  vjepa_patches: {tuple(vjepa_feats.shape)}    "
          f"({vjepa_feats.element_size() * vjepa_feats.numel() / 1e9:.2f} GB)")

    img_path = out_dir / "image_features.pt"
    torch.save({
        "clip_vision": clip_vis_feats,
        "vjepa_patches": vjepa_feats,
        "image_ids": unique_image_ids,
        "fp16": args.fp16,
    }, img_path)
    print(f"  → wrote {img_path}")

    print("\n" + "=" * 64)
    print("Pre-compute complete.")
    print("=" * 64)
    print(f"\nTo use: set `cache_dir: {out_dir}` in configs/stage1_alignment.yaml.")


if __name__ == "__main__":
    main()
