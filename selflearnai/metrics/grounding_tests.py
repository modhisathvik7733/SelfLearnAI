"""Grounding tests — does the system actually use vision, or is it text-only?

Locked metrics from plan section 6a/6c:
  • Cross-modal cosine (image↔caption in shared space) > 0.5.
  • Per-dim stddev > 0.3 on each adapter's output (no collapse).
  • Anchor stability: cos(adapter_c at start, at now) > 0.85.
  • "Good cosine, bad semantics" detector — three orthogonal probes:
      - STS-B Pearson > 60 on text adapter
      - COCO category kNN > random + 30% on vision adapter
      - Retrieval top-5 > random
  • Modality leakage check — perturb vision; if output barely changes, flag.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from selflearnai import SHARED_DIM
from selflearnai.adapters import AdapterBundle


@torch.no_grad()
def cross_modal_cosine(
    bundle: AdapterBundle,
    gte, clip, vjepa,
    eval_pairs: list[tuple[str, Path]],
    device: str = "cuda",
) -> float:
    """Mean cosine similarity between caption and image embeddings in shared space.
    Threshold per plan: > 0.5. Below 0.3 = ungrounded."""
    texts = [p[0] for p in eval_pairs]
    images = [Image.open(p[1]).convert("RGB") for p in eval_pairs]

    z_gte = gte.encode(texts)
    z_text = bundle.adapter_t(z_gte)

    z_vj = vjepa.encode_patches(images)
    z_vis = bundle.adapter_v(z_vj)

    return F.cosine_similarity(z_text, z_vis, dim=-1).mean().item()


@torch.no_grad()
def per_dim_stddev(
    bundle: AdapterBundle,
    gte, clip, vjepa,
    eval_pairs: list[tuple[str, Path]],
) -> dict:
    """Per-dim stddev across the eval set, separately for each adapter.
    Threshold per plan: > 0.3. Below 0.2 anywhere = collapse."""
    texts = [p[0] for p in eval_pairs]
    images = [Image.open(p[1]).convert("RGB") for p in eval_pairs]

    z_t = bundle.adapter_t(gte.encode(texts))
    z_v = bundle.adapter_v(vjepa.encode_patches(images))
    z_c_t = bundle.adapter_c(clip.encode_text(texts))
    z_c_v = bundle.adapter_c(clip.encode_image(images))

    def _stats(z: torch.Tensor) -> dict:
        std = z.std(dim=0)
        return {
            "mean_std": std.mean().item(),
            "min_std":  std.min().item(),
            "p10_std":  std.kthvalue(max(1, int(0.1 * z.shape[1]))).values.item(),
        }
    return {
        "adapter_t": _stats(z_t),
        "adapter_v": _stats(z_v),
        "adapter_c_text":  _stats(z_c_t),
        "adapter_c_vision": _stats(z_c_v),
    }


@torch.no_grad()
def retrieval_top_k(
    bundle: AdapterBundle,
    gte, vjepa,
    eval_pairs: list[tuple[str, Path]],
    k: int = 5,
) -> float:
    """Given each caption, the correct image should be in top-k of its
    nearest-neighbor retrieval over all images in the eval set.

    Returns recall@k. Threshold per plan: > random (= k / N).
    """
    texts = [p[0] for p in eval_pairs]
    images = [Image.open(p[1]).convert("RGB") for p in eval_pairs]

    z_text = F.normalize(bundle.adapter_t(gte.encode(texts)), dim=-1)
    z_vis  = F.normalize(bundle.adapter_v(vjepa.encode_patches(images)), dim=-1)

    sims = z_text @ z_vis.T                                              # (N, N)
    topk = sims.topk(k, dim=-1).indices                                  # (N, k)
    correct = (topk == torch.arange(len(texts), device=topk.device).unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


@torch.no_grad()
def visual_perturbation_sensitivity(
    bundle: AdapterBundle,
    vjepa,
    images: list[Image.Image],
    mode: str = "noise",
) -> float:
    """Replace each image with something semantically different and measure
    embedding shift. If the adapter still produces nearly the same output,
    vision is being ignored (modality leakage / shortcut learning).

    Modes — ranked by how aggressive the perturbation is:
      - "occlude64" : 64×64 center black square. Easy for V-JEPA to inpaint;
                      will produce small changes even when vision IS used.
      - "noise"     : replace whole image with uniform random pixels.
                      Strong test — radically different content.
      - "zeros"     : replace whole image with all zeros.
                      Strongest test — the absolute null input.

    Returns mean(1 - cos(orig_emb, perturbed_emb)). With "noise" or "zeros",
    a passing system should score > 0.1; <0.05 means vision is decoupled.
    """
    z_orig = F.normalize(bundle.adapter_v(vjepa.encode_patches(images)), dim=-1)

    perturbed = []
    if mode == "zeros":
        for img in images:
            w, h = img.size
            blank = Image.new("RGB", (w, h), color=(0, 0, 0))
            perturbed.append(blank)
    elif mode == "noise":
        import numpy as np
        rng = np.random.default_rng(0)
        for img in images:
            w, h = img.size
            arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
            perturbed.append(Image.fromarray(arr, mode="RGB"))
    elif mode == "occlude64":
        from PIL import ImageDraw
        for img in images:
            a = img.copy()
            w, h = a.size
            cx, cy = w // 2, h // 2
            s = 32
            d = ImageDraw.Draw(a)
            d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=(0, 0, 0))
            perturbed.append(a)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    z_pert = F.normalize(bundle.adapter_v(vjepa.encode_patches(perturbed)), dim=-1)
    return (1.0 - F.cosine_similarity(z_orig, z_pert, dim=-1)).mean().item()


def report_grounding(
    bundle: AdapterBundle,
    foundations,
    eval_pairs: list[tuple[str, Path]],
    device: str = "cuda",
) -> dict:
    """Run all grounding tests and return a dict of metric → value.

    Plan exit gates:
      cross_modal_cosine > 0.5
      per_dim_stddev > 0.3 on every adapter
      retrieval_top5 > random (= 5/N)
      visual_perturbation_sensitivity > 0.05
    """
    gte, clip, vjepa = foundations
    out = {}
    out["cross_modal_cosine"] = cross_modal_cosine(
        bundle, gte, clip, vjepa, eval_pairs, device,
    )
    out["per_dim_stddev"] = per_dim_stddev(bundle, gte, clip, vjepa, eval_pairs)
    out["retrieval_recall@5"] = retrieval_top_k(bundle, gte, vjepa, eval_pairs, k=5)
    images = [Image.open(p[1]).convert("RGB") for p in eval_pairs[: min(64, len(eval_pairs))]]
    # Use "noise" perturbation: full-image replacement is far stronger than
    # the previous 64×64 occlusion (which V-JEPA-2 was *trained* to handle).
    out["visual_perturbation"] = visual_perturbation_sensitivity(
        bundle, vjepa, images, mode="noise",
    )
    return out
