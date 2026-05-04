"""Multi-head operator vs single-head — does routing lift the young-animal
ceiling?

The single-head operator plateaus at 0.500 on cross-category-preserving
concepts (young-animal at N=9, no-lures pool). Hypothesis: one shared
direction can move embeddings into the baby-animal region but can't
encode source-species-specific shifts. K directions + a router should let
sources of different "shapes" route to different shift directions.

Comparison conditions (5 architectures, all trained on the same N=9 pairs,
evaluated on the same 6 held-outs, same no-lures pool, 3 seeds):

  1. single_head_mlp192   ConceptOperator          (current baseline)
  2. single_head_mlp384   ConceptOperator+bigger   (capacity control)
  3. multi_head_K2        MultiHeadConceptOperator
  4. multi_head_K3        MultiHeadConceptOperator
  5. multi_head_K4        MultiHeadConceptOperator

If multi-head improves over single_head_mlp384 (the matched-capacity
control), the gain is from ROUTING, not from capacity. If both single_head
variants and multi_heads stay at 0.500, the limit is in the latent space
(GTE), not in the operator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai import SHARED_DIM
from selflearnai.adapters.train_adapters import Stage1Config, Stage1Trainer
from selflearnai.concepts import (
    ConceptOperator,
    InverseConceptOperator,
    MultiHeadConceptOperator,
    MultiHeadInverseConceptOperator,
)
from selflearnai.foundations import FrozenGTE


# Same training pool + held-out as scripts/few_shot_young_followup.py
TRAIN_POOL = [
    ("dog", "puppy"),
    ("cat", "kitten"),
    ("cow", "calf"),
    ("chicken", "chick"),
    ("duck", "duckling"),
    ("goose", "gosling"),
    ("swan", "cygnet"),
    ("frog", "tadpole"),
    ("butterfly", "caterpillar"),
]

HELD_OUT = [
    ("horse", "foal"),
    ("deer", "fawn"),
    ("bear", "cub"),
    ("sheep", "lamb"),
    ("goat", "kid"),
    ("pig", "piglet"),
]

POOL_NO_LURES = [
    "puppy", "kitten", "calf", "foal", "fawn", "cub", "lamb", "kid",
    "piglet", "chick", "duckling", "gosling", "joey", "tadpole",
    "caterpillar", "fledgling", "cygnet", "hatchling", "fry",
    "pup", "nestling", "larva", "embryo", "infant",
]


def train(
    train_pairs, bundle, gte, device, *,
    arch: str, seed: int = 0, epochs: int = 1500, lr: float = 1e-3,
):
    """Train a forward + inverse operator pair of the requested architecture.
    Returns the trained forward operator."""
    torch.manual_seed(seed)
    with torch.no_grad():
        z_src = bundle.adapter_t(gte.encode([p[0] for p in train_pairs]))
        z_tgt = bundle.adapter_t(gte.encode([p[1] for p in train_pairs]))

    if arch == "single_head_mlp192":
        fwd = ConceptOperator(SHARED_DIM, mlp_hidden=192).to(device)
        inv = InverseConceptOperator(SHARED_DIM, mlp_hidden=192).to(device)
    elif arch == "single_head_mlp384":
        fwd = ConceptOperator(SHARED_DIM, mlp_hidden=384).to(device)
        inv = InverseConceptOperator(SHARED_DIM, mlp_hidden=384).to(device)
    elif arch.startswith("multi_head_K"):
        K = int(arch.split("K")[1])
        fwd = MultiHeadConceptOperator(SHARED_DIM, num_heads=K, mlp_hidden=192).to(device)
        inv = MultiHeadInverseConceptOperator(SHARED_DIM, num_heads=K, mlp_hidden=192).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    opt = torch.optim.AdamW(
        list(fwd.parameters()) + list(inv.parameters()), lr=lr,
    )
    for _ in range(epochs):
        opt.zero_grad()
        l_fwd = F.mse_loss(fwd(z_src), z_tgt)
        l_inv = F.mse_loss(inv(z_tgt), z_src)
        (l_fwd + l_inv).backward()
        opt.step()
    return fwd


@torch.no_grad()
def evaluate(fwd, bundle, gte, held_out, pool):
    src_texts = [p[0] for p in held_out]
    tgt_texts = [p[1] for p in held_out]
    z_src = bundle.adapter_t(gte.encode(src_texts))
    z_pred = fwd(z_src)
    z_pool = bundle.adapter_t(gte.encode(pool))
    pred_n = F.normalize(z_pred, dim=-1)
    pool_n = F.normalize(z_pool, dim=-1)
    sims = pred_n @ pool_n.T
    best = sims.argmax(dim=-1).tolist()
    preds = [pool[i] for i in best]
    correct = sum(p == t for p, t in zip(preds, tgt_texts))
    return correct / len(held_out), preds


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-ckpt", default="checkpoints/stage1/final.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    print("Loading Stage-1 adapters + GTE ...")
    cfg = Stage1Config(device=args.device)
    bundle = Stage1Trainer.load_adapters(args.stage1_ckpt, cfg)
    bundle.eval()
    gte = FrozenGTE(device=args.device)

    archs = [
        "single_head_mlp192",
        "single_head_mlp384",
        "multi_head_K2",
        "multi_head_K3",
        "multi_head_K4",
    ]

    print(f"\nDataset: {len(TRAIN_POOL)} (adult, baby) training pairs, "
          f"{len(HELD_OUT)} held-out, no-lures pool ({len(POOL_NO_LURES)} candidates)\n")

    print("=" * 80)
    print(f"{'arch':<22}  {'params':>9}  {'mean acc':>10}  {'per-seed accs':<22}")
    print("=" * 80)

    rows = []
    last_preds_per_arch: dict[str, list[str]] = {}

    for arch in archs:
        accs = []
        params = None
        last_preds = None
        for seed in range(args.seeds):
            fwd = train(TRAIN_POOL, bundle, gte, args.device, arch=arch, seed=seed)
            acc, preds = evaluate(fwd, bundle, gte, HELD_OUT, POOL_NO_LURES)
            accs.append(acc)
            last_preds = preds
            if params is None:
                params = n_params(fwd)
        mean = sum(accs) / len(accs)
        rows.append({"arch": arch, "params": params, "mean": mean, "accs": accs})
        last_preds_per_arch[arch] = last_preds
        accs_str = "[" + ", ".join(f"{a:.2f}" for a in accs) + "]"
        print(f"{arch:<22}  {params:>9,}  {mean:>10.3f}  {accs_str:<22}")

    # ---- Side-by-side per-item predictions for the best arch ----
    print("\n" + "=" * 80)
    best = max(rows, key=lambda r: r["mean"])
    print(f"DETAIL — best arch: {best['arch']} (mean acc {best['mean']:.3f})")
    print("=" * 80)
    preds = last_preds_per_arch[best["arch"]]
    for (src, tgt), pred in zip(HELD_OUT, preds):
        mark = "✓" if pred == tgt else "✗"
        print(f"  {mark} {src:>10s} → {tgt:<10s}  predicted: {pred}")

    # ---- Comparison vs baseline ----
    base = next(r for r in rows if r["arch"] == "single_head_mlp192")
    bigger = next(r for r in rows if r["arch"] == "single_head_mlp384")
    multi_best = max(
        (r for r in rows if r["arch"].startswith("multi_head")),
        key=lambda r: r["mean"],
    )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"  Single-head mlp192 (baseline):   acc = {base['mean']:.3f}  ({base['params']:,} params)")
    print(f"  Single-head mlp384 (capacity):   acc = {bigger['mean']:.3f}  ({bigger['params']:,} params)")
    print(f"  Multi-head best ({multi_best['arch']}):"
          f"   acc = {multi_best['mean']:.3f}  ({multi_best['params']:,} params)")
    print()
    delta_capacity = bigger["mean"] - base["mean"]
    delta_routing = multi_best["mean"] - bigger["mean"]
    print(f"  Δ from doubling MLP capacity:    {delta_capacity:+.3f}")
    print(f"  Δ from adding routing on top:    {delta_routing:+.3f}")
    print()
    if delta_routing > 0.10:
        print("  → Multi-head ROUTING measurably helps. The cross-category limit was")
        print("    operator expressivity, not latent-space structure.")
    elif delta_capacity > 0.10:
        print("  → Bigger MLP helps as much as routing. The bottleneck is capacity,")
        print("    not the single-vs-multi-direction architecture.")
    else:
        print("  → Neither capacity nor routing helps. The ~0.50 ceiling is in the")
        print("    latent space itself: GTE doesn't separate horse from cow strongly")
        print("    enough for the operator to route them to different baby-targets.")


if __name__ == "__main__":
    main()
