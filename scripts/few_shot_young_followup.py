"""Diagnostic follow-up on the young-animal failure.

The first few-shot run got 0/6 on young-animal (dog→puppy, ...) at N=3.
The diagnostic showed the operator was outputting predictions VERY close
to the source (cos(predicted, source) ≈ 0.94+), so the source word kept
winning nearest-neighbor over the actual baby form.

Two hypotheses for the failure:

  A) Pool design — adult animals were included in the candidate pool as
     distractors. Even a small operator shift can't escape the source's
     own attractor when the source is a candidate.

  B) Sample size — N=3 is too few for a purely-semantic concept where
     source and target are far apart in GTE space. (Compare to agentive,
     where write/writer share most letters and live close in GTE.)

This script tests both by sweeping a 3×2 grid:
   N ∈ {3, 5, 9}  ×  pool ∈ {with-lures, no-lures}

If accuracy jumps when lures are removed → it's the pool, not the
architecture (hypothesis A).
If accuracy improves with N → semantic concepts need more pairs
(hypothesis B).
Both can be true at once.
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
from selflearnai.concepts import ConceptOperator, InverseConceptOperator
from selflearnai.foundations import FrozenGTE


# Expanded training pool — 9 (adult, baby) pairs, NONE of which appear in
# the held-out set. Inline so this script is self-contained.
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

# Held-out (same 6 as the original demo).
HELD_OUT = [
    ("horse", "foal"),
    ("deer", "fawn"),
    ("bear", "cub"),
    ("sheep", "lamb"),
    ("goat", "kid"),
    ("pig", "piglet"),
]

# Pool A: only baby animals + variety of distractor babies (NO adult lures).
POOL_NO_LURES = [
    "puppy", "kitten", "calf", "foal", "fawn", "cub", "lamb", "kid",
    "piglet", "chick", "duckling", "gosling", "joey", "tadpole",
    "caterpillar", "fledgling", "cygnet", "hatchling", "fry",
    "pup", "nestling", "larva", "embryo", "infant",
]

# Pool B: adds adult animals as distractors. The source-word problem.
POOL_WITH_LURES = POOL_NO_LURES + [
    "dog", "cat", "cow", "horse", "deer", "bear", "sheep", "goat", "pig",
    "chicken", "duck", "goose", "kangaroo", "frog", "butterfly", "bird",
    "fish", "swan", "rabbit", "fox", "wolf",
]


def train_operator(train_pairs, bundle, gte, device, epochs=1500, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        z_src = bundle.adapter_t(gte.encode([p[0] for p in train_pairs]))
        z_tgt = bundle.adapter_t(gte.encode([p[1] for p in train_pairs]))
    fwd = ConceptOperator(SHARED_DIM).to(device)
    inv = InverseConceptOperator(SHARED_DIM).to(device)
    opt = torch.optim.AdamW(
        list(fwd.parameters()) + list(inv.parameters()), lr=1e-3,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-ckpt", default="checkpoints/stage1/final.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Loading Stage-1 adapters + GTE ...")
    cfg = Stage1Config(device=args.device)
    bundle = Stage1Trainer.load_adapters(args.stage1_ckpt, cfg)
    bundle.eval()
    gte = FrozenGTE(device=args.device)

    print(f"\nYoung-animal follow-up:")
    print(f"  Training pool available: {len(TRAIN_POOL)} pairs")
    print(f"  Held-out (fixed):        {len(HELD_OUT)} pairs")
    print(f"  Pool variants:           with-lures ({len(POOL_WITH_LURES)} candidates)"
          f" / no-lures ({len(POOL_NO_LURES)} candidates)")

    print("\n" + "=" * 72)
    print("RESULTS — held-out accuracy on 6 pairs")
    print("=" * 72)
    print(f"{'N':>3}  {'with-lures pool':>22}  {'no-lures pool':>22}")
    print("-" * 72)

    grid = []
    for N in [3, 5, 9]:
        train_subset = TRAIN_POOL[:N]
        fwd = train_operator(train_subset, bundle, gte, args.device)
        acc_l, preds_l = evaluate(fwd, bundle, gte, HELD_OUT, POOL_WITH_LURES)
        acc_c, preds_c = evaluate(fwd, bundle, gte, HELD_OUT, POOL_NO_LURES)
        nl = int(round(acc_l * len(HELD_OUT)))
        nc = int(round(acc_c * len(HELD_OUT)))
        print(f"{N:>3}     {nl}/{len(HELD_OUT)} = {acc_l:.3f}    "
              f"           {nc}/{len(HELD_OUT)} = {acc_c:.3f}")
        grid.append({"N": N, "lures": acc_l, "clean": acc_c,
                     "preds_lures": preds_l, "preds_clean": preds_c})

    print("\n" + "=" * 72)
    print("DETAIL — N=9, no-lures pool (the most favorable setting)")
    print("=" * 72)
    last = grid[-1]
    for (src, tgt), pred in zip(HELD_OUT, last["preds_clean"]):
        mark = "✓" if pred == tgt else "✗"
        print(f"  {mark} {src:>10s} → {tgt:<10s}  predicted: {pred}")

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    n3 = grid[0]
    n9 = grid[-1]
    print(f"  Effect of removing adult lures (at N=3): "
          f"{n3['lures']:.2f} → {n3['clean']:.2f}  (Δ = {n3['clean'] - n3['lures']:+.2f})")
    print(f"  Effect of more samples (no-lures pool):  "
          f"{n3['clean']:.2f} → {n9['clean']:.2f}  (Δ = {n9['clean'] - n3['clean']:+.2f})")
    print(f"  Best setting (N=9, no-lures):            {n9['clean']:.2f}")
    print()
    print("  Reading:")
    print("    • Big jump from removing lures = pool design was the issue.")
    print("    • Big jump from more N         = semantic concepts need more samples.")
    print("    • Both jumps                   = both factors at play; combined matters.")
    print("    • Neither jumps                = architecture genuinely can't learn this.")


if __name__ == "__main__":
    main()
