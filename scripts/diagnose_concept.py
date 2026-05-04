"""Diagnostic dump for a trained concept operator.

When held_out_transfer fails, we want to know WHY:
  • Is the operator collapsing to identity? (alpha small, no shift)
  • Is the source/target separation tiny in shared-space? (encoder issue)
  • Is the candidate pool too crowded? (right answer in top-5 but not top-1)
  • Is the predicted embedding landing in a totally wrong region?

This script prints, for each held-out source:
  - The source and expected target.
  - The model's top-5 nearest candidates with cosines.
  - cos(source_emb, target_emb)  — how separable are they in shared space?
  - cos(predicted_emb, source_emb) — does the operator actually move things?
  - cos(predicted_emb, target_emb) — how close did we get?

Plus operator parameter health:
  - alpha (linear-shift scalar)
  - v.norm() (concept direction magnitude)
  - residual MLP weight stats

Usage:
    python3 scripts/diagnose_concept.py \\
        --stage1-ckpt checkpoints/stage1/final.pt \\
        --stage2-ckpt checkpoints/stage2_past_tense/final.pt \\
        --data-dir data/past_tense \\
        --device cuda
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


def _strip_comment(s: str) -> str:
    idx = s.find("#")
    return (s[:idx] if idx >= 0 else s).strip()


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                src = _strip_comment(parts[0])
                tgt = _strip_comment(parts[1])
                if src and tgt:
                    out.append((src, tgt))
    return out


def _read_pool(path: Path) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    device = args.device
    cfg = Stage1Config(device=device)
    bundle = Stage1Trainer.load_adapters(args.stage1_ckpt, cfg)
    bundle.eval()
    gte = FrozenGTE(device=device)

    fwd = ConceptOperator(SHARED_DIM).to(device)
    inv = InverseConceptOperator(SHARED_DIM).to(device)
    ckpt = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
    fwd.load_state_dict(ckpt["fwd"])
    inv.load_state_dict(ckpt["inv"])
    fwd.eval(); inv.eval()

    data_dir = Path(args.data_dir)
    train_pairs = _read_pairs(data_dir / "text_pairs_train.tsv")
    held_out_pairs = _read_pairs(data_dir / "text_pairs_held_out.tsv")
    candidate_pool = _read_pool(data_dir / "candidate_pool.txt")

    # ---------------------- Operator parameter health ----------------------
    print("=" * 72)
    print(f"OPERATOR PARAMETER HEALTH")
    print("=" * 72)
    with torch.no_grad():
        alpha = fwd.alpha.item()
        v_norm = fwd.v.norm().item()
        v_mean = fwd.v.mean().item()
        v_std = fwd.v.std().item()
        # Residual MLP weights: expect non-trivial L2.
        res_norm = sum(p.norm().item() ** 2 for p in fwd.residual.parameters()) ** 0.5
    print(f"  forward.alpha            = {alpha:+.4f}    (linear shift scalar; if ~0 → operator is residual-only)")
    print(f"  forward.v.norm()         = {v_norm:.4f}    (concept direction magnitude; if tiny → no shift)")
    print(f"  forward.v: mean={v_mean:+.4f}, std={v_std:.4f}")
    print(f"  forward.residual.weights L2 norm = {res_norm:.4f}")

    # ---------------------- Source/target separation in shared space ----------------------
    print()
    print("=" * 72)
    print("SOURCE/TARGET SEPARATION (in adapter_t shared-space)")
    print("=" * 72)
    print("  If these cosines are >0.95 → encoder doesn't distinguish")
    print("  source vs target → no concept signal to learn from.")
    print()
    with torch.no_grad():
        srcs = [p[0] for p in train_pairs]
        tgts = [p[1] for p in train_pairs]
        z_src = bundle.adapter_t(gte.encode(srcs))
        z_tgt = bundle.adapter_t(gte.encode(tgts))
        cos_per_pair = F.cosine_similarity(z_src, z_tgt, dim=-1)
        print(f"  Training pairs ({len(train_pairs)}):")
        print(f"    cos(src, tgt) — mean={cos_per_pair.mean().item():.4f}, "
              f"min={cos_per_pair.min().item():.4f}, "
              f"max={cos_per_pair.max().item():.4f}")
        # Show individual pairs sorted from least to most-separated
        cos_list = sorted(zip(cos_per_pair.tolist(), srcs, tgts))
        print(f"    Most-separated 3:")
        for c, s, t in cos_list[:3]:
            print(f"      {s:10s} ↔ {t:10s}   cos={c:.4f}")
        print(f"    Least-separated 3 (already nearly identical):")
        for c, s, t in cos_list[-3:]:
            print(f"      {s:10s} ↔ {t:10s}   cos={c:.4f}")

    # ---------------------- Held-out predictions ----------------------
    print()
    print("=" * 72)
    print(f"HELD-OUT PREDICTIONS (top-{args.top_k} candidates per source)")
    print("=" * 72)
    with torch.no_grad():
        ho_srcs = [p[0] for p in held_out_pairs]
        ho_tgts = [p[1] for p in held_out_pairs]
        z_src_ho = bundle.adapter_t(gte.encode(ho_srcs))
        z_tgt_ho = bundle.adapter_t(gte.encode(ho_tgts))
        z_pred = fwd(z_src_ho)
        z_pool = bundle.adapter_t(gte.encode(candidate_pool))
        z_pool_n = F.normalize(z_pool, dim=-1)

        for i, (src, tgt) in enumerate(zip(ho_srcs, ho_tgts)):
            pred_n = F.normalize(z_pred[i:i+1], dim=-1)
            sims = (pred_n @ z_pool_n.T).squeeze(0)
            top_vals, top_idx = sims.topk(args.top_k)

            cos_src_tgt = F.cosine_similarity(
                z_src_ho[i:i+1], z_tgt_ho[i:i+1], dim=-1
            ).item()
            cos_pred_src = F.cosine_similarity(
                z_pred[i:i+1], z_src_ho[i:i+1], dim=-1
            ).item()
            cos_pred_tgt = F.cosine_similarity(
                z_pred[i:i+1], z_tgt_ho[i:i+1], dim=-1
            ).item()

            best = candidate_pool[top_idx[0].item()]
            verdict = "✓" if best == tgt else "✗"
            tgt_rank = None
            if tgt in candidate_pool:
                # global rank
                tgt_idx = candidate_pool.index(tgt)
                tgt_rank = (sims > sims[tgt_idx]).sum().item() + 1

            print(f"\n  {verdict} {src:10s} → {tgt:10s}     "
                  f"cos(src,tgt)={cos_src_tgt:.3f}   "
                  f"cos(pred,src)={cos_pred_src:.3f}   "
                  f"cos(pred,tgt)={cos_pred_tgt:.3f}")
            print(f"      target rank in pool: "
                  f"{tgt_rank if tgt_rank else 'NOT IN POOL'} of {len(candidate_pool)}")
            print(f"      top-{args.top_k}:")
            for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
                marker = "  ← target" if candidate_pool[idx] == tgt else ""
                print(f"        cos={v:+.4f}   {candidate_pool[idx]}{marker}")

    # ---------------------- Diagnosis hint ----------------------
    print()
    print("=" * 72)
    print("DIAGNOSIS HINTS")
    print("=" * 72)
    src_tgt_mean = cos_per_pair.mean().item()
    if src_tgt_mean > 0.95:
        print("  ⚠ Source/target cosine in training is very high (> 0.95).")
        print("    The encoder doesn't separate source from target.")
        print("    Operator can satisfy L_fwd by being near-identity → no real")
        print("    shift learned → held-out generalization fails.")
        print("    Fix: switch to morphology-sensitive encoder (T5 / char-level)")
        print("    or augment with surface-form features.")
    elif alpha < 0.05 and v_norm < 0.1:
        print("  ⚠ Operator's linear part is tiny (alpha and v near zero).")
        print("    Forward is probably collapsed to identity.")
        print("    Fix: add a 'must-not-equal-input' regularizer to force shift.")
    else:
        print("  Source/target separation looks reasonable; investigate")
        print("  candidate pool noise or operator capacity.")


if __name__ == "__main__":
    main()
