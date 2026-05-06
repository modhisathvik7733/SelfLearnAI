"""Curriculum Tier 0.1 — train + evaluate the property RelationalOperator.

The first reasoning operator in SelfLearnAI's curriculum. After ~12 months
of validating word-morphology operators (plural, past_tense, opposite,
etc.), this is the architecture's first move into ATOMIC RELATIONAL
REASONING — answering "what is X's function?", "what does X need?",
"what color is X?" — by computing in ψ-space, not retrieving.

Per the user's design feedback (this session, 2026-05-06):
  1. Operator works in ψ-space ONLY. Inputs/outputs are vectors. Never text.
  2. Atomic — one relation per call. Multi-step ('why does X need Y?')
     is the planner's job, not one fat operator's.
  3. Generalization test = held-out ENTITIES, not held-out pairs of
     known entities. If we trained on (cat, sound, meow), the test asks
     (snake, sound, ?) — the entity is new, the axis is the same. The
     operator must have learned the AXIS as a general direction, not
     memorized entity→value lookups.

Pipeline:
  1. Load 100 train pairs across 8 axes from data/property/.
  2. Build axis vocabulary (8 axes); encode every entity + every value via E5.
  3. Train RelationalOperator (~150K params) for ~3000 epochs with cosine loss.
  4. Build a value pool (all train + holdout values, deduped, ~120 candidates).
  5. Evaluate on 30 truly-novel held-out pairs:
       a. For each (entity, axis, expected_value):
            z_pred = op(encode(entity), axis_idx)
            top1   = argmax over pool of cos(z_pred, z_pool)
       b. Top-1 hit rate per axis + overall.
  6. Acceptance: held-out top-1 ≥ 60% overall AND ≥ 50% on at least 6 of 8 axes.

Run on the GPU box (~5-10 min):
  python scripts/curriculum_tier0_property.py

This is the FIRST reasoning operator. If it passes, Tier 0.2 (causation),
0.3 (needs), 0.4 (counterfactual) follow with the same recipe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.concepts.relational_operator import (
    AxisVocabulary,
    RelationalOperator,
)

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn


# ---------------------------------------------------------------------------
# Data IO
# ---------------------------------------------------------------------------

def read_property_tsv(path: Path) -> list[dict]:
    """Read a property TSV with columns: entity, axis, value (TAB-separated).
    Strips comments + blank lines."""
    rows: list[dict] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            entity = parts[0].strip()
            axis = parts[1].strip()
            value = parts[2].strip()
            if entity and axis and value:
                rows.append({"entity": entity, "axis": axis, "value": value})
    return rows


def assert_no_entity_leakage(train_pairs: list[dict],
                             holdout_pairs: list[dict]) -> None:
    """Held-out entities MUST NOT appear in training (per design rule)."""
    train_entities = {p["entity"] for p in train_pairs}
    leaks = sorted({
        p["entity"] for p in holdout_pairs if p["entity"] in train_entities
    })
    if leaks:
        print(f"FATAL: held-out entities leak into training: {leaks}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_operator(
    op: RelationalOperator,
    z_entity: torch.Tensor,
    z_value: torch.Tensor,
    axis_idx: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    log_every: int,
) -> list[dict]:
    """Cosine-loss training. Mirrors Stage 0 train_operator pattern but
    with axis-conditioned forward."""
    opt = torch.optim.AdamW(op.parameters(), lr=lr)
    history: list[dict] = []
    op.train()
    for step in range(epochs):
        opt.zero_grad()
        z_pred = op(z_entity, axis_idx)
        loss = (1.0 - F.cosine_similarity(z_pred, z_value, dim=-1)).mean()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == epochs - 1:
            with torch.no_grad():
                cos_mean = F.cosine_similarity(z_pred, z_value, dim=-1).mean().item()
            history.append({
                "step": step,
                "loss": float(loss.item()),
                "cos_train": float(cos_mean),
            })
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return history


# ---------------------------------------------------------------------------
# Held-out evaluation: top-1 retrieval over a value pool
# ---------------------------------------------------------------------------

def build_value_pool(
    train_pairs: list[dict],
    holdout_pairs: list[dict],
) -> list[str]:
    """Pool = unique values from BOTH train + holdout. Each pair's
    `expected_value` must be in this pool (otherwise top-1 can't be right
    for it)."""
    seen: set[str] = set()
    pool: list[str] = []
    for p in train_pairs + holdout_pairs:
        v = p["value"]
        if v not in seen:
            seen.add(v)
            pool.append(v)
    return pool


@torch.no_grad()
def evaluate_holdout(
    op: RelationalOperator,
    encode_fn,
    holdout_pairs: list[dict],
    pool: list[str],
    vocab: AxisVocabulary,
    device: str,
) -> dict:
    """For each held-out (entity, axis, expected_value), compute
    z_pred = op(encode(entity), axis_idx). Compare to the pool by cosine.
    Top-1 hit if argmax pool[i] == expected_value.

    Returns:
      {
        "rows": list of per-pair records (with predicted top-3),
        "n_total":      total held-out pairs,
        "n_top1_hit":   number of pairs where top-1 is the expected value,
        "top1_rate":    n_top1_hit / n_total,
        "per_axis":     {axis: {"n": ..., "top1_hit": ..., "top1_rate": ...}}
      }
    """
    z_pool = encode_fn(pool)                                        # (P, D)
    z_pool_norm = F.normalize(z_pool, p=2, dim=-1)
    rows = []
    per_axis_acc: dict[str, dict] = {}
    for p in holdout_pairs:
        entity = p["entity"]
        axis = p["axis"]
        expected = p["value"]
        if axis not in vocab:
            # Should never happen — holdout uses same axes as train.
            continue
        if expected not in pool:
            continue
        axis_idx = vocab[axis]
        z_entity = encode_fn([entity]).squeeze(0).flatten()         # (D,)
        z_pred = op(z_entity, axis_idx).flatten()
        z_pred_norm = F.normalize(z_pred.unsqueeze(0), p=2, dim=-1).squeeze(0)
        scores = z_pool_norm @ z_pred_norm                          # (P,)
        top3 = torch.topk(scores, k=min(3, scores.numel()))
        top3_words = [pool[i] for i in top3.indices.tolist()]
        top3_scores = top3.values.tolist()
        top1_word = top3_words[0]
        is_hit = (top1_word == expected)
        rows.append({
            "entity": entity,
            "axis": axis,
            "expected": expected,
            "top1": top1_word,
            "top3": top3_words,
            "top3_scores": top3_scores,
            "is_top1_hit": is_hit,
        })
        agg = per_axis_acc.setdefault(axis, {"n": 0, "top1_hit": 0})
        agg["n"] += 1
        if is_hit:
            agg["top1_hit"] += 1
    n_total = len(rows)
    n_hit = sum(1 for r in rows if r["is_top1_hit"])
    for axis, agg in per_axis_acc.items():
        agg["top1_rate"] = agg["top1_hit"] / agg["n"] if agg["n"] else 0.0
    return {
        "rows": rows,
        "n_total": n_total,
        "n_top1_hit": n_hit,
        "top1_rate": n_hit / n_total if n_total else 0.0,
        "per_axis": per_axis_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2",
                        choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--mlp-hidden", type=int, default=192)
    parser.add_argument("--top1-min", type=float, default=0.60,
                        help="Acceptance: held-out top-1 ≥ this overall.")
    parser.add_argument("--per-axis-min", type=float, default=0.50,
                        help="Per-axis top-1 ≥ this on ≥ N axes.")
    parser.add_argument("--per-axis-pass-count", type=int, default=6,
                        help="Number of axes that must clear --per-axis-min.")
    parser.add_argument("--train-tsv", default="data/property/text_pairs_train.tsv")
    parser.add_argument("--holdout-tsv", default="data/property/text_pairs_holdout.tsv")
    parser.add_argument("--out", default="results/curriculum/tier0_property.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Curriculum Tier 0.1 — Property RelationalOperator")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Acceptance: held-out top-1 ≥ {args.top1_min} overall AND "
          f"per-axis ≥ {args.per_axis_min} on ≥ {args.per_axis_pass_count}/8 axes")

    # ---- Load data ----------------------------------------------------
    print("\n[1] Loading property pairs")
    print("-" * 78)
    train_pairs = read_property_tsv(Path(args.train_tsv))
    holdout_pairs = read_property_tsv(Path(args.holdout_tsv))
    print(f"  train:    {len(train_pairs)} pairs")
    print(f"  holdout:  {len(holdout_pairs)} pairs (truly-novel entities)")

    assert_no_entity_leakage(train_pairs, holdout_pairs)
    print(f"  ✓ no entity leakage between train + holdout")

    vocab = AxisVocabulary.from_pairs(train_pairs + holdout_pairs)
    print(f"  axes ({len(vocab)}): {vocab.axis_names()}")

    pool = build_value_pool(train_pairs, holdout_pairs)
    print(f"  value pool: {len(pool)} unique values")

    # ---- Encoder ------------------------------------------------------
    print(f"\n[2] Loading {enc_cfg['model']}")
    print("-" * 78)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)
    DIM = enc_cfg["dim"]

    # ---- Encode train data --------------------------------------------
    print("\n[3] Encoding training data")
    print("-" * 78)
    train_entities = [p["entity"] for p in train_pairs]
    train_values = [p["value"] for p in train_pairs]
    train_axis_idx = torch.tensor(
        [vocab[p["axis"]] for p in train_pairs],
        dtype=torch.long, device=args.device,
    )
    z_entity = encode(train_entities)                # (N, D)
    z_value = encode(train_values)                   # (N, D)
    print(f"  z_entity: {tuple(z_entity.shape)}")
    print(f"  z_value:  {tuple(z_value.shape)}")

    # ---- Train --------------------------------------------------------
    print(f"\n[4] Training RelationalOperator "
          f"({args.epochs} epochs, lr={args.lr})")
    print("-" * 78)
    torch.manual_seed(args.seed)
    op = RelationalOperator(
        dim=DIM, num_axes=len(vocab), mlp_hidden=args.mlp_hidden,
    ).to(args.device)
    n_params = sum(p.numel() for p in op.parameters())
    print(f"  RelationalOperator: {n_params/1e3:.1f}K params, "
          f"{len(vocab)} axes")
    history = train_operator(
        op, z_entity, z_value, train_axis_idx,
        epochs=args.epochs, lr=args.lr, log_every=args.log_every,
    )
    for h in history:
        print(f"  step {h['step']:>4}  loss={h['loss']:.4f}  "
              f"cos(train)={h['cos_train']:.4f}")

    # ---- Held-out evaluation -----------------------------------------
    print(f"\n[5] Held-out evaluation ({len(holdout_pairs)} truly-novel pairs)")
    print("-" * 78)
    eval_result = evaluate_holdout(
        op, encode, holdout_pairs, pool, vocab, args.device,
    )

    print(f"\n  per-axis top-1:")
    print(f"    {'axis':<12}  {'n':>3}  {'top-1':>5}  {'rate':>6}")
    for axis in sorted(eval_result["per_axis"].keys()):
        agg = eval_result["per_axis"][axis]
        passing = "✓" if agg["top1_rate"] >= args.per_axis_min else "✗"
        print(f"    {axis:<12}  {agg['n']:>3}  "
              f"{agg['top1_hit']:>2}/{agg['n']:<2}  "
              f"{agg['top1_rate']:.3f}  {passing}")

    print(f"\n  per-pair detail:")
    print(f"    {'#':<2} {'entity':<14} {'axis':<10} {'expected':<16} "
          f"{'top1':<16} {'hit?'}")
    for i, r in enumerate(eval_result["rows"]):
        mark = "✓" if r["is_top1_hit"] else "✗"
        print(f"    {i+1:<2} {r['entity']:<14} {r['axis']:<10} "
              f"{r['expected']:<16} {r['top1']:<16} {mark}")
        if not r["is_top1_hit"]:
            top3_pairs = ", ".join(
                f"{w}({s:.3f})" for w, s in zip(r["top3"], r["top3_scores"])
            )
            print(f"       top-3: {top3_pairs}")

    # ---- Acceptance gate ---------------------------------------------
    overall_rate = eval_result["top1_rate"]
    n_axes_passing = sum(
        1 for axis, agg in eval_result["per_axis"].items()
        if agg["top1_rate"] >= args.per_axis_min
    )
    n_axes_total = len(eval_result["per_axis"])

    print("\n" + "=" * 78)
    print("ACCEPTANCE")
    print("=" * 78)
    overall_pass = overall_rate >= args.top1_min
    axes_pass = n_axes_passing >= args.per_axis_pass_count
    accept = overall_pass and axes_pass
    print(f"  overall top-1: {eval_result['n_top1_hit']}/{eval_result['n_total']} "
          f"= {overall_rate:.3f}  {'✓' if overall_pass else '✗'} "
          f"(target ≥ {args.top1_min})")
    print(f"  axes passing ≥ {args.per_axis_min}: "
          f"{n_axes_passing}/{n_axes_total}  "
          f"{'✓' if axes_pass else '✗'} (target ≥ {args.per_axis_pass_count})")

    if accept:
        verdict = "TIER_0_PROPERTY_PASS"
        message = (
            f"The first reasoning operator works. Held-out top-1 "
            f"{overall_rate:.0%} on truly-novel entities "
            f"({n_axes_passing}/{n_axes_total} axes pass per-axis gate). "
            f"The brain has its first relational primitive — "
            f"property(ψ_entity, axis) → ψ_value computed in pure ψ-space. "
            f"Proceed to Tier 0.2 (causation operator)."
        )
    elif overall_rate >= 0.40:
        verdict = "TIER_0_PROPERTY_PARTIAL"
        message = (
            f"Operator partially works (held-out top-1 {overall_rate:.0%}) "
            f"but below the {args.top1_min:.0%} gate. Inspect per-axis "
            f"breakdown — likely some axes (e.g. those with sparse training) "
            f"are weaker than others. Try (a) more pairs per weak axis, "
            f"(b) longer training, (c) per-axis MLP instead of shared."
        )
    else:
        verdict = "TIER_0_PROPERTY_FAIL"
        message = (
            f"Operator did not generalize beyond training (held-out top-1 "
            f"{overall_rate:.0%}). Investigate: did training cos converge? "
            f"Are the axis embeddings diverging? Maybe entity-novelty in "
            f"E5 is too high — encoded test entities cluster differently "
            f"from training entities. May need richer training pairs."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save ---------------------------------------------------------
    payload = {
        "task": "tier0_property",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "n_train_pairs": len(train_pairs),
        "n_holdout_pairs": len(holdout_pairs),
        "axes": vocab.axis_names(),
        "operator": {
            "n_params": n_params,
            "epochs": args.epochs,
            "lr": args.lr,
            "loss_history": history,
        },
        "thresholds": {
            "top1_min": args.top1_min,
            "per_axis_min": args.per_axis_min,
            "per_axis_pass_count": args.per_axis_pass_count,
        },
        "eval": {
            "n_total": eval_result["n_total"],
            "n_top1_hit": eval_result["n_top1_hit"],
            "top1_rate": overall_rate,
            "per_axis": eval_result["per_axis"],
            "rows": eval_result["rows"],
        },
        "verdict": verdict,
        "message": message,
        "accept": accept,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if accept else 1)


if __name__ == "__main__":
    main()
