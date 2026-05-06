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
from selflearnai.concepts.property_pkg import PropertyOperatorPackage

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

def train_operator_cosine(
    op: RelationalOperator,
    z_entity: torch.Tensor,
    z_value: torch.Tensor,
    axis_idx: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    log_every: int,
    weight_decay: float = 0.0,
) -> list[dict]:
    """Cosine-loss training. v1/v2 default. Penalizes distance to target
    but doesn't force ranking — explains v2's 'right answer in top-3
    but loses by 0.01 cosine' failure mode."""
    opt = torch.optim.AdamW(op.parameters(), lr=lr, weight_decay=weight_decay)
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
                "step": step, "loss": float(loss.item()),
                "metric": float(cos_mean), "metric_name": "cos_train",
            })
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return history


def train_operator_contrastive(
    op: RelationalOperator,
    z_entity: torch.Tensor,
    train_value_pool: torch.Tensor,           # (P, D) — encoded unique training values
    train_value_idx: torch.Tensor,            # (N,) — index into pool for each pair
    axis_idx: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    log_every: int,
    weight_decay: float = 0.0,
    temperature: float = 0.05,
    axis_value_mask: torch.Tensor | None = None,    # (num_axes, P), True = valid negative
) -> list[dict]:
    """InfoNCE / NT-Xent contrastive training with optional per-axis
    hard-negative masking.

    For each (entity, axis, value) pair, the operator's prediction must
    rank its TRUE value above all OTHER candidate values.

    When `axis_value_mask` is provided: negatives are restricted to
    values that appear in the SAME axis in training. This is the
    standard KG-embedding / supervised-SimCSE pattern (research-validated
    2026-05-06): random cross-axis negatives are trivially easy
    ('meow' is far from 'blue'); hard within-axis negatives ('yellow'
    vs 'blue' for color queries) force the operator to learn fine-
    grained distinctions.

    When `axis_value_mask=None`: falls back to all-training-values as
    negatives (v3 behavior).

    Loss:
        logits = (z_pred_norm @ pool_norm.T) / temperature
        if hard negatives: mask out cross-axis values to -inf
        loss = CE(logits, positive_indices)
    """
    pool_norm = F.normalize(train_value_pool, p=2, dim=-1)        # (P, D)
    opt = torch.optim.AdamW(op.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict] = []
    op.train()
    for step in range(epochs):
        opt.zero_grad()
        z_pred = op(z_entity, axis_idx)                            # (N, D)
        z_pred_norm = F.normalize(z_pred, p=2, dim=-1)
        logits = z_pred_norm @ pool_norm.T / temperature           # (N, P)
        if axis_value_mask is not None:
            # Per-axis hard negatives: keep only same-axis values.
            batch_mask = axis_value_mask[axis_idx]                 # (N, P) bool
            # Mask cross-axis values to -inf so cross_entropy ignores them.
            logits = logits.masked_fill(~batch_mask, float("-inf"))
        loss = F.cross_entropy(logits, train_value_idx)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == epochs - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                top1_acc = float((preds == train_value_idx).float().mean().item())
            history.append({
                "step": step, "loss": float(loss.item()),
                "metric": top1_acc, "metric_name": "train_top1",
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
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
    pool_mean: torch.Tensor | None = None,
) -> dict:
    """Top-K retrieval over a value pool. Top-K hit if expected_value
    is among the top-K nearest pool words by cosine.

    Per the v3 fix: top-1 alone is misleading in encoder space where
    many candidates cluster at cos 0.85-0.95 with margins of 0.01-0.02.
    Top-3 and top-5 measure whether the operator landed in the right
    semantic neighborhood.

    Per the v4 fix: pass `pool_mean` (fitted on TRAINING values only)
    to subtract from BOTH eval pool and entities at eval time. Removes
    anisotropic encoder bias consistently with training.
    """
    z_pool = encode_fn(pool)                                        # (P, D)
    if pool_mean is not None:
        z_pool = z_pool - pool_mean
    z_pool_norm = F.normalize(z_pool, p=2, dim=-1)
    max_k = max(k_values)

    rows = []
    per_axis_acc: dict[str, dict] = {}
    for p in holdout_pairs:
        entity = p["entity"]
        axis = p["axis"]
        expected = p["value"]
        if axis not in vocab:
            continue
        if expected not in pool:
            continue
        axis_idx = vocab[axis]
        z_entity = encode_fn([entity]).squeeze(0).flatten()         # (D,)
        if pool_mean is not None:
            z_entity = z_entity - pool_mean.squeeze(0)
        z_pred = op(z_entity, axis_idx).flatten()
        z_pred_norm = F.normalize(z_pred.unsqueeze(0), p=2, dim=-1).squeeze(0)
        scores = z_pool_norm @ z_pred_norm                          # (P,)
        topk = torch.topk(scores, k=min(max_k, scores.numel()))
        topk_words = [pool[i] for i in topk.indices.tolist()]
        topk_scores = topk.values.tolist()

        topk_hits = {k: int(expected in topk_words[:k]) for k in k_values}

        rows.append({
            "entity": entity, "axis": axis, "expected": expected,
            "topk_words": topk_words,
            "topk_scores": topk_scores,
            **{f"top{k}_hit": topk_hits[k] for k in k_values},
        })
        agg = per_axis_acc.setdefault(
            axis,
            {"n": 0, **{f"top{k}_hit": 0 for k in k_values}},
        )
        agg["n"] += 1
        for k in k_values:
            agg[f"top{k}_hit"] += topk_hits[k]

    n_total = len(rows)
    summary = {f"n_top{k}_hit": sum(r[f"top{k}_hit"] for r in rows) for k in k_values}
    summary["rates"] = {
        f"top{k}": (summary[f"n_top{k}_hit"] / n_total if n_total else 0.0)
        for k in k_values
    }
    for axis, agg in per_axis_acc.items():
        for k in k_values:
            agg[f"top{k}_rate"] = agg[f"top{k}_hit"] / agg["n"] if agg["n"] else 0.0
    summary["n_total"] = n_total
    summary["per_axis"] = per_axis_acc
    summary["k_values"] = list(k_values)
    summary["rows"] = rows
    return summary


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
    parser.add_argument("--no-mlp", action="store_true",
                        help="Diagnostic: drop the shared MLP, use only "
                             "per-axis direction (delta = alpha · v_axis). "
                             "Confirmed in v2 to lift top-1 3%% → 20%%.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss", default="contrastive",
                        choices=["cosine", "contrastive"],
                        help="Training loss. v3 default is contrastive (InfoNCE) "
                             "for proper ranking — makes positive value RANK ABOVE "
                             "all other training values, not just be 'close to' it.")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="InfoNCE temperature. v4 default 0.05 (was 0.1) — "
                             "SimCSE-supervised default; lower temp sharpens "
                             "ranking on tightly-clustered same-axis pools.")
    parser.add_argument("--center-embeddings", action="store_true", default=True,
                        help="Subtract training-value pool mean from values + "
                             "entities. Removes anisotropic encoder bias toward "
                             "common words ('water', 'yellow', 'milk') that "
                             "dominate ranking in tight clusters. v4 default ON.")
    parser.add_argument("--no-center-embeddings",
                        dest="center_embeddings", action="store_false",
                        help="Disable centering (ablation).")
    parser.add_argument("--hard-negatives", action="store_true", default=True,
                        help="Per-axis hard negatives in contrastive loss. "
                             "Each pair's negatives are restricted to values "
                             "that appear under the SAME axis in training. "
                             "Standard KG-embedding/SimCSE pattern. v4 default ON.")
    parser.add_argument("--no-hard-negatives",
                        dest="hard_negatives", action="store_false",
                        help="Use all training values as negatives (v3 ablation).")
    parser.add_argument("--top1-min", type=float, default=0.30,
                        help="Acceptance: held-out top-1 ≥ this. Looser than v1/v2 "
                             "since encoder-geometry margins make top-1 a hard bar.")
    parser.add_argument("--top3-min", type=float, default=0.60,
                        help="Acceptance: held-out top-3 ≥ this. Top-3 is the "
                             "more informative metric in tight encoder clusters.")
    parser.add_argument("--per-axis-min", type=float, default=0.50,
                        help="Per-axis top-3 ≥ this on ≥ N axes.")
    parser.add_argument("--per-axis-pass-count", type=int, default=6,
                        help="Number of axes that must clear --per-axis-min.")
    parser.add_argument("--train-tsv", default="data/property/text_pairs_train.tsv")
    parser.add_argument("--holdout-tsv", default="data/property/text_pairs_holdout.tsv")
    parser.add_argument("--out", default="results/curriculum/tier0_property.json")
    parser.add_argument("--ckpt-root", default="data/property/checkpoints",
                        help="Save trained PropertyOperatorPackage under this dir. "
                             "Loaded by Path B demo for end-to-end inference.")
    parser.add_argument("--save-on-pass-only", action="store_true",
                        help="If set, only save the package when verdict is PASS.")
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
    use_mlp = not args.no_mlp
    arch_label = "MLP-on" if use_mlp else "linear-only (no MLP)"
    print(f"\n[4] Training RelationalOperator [{arch_label}] "
          f"({args.epochs} epochs, lr={args.lr}, "
          f"weight_decay={args.weight_decay})")
    print("-" * 78)
    torch.manual_seed(args.seed)
    op = RelationalOperator(
        dim=DIM, num_axes=len(vocab), mlp_hidden=args.mlp_hidden,
        use_mlp=use_mlp,
    ).to(args.device)
    n_params = sum(p.numel() for p in op.parameters())
    print(f"  RelationalOperator: {n_params/1e3:.1f}K params, "
          f"{len(vocab)} axes, mode={arch_label}")
    if not use_mlp:
        print(f"  (linear-only diagnostic — cannot memorize entity→value; "
              f"forces axis-as-direction learning)")
    # Pool mean (for centering) — fitted ON THE TRAINING VALUE POOL ONLY.
    # This removes anisotropic encoder bias without leaking holdout statistics.
    pool_mean: torch.Tensor | None = None

    if args.loss == "contrastive":
        # Build the training value pool (unique training values only —
        # no holdout leakage). Each pair's value must map to a pool index.
        train_value_pool = sorted({p["value"] for p in train_pairs})
        train_value_idx_int = [
            train_value_pool.index(p["value"]) for p in train_pairs
        ]
        train_value_idx = torch.tensor(
            train_value_idx_int, dtype=torch.long, device=args.device,
        )
        z_train_value_pool = encode(train_value_pool)
        z_entity_train = z_entity

        if args.center_embeddings:
            pool_mean = z_train_value_pool.mean(dim=0, keepdim=True)  # (1, D)
            z_train_value_pool = z_train_value_pool - pool_mean
            z_entity_train = z_entity_train - pool_mean

        # Per-axis hard-negative mask: (num_axes, P_train).
        # axis_value_mask[a, j] = True iff training value `j` appears
        # under axis `a` in the training data — the only valid negatives
        # for axis-`a` pairs.
        axis_value_mask: torch.Tensor | None = None
        if args.hard_negatives:
            n_axes = len(vocab)
            P = len(train_value_pool)
            axis_value_mask = torch.zeros(
                n_axes, P, dtype=torch.bool, device=args.device,
            )
            for p in train_pairs:
                a_idx = vocab[p["axis"]]
                v_idx = train_value_pool.index(p["value"])
                axis_value_mask[a_idx, v_idx] = True
            avg_negatives = axis_value_mask.float().sum(dim=1).mean().item() - 1
            print(f"  loss=contrastive (InfoNCE)  pool size={P}  "
                  f"temperature={args.temperature}  "
                  f"hard-negatives=ON (avg {avg_negatives:.1f} same-axis negatives/pair)")
        else:
            print(f"  loss=contrastive (InfoNCE)  pool size={len(train_value_pool)}  "
                  f"temperature={args.temperature}  hard-negatives=OFF")
        if args.center_embeddings:
            print(f"  centering=ON (pool_mean fitted on {len(train_value_pool)} train values)")
        else:
            print(f"  centering=OFF")

        history = train_operator_contrastive(
            op, z_entity_train, z_train_value_pool, train_value_idx, train_axis_idx,
            epochs=args.epochs, lr=args.lr, log_every=args.log_every,
            weight_decay=args.weight_decay, temperature=args.temperature,
            axis_value_mask=axis_value_mask,
        )
    else:
        print(f"  loss=cosine (1 - cos(z_pred, z_value))")
        history = train_operator_cosine(
            op, z_entity, z_value, train_axis_idx,
            epochs=args.epochs, lr=args.lr, log_every=args.log_every,
            weight_decay=args.weight_decay,
        )
    for h in history:
        print(f"  step {h['step']:>4}  loss={h['loss']:.4f}  "
              f"{h['metric_name']}={h['metric']:.4f}")

    # ---- Held-out evaluation -----------------------------------------
    print(f"\n[5] Held-out evaluation ({len(holdout_pairs)} truly-novel pairs)")
    print("-" * 78)
    eval_result = evaluate_holdout(
        op, encode, holdout_pairs, pool, vocab,
        pool_mean=pool_mean,    # use training-pool mean (consistent with training)
    )
    rates = eval_result["rates"]
    n_total = eval_result["n_total"]

    print(f"\n  per-axis top-K:")
    print(f"    {'axis':<12}  {'n':>3}  {'top-1':>10}  {'top-3':>10}  {'top-5':>10}")
    for axis in sorted(eval_result["per_axis"].keys()):
        agg = eval_result["per_axis"][axis]
        passing_3 = "✓" if agg["top3_rate"] >= args.per_axis_min else "✗"
        print(f"    {axis:<12}  {agg['n']:>3}  "
              f"{agg['top1_hit']:>2}/{agg['n']:<2}={agg['top1_rate']:.2f}  "
              f"{agg['top3_hit']:>2}/{agg['n']:<2}={agg['top3_rate']:.2f}{passing_3}  "
              f"{agg['top5_hit']:>2}/{agg['n']:<2}={agg['top5_rate']:.2f}")

    print(f"\n  per-pair detail (top-5):")
    print(f"    {'#':<2} {'entity':<14} {'axis':<10} {'expected':<16} "
          f"{'top1':<16} {'top1?':<5} {'top3?':<5}")
    for i, r in enumerate(eval_result["rows"]):
        m1 = "✓" if r["top1_hit"] else "✗"
        m3 = "✓" if r["top3_hit"] else "✗"
        top1 = r["topk_words"][0]
        print(f"    {i+1:<2} {r['entity']:<14} {r['axis']:<10} "
              f"{r['expected']:<16} {top1:<16} {m1:<5} {m3:<5}")
        if not r["top1_hit"]:
            top5_pairs = ", ".join(
                f"{w}({s:.3f})"
                for w, s in zip(r["topk_words"][:5], r["topk_scores"][:5])
            )
            print(f"       top-5: {top5_pairs}")

    # ---- Acceptance gate (v3: top-3 primary, top-1 secondary) -------
    rate_top1 = rates["top1"]
    rate_top3 = rates["top3"]
    rate_top5 = rates["top5"]
    n_axes_passing_top3 = sum(
        1 for axis, agg in eval_result["per_axis"].items()
        if agg["top3_rate"] >= args.per_axis_min
    )
    n_axes_total = len(eval_result["per_axis"])

    print("\n" + "=" * 78)
    print("ACCEPTANCE (v3 — top-K based)")
    print("=" * 78)
    top1_pass = rate_top1 >= args.top1_min
    top3_pass = rate_top3 >= args.top3_min
    axes_pass = n_axes_passing_top3 >= args.per_axis_pass_count
    accept = top1_pass and top3_pass and axes_pass
    print(f"  overall top-1: {eval_result['n_top1_hit']}/{n_total} "
          f"= {rate_top1:.3f}  {'✓' if top1_pass else '✗'} (target ≥ {args.top1_min})")
    print(f"  overall top-3: {eval_result['n_top3_hit']}/{n_total} "
          f"= {rate_top3:.3f}  {'✓' if top3_pass else '✗'} (target ≥ {args.top3_min})")
    print(f"  overall top-5: {eval_result['n_top5_hit']}/{n_total} "
          f"= {rate_top5:.3f}")
    print(f"  axes top-3 ≥ {args.per_axis_min}: "
          f"{n_axes_passing_top3}/{n_axes_total}  "
          f"{'✓' if axes_pass else '✗'} (target ≥ {args.per_axis_pass_count})")

    if accept:
        verdict = "TIER_0_PROPERTY_PASS"
        message = (
            f"The first reasoning operator works at the v3 acceptance bar. "
            f"Top-1 {rate_top1:.0%}, top-3 {rate_top3:.0%}, top-5 {rate_top5:.0%} "
            f"on truly-novel entities ({n_axes_passing_top3}/{n_axes_total} "
            f"axes top-3 ≥ {args.per_axis_min:.0%}). The brain has its first "
            f"relational primitive — property(ψ_entity, axis) → ψ_value, with "
            f"contrastive ranking. Proceed to Tier 0.2 (causation)."
        )
    elif rate_top3 >= 0.40:
        verdict = "TIER_0_PROPERTY_PARTIAL"
        message = (
            f"Operator partially works (top-3 {rate_top3:.0%}) but below the "
            f"{args.top3_min:.0%} gate. Inspect per-axis breakdown. Common "
            f"fixes: more pairs per weak axis, expand value pool with "
            f"distractors, increase contrastive temperature."
        )
    else:
        verdict = "TIER_0_PROPERTY_FAIL"
        message = (
            f"Operator did not generalize beyond training (top-3 {rate_top3:.0%}). "
            f"This is a deeper issue than ranking. Possible: entity-novelty in "
            f"E5 too high; per-axis directions don't generalize. Try richer "
            f"training data (50-100 pairs per axis from ConceptNet)."
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
            "top3_min": args.top3_min,
            "per_axis_min": args.per_axis_min,
            "per_axis_pass_count": args.per_axis_pass_count,
        },
        "config": {
            "loss": args.loss,
            "use_mlp": not args.no_mlp,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "center_embeddings": args.center_embeddings,
        },
        "eval": {
            "n_total": eval_result["n_total"],
            "n_top1_hit": eval_result["n_top1_hit"],
            "n_top3_hit": eval_result["n_top3_hit"],
            "n_top5_hit": eval_result["n_top5_hit"],
            "top1_rate": rate_top1,
            "top3_rate": rate_top3,
            "top5_rate": rate_top5,
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

    # Persist the trained PropertyOperatorPackage for downstream Path B
    # inference. Skipped only when running cosine ablation (no training
    # value pool was built) or when --save-on-pass-only and accept=False.
    save_pkg = (args.loss == "contrastive") and (accept or not args.save_on_pass_only)
    if save_pkg:
        ckpt_root = Path(args.ckpt_root)
        # Use the training value pool (training values only — no holdout
        # leakage; the demo will re-encode at load time).
        train_value_pool_strs = sorted({p["value"] for p in train_pairs})
        PropertyOperatorPackage.save_from_training(
            root=ckpt_root,
            op=op,
            vocab=vocab,
            pool_mean=pool_mean,
            value_pool=train_value_pool_strs,
        )
        print(f"→ saved PropertyOperatorPackage to {ckpt_root}/ "
              f"(operator.pt, vocab.json, value_pool.json"
              f"{', pool_mean.pt' if pool_mean is not None else ''})")
        print(f"  Path B demo can now load this via "
              f"PropertyOperatorPackage.load('{ckpt_root}', encode_fn)")
    else:
        if args.save_on_pass_only and not accept:
            print(f"→ NOT saving package (verdict failed and --save-on-pass-only set)")

    raise SystemExit(0 if accept else 1)


if __name__ == "__main__":
    main()
