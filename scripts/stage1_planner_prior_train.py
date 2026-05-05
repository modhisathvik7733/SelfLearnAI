"""Task 1.8 — train operator prior P(op|ψ) + integration smoke test.

Builds and trains the operator prior over the 7-concept library:
  plural, past_tense, comparative, superlative, opposite, agentive, young.

Training data is synthesized from each concept's existing
text_pairs_train.tsv: every source word becomes one (encoder embedding,
correct concept) pair. Concepts have very different counts (44 for
plural down to 3 for the few-shot ones), so the loss is
class-frequency-weighted to keep the prior from collapsing onto the
high-data classes.

Held-out evaluation: each concept's text_pairs_held_out.tsv (6 source
words per concept ⇒ 42 evaluation rows). Top-1 accuracy is the prior's
quality metric.

Plan §19.2 row 1.8:
  > 1.8 | Operator prior p(op | ψ) | search efficiency improves at
  >       matched accuracy.

So we additionally run an INTEGRATION test: re-use the agentive ∘
plural test cases from Task 1.6/1.7 (paint, drive, sing, dance, run,
help). Run the planner WITH the prior pruning operator expansion to
top-K, and confirm chain recovery + end-state correctness do not
regress relative to Task 1.7. Also report search-fanout reduction so
the efficiency claim is concrete.

Acceptance gates (Task 1.8):
  - Prior held-out top-1 accuracy ≥ 0.70 (overall across 7 concepts).
  - Planner-with-prior chain recovery ≥ 5/6 (no regression).
  - Planner-with-prior end-state correctness ≥ 5/6.

The held-out gate is a soft 0.70 because the few-shot concepts (3
training examples) realistically cap their per-class accuracy. With
4 full-data concepts at ~95% and 3 few-shot at ~50%, the weighted
overall is ~0.78. Setting the gate at 0.70 is honest about the data
shape while still rejecting a broken prior.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from selflearnai.planner import (
    BeamSearchPlanner,
    OperatorPriorMLP,
    make_prior_callable,
)
from selflearnai.concepts import ConceptOperator

# Reuse data + helpers from Task 1.6 to stay consistent.
from scripts.stage1_planner_beam_smoke import (
    CHAIN_TRIPLES,
    POOL_FAIR,
    ENCODERS,
    read_pairs,
    make_encode_fn,
    train_operator,
)


# Concept name → directory with text_pairs_train.tsv + text_pairs_held_out.tsv.
# Order is canonical and used as the prior's class index ordering.
CONCEPTS_DATA: list[tuple[str, str]] = [
    ("plural",       "data/plurality"),
    ("past_tense",   "data/past_tense"),
    ("comparative",  "data/comparative"),
    ("superlative",  "data/few_shot/superlative"),
    ("opposite",     "data/opposite_v2"),
    ("agentive",     "data/few_shot/agentive"),
    ("young",        "data/few_shot/young"),
]


def build_prior_corpus(
    concepts_data: list[tuple[str, str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Returns ({concept: train_sources}, {concept: heldout_sources}).

    Source words are the FIRST element of each pair in the concept's
    text_pairs_train.tsv / text_pairs_held_out.tsv. Targets are not
    needed for prior training.
    """
    train_by_concept: dict[str, list[str]] = {}
    heldout_by_concept: dict[str, list[str]] = {}
    for concept, ddir in concepts_data:
        train_pairs = read_pairs(Path(ddir) / "text_pairs_train.tsv")
        held_pairs = read_pairs(Path(ddir) / "text_pairs_held_out.tsv")
        train_by_concept[concept] = [p[0] for p in train_pairs]
        heldout_by_concept[concept] = [p[0] for p in held_pairs]
    return train_by_concept, heldout_by_concept


def encode_per_concept(
    encode_fn,
    src_by_concept: dict[str, list[str]],
) -> dict[str, torch.Tensor]:
    """Encode each concept's source words once. Returns {concept: (n, D)}."""
    out: dict[str, torch.Tensor] = {}
    for concept, sources in src_by_concept.items():
        if not sources:
            continue
        out[concept] = encode_fn(sources)
    return out


def train_prior(
    *,
    encoded_train: dict[str, torch.Tensor],
    operator_names: tuple[str, ...],
    encoder_dim: int,
    hidden_dim: int,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> OperatorPriorMLP:
    """Train the prior with class-frequency-weighted cross-entropy.

    Class weights inversely proportional to per-class sample count
    (sklearn convention: weight = N_total / (n_classes * n_class)).
    Without this the few-shot classes would be ignored at the loss
    level by the high-data classes.
    """
    torch.manual_seed(seed)
    name_to_idx = {n: i for i, n in enumerate(operator_names)}

    # Build flat training tensors.
    z_list: list[torch.Tensor] = []
    y_list: list[int] = []
    for concept in operator_names:
        if concept not in encoded_train:
            continue
        z = encoded_train[concept]
        z_list.append(z)
        y_list.extend([name_to_idx[concept]] * z.shape[0])
    if not z_list:
        raise RuntimeError("Empty training corpus for prior")
    z_train = torch.cat(z_list, dim=0).to(device)            # (N, D)
    y_train = torch.tensor(y_list, dtype=torch.long, device=device)  # (N,)

    n_total = y_train.shape[0]
    n_classes = len(operator_names)
    counts = np.bincount(
        y_train.cpu().numpy(), minlength=n_classes,
    ).astype(float)
    # Avoid division by zero when a class has no training data; that
    # class gets weight 0 (won't contribute to loss).
    weights = np.where(
        counts > 0,
        n_total / (n_classes * counts),
        0.0,
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)

    prior = OperatorPriorMLP(
        encoder_dim=encoder_dim,
        operator_names=operator_names,
        hidden_dim=hidden_dim,
    ).to(device)
    opt = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    for _ in range(epochs):
        opt.zero_grad()
        logits = prior(z_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

    prior.freeze()
    return prior


def evaluate_prior(
    prior: OperatorPriorMLP,
    encoded_heldout: dict[str, torch.Tensor],
) -> dict:
    """Per-concept and overall top-1 accuracy on held-out source words."""
    by_concept: dict[str, dict] = {}
    n_total = 0
    n_correct_total = 0
    for concept in prior.operator_names:
        if concept not in encoded_heldout:
            continue
        z = encoded_heldout[concept]
        if z.shape[0] == 0:
            continue
        logits = prior(z)                                # (n, n_ops)
        preds = logits.argmax(dim=-1).cpu().numpy()      # (n,)
        true_idx = prior.operator_names.index(concept)
        n_correct = int((preds == true_idx).sum())
        n = z.shape[0]
        by_concept[concept] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / n if n else 0.0,
            "predictions": [prior.operator_names[i] for i in preds.tolist()],
        }
        n_total += n
        n_correct_total += n_correct
    overall = n_correct_total / n_total if n_total else 0.0
    return {
        "by_concept": by_concept,
        "n_total": n_total,
        "n_correct_total": n_correct_total,
        "overall_top1": overall,
    }


def build_concept_operators(
    encode_fn,
    concepts_data: list[tuple[str, str]],
    *,
    dim: int,
    device: str,
    seed: int,
    epochs: int,
) -> dict[str, ConceptOperator]:
    """Train one ConceptOperator per concept on its own train pairs.
    Reuses train_operator() from Task 1.6's smoke. Returns a dict
    suitable for passing to BeamSearchPlanner."""
    out: dict[str, ConceptOperator] = {}
    for concept, ddir in concepts_data:
        train_pairs = read_pairs(Path(ddir) / "text_pairs_train.tsv")
        out[concept] = train_operator(
            encode_fn, train_pairs, dim=dim, device=device,
            seed=seed, epochs=epochs,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="gte-base", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--operator-epochs", type=int, default=2000,
                        help="Epochs for each ConceptOperator (used in integration test).")
    parser.add_argument(
        "--top-k-operators", type=int, default=3,
        help="When the prior is wired into the planner for the integration "
             "test, expand only the top-K operators per state.",
    )
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--step-bonus", type=float, default=0.015)
    parser.add_argument("--prior-min", type=float, default=0.70)
    parser.add_argument("--chain-min", type=int, default=5)
    parser.add_argument("--end-state-min", type=int, default=5)
    parser.add_argument("--out", default="results/stage1/planner_prior.json")
    parser.add_argument("--save-ckpt", default="checkpoints/operator_prior.pt")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Task 1.8 — operator prior P(op|ψ) trainer + integration test")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Concepts: {[c for c, _ in CONCEPTS_DATA]}")

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Build prior corpus ----
    train_by_concept, heldout_by_concept = build_prior_corpus(CONCEPTS_DATA)
    print("\nPrior training corpus (per-concept source-word counts):")
    for concept, _ in CONCEPTS_DATA:
        n_train = len(train_by_concept.get(concept, []))
        n_held = len(heldout_by_concept.get(concept, []))
        print(f"  {concept:<14s}  train={n_train:<3d}  held-out={n_held}")

    print("\nEncoding source words ...")
    encoded_train = encode_per_concept(encode, train_by_concept)
    encoded_heldout = encode_per_concept(encode, heldout_by_concept)

    # ---- Train prior ----
    operator_names = tuple(c for c, _ in CONCEPTS_DATA)
    print(f"\nTraining prior MLP (hidden={args.hidden_dim}, epochs={args.epochs}) ...")
    prior = train_prior(
        encoded_train=encoded_train,
        operator_names=operator_names,
        encoder_dim=enc_cfg["dim"],
        hidden_dim=args.hidden_dim,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    print(f"  prior params: {prior.num_parameters():,}")

    # ---- Evaluate prior ----
    eval_report = evaluate_prior(prior, encoded_heldout)
    print("\n" + "=" * 78)
    print("PRIOR HELD-OUT ACCURACY  (top-1 over 7 operators)")
    print("=" * 78)
    print(f"  {'concept':<14s}  {'n':>3s}  {'correct':>7s}  {'acc':>6s}")
    for concept in operator_names:
        d = eval_report["by_concept"].get(concept)
        if d is None:
            continue
        print(
            f"  {concept:<14s}  {d['n']:>3d}  {d['n_correct']:>7d}  {d['accuracy']:>6.2f}"
        )
    print(
        f"  {'-' * 14}  {'-' * 3}  {'-' * 7}  {'-' * 6}\n"
        f"  {'overall':<14s}  {eval_report['n_total']:>3d}  "
        f"{eval_report['n_correct_total']:>7d}  {eval_report['overall_top1']:>6.2f}"
    )

    # ---- Integration: planner WITH prior on agentive ∘ plural test ----
    print("\n" + "=" * 78)
    print(f"INTEGRATION TEST: planner with prior pruning (top_k={args.top_k_operators})")
    print("=" * 78)
    print("Training all 7 ConceptOperators for the planner ...")
    operators = build_concept_operators(
        encode, CONCEPTS_DATA, dim=enc_cfg["dim"], device=args.device,
        seed=args.seed, epochs=args.operator_epochs,
    )

    # Planner WITH prior: top_k operators per state by prior probability.
    planner = BeamSearchPlanner(
        operators=operators,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        step_bonus=args.step_bonus,
        prior=make_prior_callable(prior),
        top_k_operators=args.top_k_operators,
    )

    # Encode FAIR pool once.
    z_pool = encode(POOL_FAIR)
    pool_n = F.normalize(z_pool, dim=-1)

    expected_chain = ("agentive", "plural")
    chain_correct = 0
    end_state_correct = 0
    case_records: list[dict] = []
    n_ops = len(operators)
    print(
        f"  Operator-fanout reduction: {args.top_k_operators}/{n_ops} ops per "
        f"state (× across {args.max_depth} depths = "
        f"{(n_ops/args.top_k_operators)**args.max_depth:.1f}× full-search)"
    )
    print()
    for verb, _, plural_agent in CHAIN_TRIPLES:
        psi_start = encode([verb]).squeeze(0)
        psi_goal = encode([plural_agent]).squeeze(0)
        result = planner.search(psi_start, psi_goal)

        pred_n = F.normalize(result.psi.unsqueeze(0), dim=-1)
        sims = pred_n @ pool_n.T
        top1_idx = int(sims.argmax(dim=-1).item())
        top1_word = POOL_FAIR[top1_idx]
        end_state_pass = top1_word == plural_agent
        chain_pass = result.chain == expected_chain
        if chain_pass:
            chain_correct += 1
        if end_state_pass:
            end_state_correct += 1

        # What did the prior say about the start state? Surface for diagnosis.
        prior_top3 = prior.predict_top_k(psi_start, k=3)
        prior_str = ", ".join(f"{n}:{p:.2f}" for n, p in prior_top3)

        chain_repr = " ∘ ".join(result.chain) if result.chain else "(no-op)"
        c_mark = "✓" if chain_pass else "✗"
        e_mark = "✓" if end_state_pass else "✗"
        print(
            f"  {verb:<6}→ {plural_agent:<10}  "
            f"{c_mark} chain: {chain_repr:<28}  "
            f"{e_mark} top-1: {top1_word:<10}  "
            f"cos={result.cos_to_goal:+.3f}"
        )
        print(f"        prior@start: {prior_str}")

        case_records.append({
            "verb": verb,
            "expected_target": plural_agent,
            "chain": list(result.chain),
            "chain_correct": chain_pass,
            "top1_pool_word": top1_word,
            "end_state_correct": end_state_pass,
            "cos_to_goal": result.cos_to_goal,
            "score": result.score,
            "prior_top3": prior_top3,
        })

    # ---- Acceptance ----
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.8)")
    print("=" * 78)
    prior_pass = eval_report["overall_top1"] >= args.prior_min
    chain_pass_overall = chain_correct >= args.chain_min
    end_state_pass_overall = end_state_correct >= args.end_state_min

    print(
        f"  Prior overall top-1:        {eval_report['overall_top1']:.3f}  "
        f"(gate >= {args.prior_min:.2f})  "
        f"→ {'PASS' if prior_pass else 'FAIL'}"
    )
    print(
        f"  Chain recovery (with prior): {chain_correct}/6  "
        f"(gate >= {args.chain_min})  "
        f"→ {'PASS' if chain_pass_overall else 'FAIL'}"
    )
    print(
        f"  End-state correctness:       {end_state_correct}/6  "
        f"(gate >= {args.end_state_min})  "
        f"→ {'PASS' if end_state_pass_overall else 'FAIL'}"
    )
    overall = prior_pass and chain_pass_overall and end_state_pass_overall
    print(f"\n→ Task 1.8: {'PASS' if overall else 'FAIL'}")

    # ---- Save checkpoint + JSON ----
    ckpt_path = Path(args.save_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": args.encoder,
            "encoder_dim": enc_cfg["dim"],
            "hidden_dim": args.hidden_dim,
            "operator_names": list(prior.operator_names),
            "state_dict": prior.state_dict(),
        },
        ckpt_path,
    )

    payload = {
        "task": "1.8",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "hidden_dim": args.hidden_dim,
        "operator_names": list(operator_names),
        "training_corpus": {
            c: len(train_by_concept.get(c, [])) for c in operator_names
        },
        "held_out_corpus": {
            c: len(heldout_by_concept.get(c, [])) for c in operator_names
        },
        "prior_eval": {
            k: v for k, v in eval_report.items() if k != "by_concept"
        }
        | {
            "by_concept": {
                c: {kk: vv for kk, vv in d.items() if kk != "predictions"}
                for c, d in eval_report["by_concept"].items()
            },
        },
        "integration": {
            "top_k_operators": args.top_k_operators,
            "max_depth": args.max_depth,
            "step_bonus": args.step_bonus,
            "n_ops_total": n_ops,
            "fanout_reduction": (n_ops / args.top_k_operators) ** args.max_depth,
            "cases": case_records,
            "chain_correct": chain_correct,
            "end_state_correct": end_state_correct,
        },
        "thresholds": {
            "prior_min": args.prior_min,
            "chain_min": args.chain_min,
            "end_state_min": args.end_state_min,
        },
        "pass": overall,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved checkpoint to {ckpt_path}")
    print(f"→ saved JSON to {out_path}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
