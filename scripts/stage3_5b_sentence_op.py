"""Stage 3 / Sub-task 3.5b — fix 3.5 by training the plural operator on
SENTENCE pairs (in the manifold it will be applied in), not word pairs.

3.5 (commit fcded6a) failed because the word-pair plural operator's δ
moves sentence-pooled ψ in the WRONG direction in sentence space:

  cos(operated_psi, plural_ref_psi):   0.9506   ← operated
  cos(no-op_psi,    plural_ref_psi):   0.9652   ← baseline
  lift:                               -0.0146

Word-encoder geometry vs sentence-encoder geometry are different even
though both come from the same E5: word-ψ is dominated by lexical
content; sentence-ψ is dominated by template/syntax structure ("X is a
Y"). A δ learned in one manifold doesn't transfer to the other.

Fix: train the plural operator on SENTENCE pairs (singular definitional
"X is a Y" / plural definitional "Xs are Ys"), so the δ is learned in
the same encoder geometry it will be applied in. Operator architecture
unchanged. Test on 3.1's 8 truly-novel subjects (mango, peach, ...) —
held out from operator training, so this is still a genuine transfer
test on novel content.

This addresses the plan §13 #4 failure mode (encoder topology limit)
with the principled remedy: train the operator in the destination
manifold. NOT an architecture change — a training-data change.

Acceptance gate (same as 3.5 per plan §19.17):
  ≥ 50% of operated outputs contain plural-form subject.

Run on the GPU box (~5-10 min):
  python scripts/stage3_5b_sentence_op.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.concepts import ConceptOperator
from selflearnai.generator import PointerSeqCondDecoder

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn
from scripts.stage3_5_cross_domain_compose import (
    TEST_PAIRS,
    _words_in,
    has_any,
    encode_activations,
    decode_h,
)


# ---------------------------------------------------------------------------
# Sentence-pair training set for the plural operator.
#
# Drawn from Stage 3.1's TRAIN_PAIRS, restricted to subjects/categories
# with clean regular pluralizations. The 8 truly-novel subjects from
# TEST_PAIRS (mango, peach, cabbage, penguin, bookcase, trumpet,
# helicopter, church) are NOT in this set — that's the held-out
# transfer test.
#
# 30 pairs × 3 templates = 90 sentence-pair training examples for the
# operator. Same encoder geometry as where the operator will be applied.
# ---------------------------------------------------------------------------

OP_TRAIN_PAIRS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    # (subject_sing, subject_plur), (category_sing, category_plur)
    (("apple",     "apples"),    ("fruit",      "fruits")),
    (("banana",    "bananas"),   ("fruit",      "fruits")),
    (("orange",    "oranges"),   ("fruit",      "fruits")),
    (("grape",     "grapes"),    ("fruit",      "fruits")),
    (("lemon",     "lemons"),    ("fruit",      "fruits")),
    (("carrot",    "carrots"),   ("vegetable",  "vegetables")),
    (("potato",    "potatoes"),  ("vegetable",  "vegetables")),
    (("tiger",     "tigers"),    ("animal",     "animals")),
    (("dolphin",   "dolphins"),  ("animal",     "animals")),
    (("eagle",     "eagles"),    ("animal",     "animals")),
    (("rabbit",    "rabbits"),   ("animal",     "animals")),
    (("snake",     "snakes"),    ("animal",     "animals")),
    (("chair",     "chairs"),    ("furniture",  "furnitures")),
    (("table",     "tables"),    ("furniture",  "furnitures")),
    (("sofa",      "sofas"),     ("furniture",  "furnitures")),
    (("desk",      "desks"),     ("furniture",  "furnitures")),
    (("bed",       "beds"),      ("furniture",  "furnitures")),
    (("piano",     "pianos"),    ("instrument", "instruments")),
    (("guitar",    "guitars"),   ("instrument", "instruments")),
    (("drum",      "drums"),     ("instrument", "instruments")),
    (("violin",    "violins"),   ("instrument", "instruments")),
    (("flute",     "flutes"),    ("instrument", "instruments")),
    (("car",       "cars"),      ("vehicle",    "vehicles")),
    (("bus",       "buses"),     ("vehicle",    "vehicles")),
    (("plane",     "planes"),    ("vehicle",    "vehicles")),
    (("train",     "trains"),    ("vehicle",    "vehicles")),
    (("bicycle",   "bicycles"),  ("vehicle",    "vehicles")),
    (("house",     "houses"),    ("building",   "buildings")),
    (("school",    "schools"),   ("building",   "buildings")),
    (("library",   "libraries"), ("building",   "buildings")),
    (("museum",    "museums"),   ("building",   "buildings")),
]

OP_TRAIN_TEMPLATES_SING_PLUR: list[tuple[str, str]] = [
    ("{subj_s} is a {cat_s}",            "{subj_p} are {cat_p}"),
    ("a {subj_s} is a {cat_s}",          "{subj_p} are {cat_p}"),
    ("the {subj_s} is a {cat_s}",        "the {subj_p} are {cat_p}"),
]


def build_op_train_sentence_pairs() -> list[tuple[str, str]]:
    """Returns list of (singular_sentence, plural_sentence) pairs."""
    pairs: list[tuple[str, str]] = []
    for (subj_s, subj_p), (cat_s, cat_p) in OP_TRAIN_PAIRS:
        for tpl_s, tpl_p in OP_TRAIN_TEMPLATES_SING_PLUR:
            sing = tpl_s.format(subj_s=subj_s, cat_s=cat_s)
            plur = tpl_p.format(subj_p=subj_p, cat_p=cat_p)
            pairs.append((sing, plur))
    return pairs


def assert_no_test_subject_leakage() -> None:
    """The 8 truly-novel test subjects must NOT appear in the operator
    training set."""
    train_subjects = {sp[0] for sp, _ in OP_TRAIN_PAIRS}
    test_subjects = {p["subj_sing"] for p in TEST_PAIRS}
    leaks = train_subjects & test_subjects
    if leaks:
        print(f"FATAL: test subjects leak into operator training: {leaks}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Sentence-level operator training (mirrors train_operator but on sentence ψ)
# ---------------------------------------------------------------------------

def train_sentence_operator(
    encode_fn,
    sent_pairs: list[tuple[str, str]],
    *,
    dim: int,
    device: str,
    seed: int,
    epochs: int = 3000,
    lr: float = 1e-3,
) -> ConceptOperator:
    torch.manual_seed(seed)
    op = ConceptOperator(dim=dim).to(device)
    opt = torch.optim.AdamW(op.parameters(), lr=lr)
    z_src = encode_fn([p[0] for p in sent_pairs])
    z_tgt = encode_fn([p[1] for p in sent_pairs])
    history = []
    for step in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(op(z_src), z_tgt)
        loss.backward()
        opt.step()
        if step % max(epochs // 6, 200) == 0 or step == epochs - 1:
            with torch.no_grad():
                cos = F.cosine_similarity(op(z_src), z_tgt, dim=-1).mean().item()
            history.append({"step": step, "loss": float(loss.item()), "cos": cos})
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decoder-ckpt",
                        default="data/explanations_v2/checkpoints/decoder_3a1.pt")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--subj-plural-min", type=float, default=0.50)
    parser.add_argument("--out", default="results/stage3/cross_domain_compose_v2.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Stage 3 / Sub-task 3.5b — sentence-trained plural operator")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    assert_no_test_subject_leakage()

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]
    encode_pooled = make_encode_fn(mdl, tok, args.device)

    # ---- Build sentence-pair training set ----------------------------
    sent_pairs = build_op_train_sentence_pairs()
    print(f"\n[1] Sentence-pair training set: {len(sent_pairs)} pairs "
          f"({len(OP_TRAIN_PAIRS)} subjects × {len(OP_TRAIN_TEMPLATES_SING_PLUR)} templates)")
    print("-" * 78)
    for i in range(3):
        print(f"  {sent_pairs[i][0]!r:>40}  →  {sent_pairs[i][1]!r}")
    print(f"  ... ({len(sent_pairs) - 3} more)")
    test_subjects = {p["subj_sing"] for p in TEST_PAIRS}
    print(f"\n  held-out test subjects (NOT in op training): "
          f"{sorted(test_subjects)}")

    # ---- Train sentence-level plural operator -----------------------
    print(f"\n[2] Training plural operator on sentence pairs "
          f"({args.epochs} epochs, lr={args.lr})")
    print("-" * 78)
    plural_op, history = train_sentence_operator(
        encode_pooled, sent_pairs, dim=DIM, device=args.device,
        seed=args.seed, epochs=args.epochs, lr=args.lr,
    )
    for h in history:
        print(f"  step {h['step']:>5}  loss={h['loss']:.4f}  cos(op→tgt)={h['cos']:.4f}")
    smoke_src = encode_pooled([p[0] for p in sent_pairs[:5]])
    smoke_tgt = encode_pooled([p[1] for p in sent_pairs[:5]])
    smoke_op = plural_op(smoke_src)
    cos_train_op = F.cosine_similarity(smoke_op, smoke_tgt, dim=-1).mean().item()
    cos_train_noop = F.cosine_similarity(smoke_src, smoke_tgt, dim=-1).mean().item()
    print(f"\n  smoke (5 train pairs):  cos(op→tgt)   = {cos_train_op:.4f}")
    print(f"                            cos(no-op→tgt)= {cos_train_noop:.4f}")
    print(f"                            lift           = {cos_train_op - cos_train_noop:+.4f}")

    # ---- Load Stage 3.1 decoder --------------------------------------
    decoder_path = Path(args.decoder_ckpt)
    if not decoder_path.exists():
        raise SystemExit(f"FATAL: Stage 3.1 decoder not found at {decoder_path}")
    print(f"\n[3] Loading Stage 3.1 definitional decoder")
    print("-" * 78)
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size,
        n_layers=args.n_layers, n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
    ).to(args.device)
    sd = torch.load(str(decoder_path), map_location=args.device)
    decoder.load_state_dict(sd)
    decoder.eval()
    print(f"  loaded {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params")

    # ---- Test on 8 truly-novel subjects -----------------------------
    n = len(TEST_PAIRS)
    print(f"\n[4] Test set: {n} truly-novel subjects (held out from operator training)")
    print("-" * 78)
    sing_sents = [f"{p['subj_sing']} is a {p['cat_sing']}" for p in TEST_PAIRS]
    plur_refs = [f"{p['subj_plur'][0]} are {p['cat_plur'][0]}" for p in TEST_PAIRS]

    h_sing, hmask_sing, ids_sing = encode_activations(
        sing_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    h_pref, hmask_pref, ids_pref = encode_activations(
        plur_refs, tok, mdl, args.device, t_max=args.t_max,
    )
    psi_sing = encode_pooled(sing_sents)
    psi_pref = encode_pooled(plur_refs)

    # ---- Operate, propagate δ, decode -------------------------------
    print(f"\n[5] Apply operator + δ-broadcast + decode (3 settings)")
    print("-" * 78)
    with torch.no_grad():
        psi_op = plural_op(psi_sing)
        delta = psi_op - psi_sing
    real_mask = (hmask_sing > 0).unsqueeze(-1).float()
    h_op = h_sing + delta.unsqueeze(1) * real_mask

    cos_op_pref = F.cosine_similarity(psi_op, psi_pref, dim=-1)
    cos_sing_pref = F.cosine_similarity(psi_sing, psi_pref, dim=-1)
    print(f"  cos(operated_psi, plural_ref_psi):   {cos_op_pref.mean().item():.4f}")
    print(f"  cos(no-op_psi,    plural_ref_psi):   {cos_sing_pref.mean().item():.4f}")
    print(f"  lift:                                 "
          f"{(cos_op_pref - cos_sing_pref).mean().item():+.4f}  ← was -0.0146 in 3.5")

    text_baseline = decode_h(decoder, tok, h_sing, hmask_sing, ids_sing)
    text_operated = decode_h(decoder, tok, h_op, hmask_sing, ids_sing)
    text_oracle = decode_h(decoder, tok, h_pref, hmask_pref, ids_pref)

    # ---- Score per case ---------------------------------------------
    print(f"\n[6] Per-case word-fidelity")
    print("-" * 78)
    print(f"  {'#':<2} {'subj':<11} {'baseline':<28} {'operated':<28} {'oracle':<28}")
    rows = []
    for i, p in enumerate(TEST_PAIRS):
        words_b = _words_in(text_baseline[i])
        words_o = _words_in(text_operated[i])
        words_x = _words_in(text_oracle[i])
        b_subj = has_any(words_b, p["subj_plur"])
        o_subj = has_any(words_o, p["subj_plur"])
        x_subj = has_any(words_x, p["subj_plur"])
        b_cat  = has_any(words_b, p["cat_plur"])
        o_cat  = has_any(words_o, p["cat_plur"])
        x_cat  = has_any(words_x, p["cat_plur"])
        rows.append({
            "subj_sing": p["subj_sing"], "cat_sing": p["cat_sing"],
            "subj_plur_options": p["subj_plur"], "cat_plur_options": p["cat_plur"],
            "baseline_text": text_baseline[i],
            "operated_text": text_operated[i],
            "oracle_text":   text_oracle[i],
            "baseline_subj_plur": b_subj, "operated_subj_plur": o_subj, "oracle_subj_plur": x_subj,
            "baseline_cat_plur":  b_cat,  "operated_cat_plur":  o_cat,  "oracle_cat_plur":  x_cat,
        })
        b_m = "✓" if b_subj else " "
        o_m = "✓" if o_subj else " "
        x_m = "✓" if x_subj else " "
        print(f"  {i+1:<2} {p['subj_sing']:<11} "
              f"{b_m} {text_baseline[i]:<26.26} "
              f"{o_m} {text_operated[i]:<26.26} "
              f"{x_m} {text_oracle[i]:<26.26}")

    # ---- Roll-up -----------------------------------------------------
    n_baseline = sum(1 for r in rows if r["baseline_subj_plur"])
    n_operated = sum(1 for r in rows if r["operated_subj_plur"])
    n_oracle   = sum(1 for r in rows if r["oracle_subj_plur"])
    n_op_cat   = sum(1 for r in rows if r["operated_cat_plur"])
    n_op_both  = sum(1 for r in rows if r["operated_subj_plur"] and r["operated_cat_plur"])

    print("\n" + "=" * 78)
    print("ROLL-UP")
    print("=" * 78)
    print(f"  subject-plural in output:")
    print(f"    baseline (no op):                {n_baseline}/{n}")
    print(f"    operated (sentence-trained op):   {n_operated}/{n}  ← GATE METRIC")
    print(f"    oracle (encoded plural-ref h):    {n_oracle}/{n}  (upper bound)")
    print(f"  operated category-plural:           {n_op_cat}/{n}")
    print(f"  operated BOTH (subj + cat):         {n_op_both}/{n}")

    rate_op = n_operated / n
    rate_oracle = n_oracle / n
    print(f"\n  operated rate: {rate_op:.4f}  (oracle ceiling {rate_oracle:.4f})")
    print(f"  gate target:   ≥ {args.subj_plural_min}")

    if rate_op >= args.subj_plural_min:
        verdict = "STAGE_3_5_PASS"
        message = (
            f"Sentence-trained plural operator hits {n_operated}/{n} "
            f"({rate_op:.0%}) subject-plural, meeting the ≥{args.subj_plural_min:.0%} "
            f"gate. Cross-domain composition VALIDATED with the principled "
            f"fix: operators must be trained in the destination encoder "
            f"manifold (sentence-ψ here), not in a different geometry "
            f"(word-ψ in 3.5). Architectural lesson: the universal-pipeline "
            f"thesis holds, with the refinement that operator training data "
            f"must match the deployment manifold. This is a TRAINING-DATA "
            f"refinement, not an architectural change."
        )
    elif rate_op >= 0.30 and rate_op > rate_oracle * 0.7:
        verdict = "STAGE_3_5_BELOW_TARGET_BUT_NEAR_ORACLE"
        message = (
            f"Operated rate {rate_op:.4f} below gate but ≥70% of oracle "
            f"ceiling {rate_oracle:.4f}. The remaining gap is decoder "
            f"capacity (oracle is the ceiling) — the operator is doing its "
            f"job. Document and proceed; v2/CSIL can extend by training "
            f"the decoder on plural-form templates."
        )
    else:
        verdict = "STAGE_3_5_FAIL_AFTER_FIX"
        message = (
            f"Sentence-trained operator still below gate ({rate_op:.4f}). "
            f"Inspect: did operator converge (cos_train_op = "
            f"{cos_train_op:.4f})? Is oracle high enough? If oracle is "
            f"5/8 and operated is 0/8, the operator's output ψ is not "
            f"close to the manifold the decoder learned plural-form on."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    payload = {
        "task": "3.5b",
        "fix_strategy": "operator-trained-on-sentence-pairs",
        "encoder": args.encoder, "encoder_dim": DIM,
        "n_test_cases": n,
        "n_op_train_pairs": len(sent_pairs),
        "n_op_train_subjects": len(OP_TRAIN_PAIRS),
        "op_train": {
            "epochs": args.epochs,
            "history": history,
            "smoke_cos_op": cos_train_op,
            "smoke_cos_noop": cos_train_noop,
        },
        "psi_space_metrics": {
            "cos_op_pref_mean":   float(cos_op_pref.mean().item()),
            "cos_sing_pref_mean": float(cos_sing_pref.mean().item()),
            "lift": float((cos_op_pref - cos_sing_pref).mean().item()),
            "lift_3_5_for_comparison": -0.0146,
        },
        "subj_plural": {"baseline": n_baseline, "operated": n_operated, "oracle": n_oracle},
        "cat_plural_operated": n_op_cat,
        "both_plural_operated": n_op_both,
        "rate_operated_subj_plur": rate_op,
        "rate_oracle_subj_plur": rate_oracle,
        "gate_target": args.subj_plural_min,
        "verdict": verdict,
        "message": message,
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if rate_op >= args.subj_plural_min else 1)


if __name__ == "__main__":
    main()
