"""Stage 3 / Sub-task 3.5 — cross-domain operator transfer test.

Tests the SECOND architectural claim of Stage 3 (per plan §19.17):

  Stage 0/1 concept operators (e.g. plural) compose with Stage 3
  per-domain decoders. Concretely: chain `definitional ∘ plural` —
  apply the Stage 0 plural operator to a definitional sentence's ψ,
  then decode through the Stage 3.1 definitional decoder, and check
  whether the output contains plural-form subject + category words.

This is a non-trivial test:
  - The plural operator was trained on isolated word-pair embeddings
    (cat/cats, dog/dogs, ...). It learned a ψ-space DELTA, not a
    sentence-level transform.
  - The Stage 3.1 decoder was trained on SINGULAR definitional
    sentences ("apple is a fruit"). It never saw plural-form
    definitional templates ("apples are fruits") in training.
  - For the chain to work, the plural delta must (a) transfer across
    encoder geometries (single word → sentence) and (b) the decoder
    must be able to map the shifted ψ back to plural surface form
    despite never being trained on plural-form definitional text.

If the chain works, this is empirical evidence that ψ-space operators
are universal — concepts learned on one domain transfer to another
without retraining. If it fails, the universal-pipeline claim weakens
to "per-domain decoders work, but cross-domain composition needs
domain-specific operator retraining."

Acceptance gate (per plan §19.17 row 3.5):
  ≥ 50% of generations contain plural-form subject in the output text
  (the strict transfer signal — does the operator pluralize the
  generated content at all?).

If 3.5 passes: cross-domain composition validated. Universal-pipeline
thesis fully empirically supported.
If 3.5 fails: document as a v2/CSIL follow-up (per plan §3 row Stage 5
"cross-domain operator transfer measured").

Three settings reported:
  baseline  — no operator (decode h_sing directly): expected singular
  operated  — apply plural delta to all activation positions then decode
  oracle    — decode h_plural_ref (encoded directly from the plural
              reference sentence): upper bound for what the decoder
              CAN produce given the right ψ — independent of whether
              the operator can produce that ψ.

Run on the GPU box:
  python scripts/stage3_5_cross_domain_compose.py
  python scripts/stage3_5_cross_domain_compose.py --epochs 5000  # longer op train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.concepts import ConceptOperator
from selflearnai.generator import PointerSeqCondDecoder

from scripts.stage1_planner_beam_smoke import (
    ENCODERS,
    make_encode_fn,
    read_pairs,
    train_operator,
)


# ---------------------------------------------------------------------------
# Test pairs — Stage 3.1 truly-novel subjects, restricted to clearly-
# pluralizable forms. white/black/cricket/thunder dropped (ambiguous
# pluralization or uncountable noun).
# ---------------------------------------------------------------------------

TEST_PAIRS: list[dict] = [
    {"subj_sing": "mango",      "subj_plur": ["mangos", "mangoes"],
     "cat_sing": "fruit",       "cat_plur": ["fruits"]},
    {"subj_sing": "peach",      "subj_plur": ["peaches"],
     "cat_sing": "fruit",       "cat_plur": ["fruits"]},
    {"subj_sing": "cabbage",    "subj_plur": ["cabbages"],
     "cat_sing": "vegetable",   "cat_plur": ["vegetables"]},
    {"subj_sing": "penguin",    "subj_plur": ["penguins"],
     "cat_sing": "animal",      "cat_plur": ["animals"]},
    {"subj_sing": "bookcase",   "subj_plur": ["bookcases"],
     "cat_sing": "furniture",   "cat_plur": ["furnitures", "furniture"]},
    {"subj_sing": "trumpet",    "subj_plur": ["trumpets"],
     "cat_sing": "instrument",  "cat_plur": ["instruments"]},
    {"subj_sing": "helicopter", "subj_plur": ["helicopters"],
     "cat_sing": "vehicle",     "cat_plur": ["vehicles"]},
    {"subj_sing": "church",     "subj_plur": ["churches"],
     "cat_sing": "building",    "cat_plur": ["buildings"]},
]


import re

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _words_in(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def has_any(words: set[str], candidates: list[str]) -> bool:
    return any(c.lower() in words for c in candidates)


@torch.no_grad()
def encode_activations(sents, tok, mdl, device, t_max=32):
    inputs = tok(sents, padding="max_length", truncation=True,
                 max_length=t_max, return_tensors="pt").to(device)
    out = mdl(**inputs).last_hidden_state
    return out, inputs.attention_mask.float(), inputs.input_ids


@torch.no_grad()
def decode_h(decoder, tok, h, h_mask, ids, eval_bs=8):
    """Greedy decode h-batches through the decoder. Returns list of texts."""
    n = h.size(0)
    out_texts = []
    for s in range(0, n, eval_bs):
        log_probs, _, _ = decoder(h[s:s + eval_bs], h_mask[s:s + eval_bs],
                                   ids[s:s + eval_bs])
        gen = log_probs.argmax(dim=-1)
        for i in range(gen.size(0)):
            out_texts.append(tok.decode(gen[i].tolist(), skip_special_tokens=True))
    return out_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # Plural operator training
    parser.add_argument("--plural-pairs", default="data/plurality/text_pairs_train.tsv")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    # Stage 3.1 decoder
    parser.add_argument("--decoder-ckpt",
                        default="data/explanations_v2/checkpoints/decoder_3a1.pt")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    # Acceptance gate
    parser.add_argument("--subj-plural-min", type=float, default=0.50,
                        help="≥ 50% of operated outputs must contain plural subject")
    parser.add_argument("--out", default="results/stage3/cross_domain_compose.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Stage 3 / Sub-task 3.5 — cross-domain operator transfer (definitional ∘ plural)")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]
    encode_pooled = make_encode_fn(mdl, tok, args.device)

    # ---- Train plural operator on word-pair data ---------------------
    plural_pairs = read_pairs(Path(args.plural_pairs))
    print(f"\n[1] Training plural operator on {len(plural_pairs)} word pairs "
          f"({args.epochs} epochs)")
    print("-" * 78)
    plural_op = train_operator(
        encode_pooled, plural_pairs, dim=DIM, device=args.device,
        seed=args.seed, epochs=args.epochs, lr=args.lr,
    )
    # Smoke: confirm operator does something on training pairs.
    smoke_src = encode_pooled([p[0] for p in plural_pairs[:5]])
    smoke_tgt = encode_pooled([p[1] for p in plural_pairs[:5]])
    smoke_op = plural_op(smoke_src)
    cos_train = F.cosine_similarity(smoke_op, smoke_tgt, dim=-1).mean().item()
    cos_noop = F.cosine_similarity(smoke_src, smoke_tgt, dim=-1).mean().item()
    print(f"  smoke (train pairs):  cos(operated, target) = {cos_train:.4f}")
    print(f"                         cos(no-op,    target) = {cos_noop:.4f}")
    print(f"                         operator lift          = {cos_train - cos_noop:+.4f}")

    # ---- Load Stage 3.1 decoder --------------------------------------
    decoder_path = Path(args.decoder_ckpt)
    if not decoder_path.exists():
        raise SystemExit(f"FATAL: Stage 3.1 decoder not found at {decoder_path}")
    print(f"\n[2] Loading Stage 3.1 definitional decoder")
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
    print(f"  loaded {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params from {decoder_path}")

    # ---- Build singular and plural-reference test sentences ----------
    n = len(TEST_PAIRS)
    print(f"\n[3] Test set: {n} cross-domain composition cases")
    print("-" * 78)
    sing_sents = [
        f"{p['subj_sing']} is a {p['cat_sing']}" for p in TEST_PAIRS
    ]
    # Plural reference uses the SHORTEST canonical plural form for both
    # subject and category. Subjects: pick first plural form. Categories:
    # use plural-form (uncountables fall back to singular).
    plur_refs = []
    for p in TEST_PAIRS:
        s_p = p["subj_plur"][0]
        c_p = p["cat_plur"][0]
        plur_refs.append(f"{s_p} are {c_p}")
    for sing, plur in zip(sing_sents, plur_refs):
        print(f"  {sing!r}  →  {plur!r}")

    # ---- Encode all three sets ---------------------------------------
    print(f"\n[4] Encoding singular + plural-reference sentences via {args.encoder}")
    print("-" * 78)
    h_sing, hmask_sing, ids_sing = encode_activations(
        sing_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    h_pref, hmask_pref, ids_pref = encode_activations(
        plur_refs, tok, mdl, args.device, t_max=args.t_max,
    )
    psi_sing = encode_pooled(sing_sents)
    psi_pref = encode_pooled(plur_refs)

    # ---- Apply plural operator in pooled-ψ space + propagate δ -------
    print(f"\n[5] Apply plural operator in pooled-ψ space, broadcast δ to activations")
    print("-" * 78)
    with torch.no_grad():
        psi_op = plural_op(psi_sing)
        delta = psi_op - psi_sing                                     # [N, D]
    # Broadcast δ across non-pad positions of h_sing.
    real_mask = (hmask_sing > 0).unsqueeze(-1).float()                 # [N, T, 1]
    h_op = h_sing + delta.unsqueeze(1) * real_mask                     # [N, T, D]

    # Sanity: pooled(h_op) should ≈ ψ_op (since δ broadcast is the inverse
    # of the masked mean).
    psi_op_pooled = (h_op * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1.0)
    cos_pooled = F.cosine_similarity(psi_op_pooled, psi_op, dim=-1).mean().item()
    print(f"  cos(pool(h_op), psi_op) over batch: {cos_pooled:.4f}  "
          f"(should be ≈ 1.0)")
    cos_op_pref = F.cosine_similarity(psi_op, psi_pref, dim=-1)
    cos_sing_pref = F.cosine_similarity(psi_sing, psi_pref, dim=-1)
    print(f"  cos(operated_psi, plural_ref_psi): "
          f"mean {cos_op_pref.mean().item():.4f} (no-op: {cos_sing_pref.mean().item():.4f})  "
          f"lift {(cos_op_pref - cos_sing_pref).mean().item():+.4f}")

    # ---- Decode all three settings -----------------------------------
    print(f"\n[6] Decoding three settings: baseline / operated / oracle")
    print("-" * 78)
    text_baseline = decode_h(decoder, tok, h_sing, hmask_sing, ids_sing)
    text_operated = decode_h(decoder, tok, h_op, hmask_sing, ids_sing)
    text_oracle = decode_h(decoder, tok, h_pref, hmask_pref, ids_pref)

    # ---- Score ---------------------------------------------------------
    print(f"\n[7] Per-case word-fidelity (subject-plural in generated text)")
    print("-" * 78)
    print(f"  {'#':<2} {'subject_sing':<11} {'cat_sing':<10}   {'baseline':<28} {'operated':<28} {'oracle':<28}")
    rows = []
    for i, p in enumerate(TEST_PAIRS):
        words_b = _words_in(text_baseline[i])
        words_o = _words_in(text_operated[i])
        words_x = _words_in(text_oracle[i])
        # Subject-plural in output (the gate metric per §19.17)
        b_subj_plur = has_any(words_b, p["subj_plur"])
        o_subj_plur = has_any(words_o, p["subj_plur"])
        x_subj_plur = has_any(words_x, p["subj_plur"])
        # Category-plural in output (looser)
        b_cat_plur = has_any(words_b, p["cat_plur"])
        o_cat_plur = has_any(words_o, p["cat_plur"])
        x_cat_plur = has_any(words_x, p["cat_plur"])
        # Subject-singular still present (sanity)
        b_subj_sing = p["subj_sing"].lower() in words_b
        o_subj_sing = p["subj_sing"].lower() in words_o
        rows.append({
            "subj_sing": p["subj_sing"],
            "cat_sing":  p["cat_sing"],
            "subj_plur_options": p["subj_plur"],
            "cat_plur_options":  p["cat_plur"],
            "baseline_text":    text_baseline[i],
            "operated_text":    text_operated[i],
            "oracle_text":      text_oracle[i],
            "baseline_subj_plur": b_subj_plur,
            "operated_subj_plur": o_subj_plur,
            "oracle_subj_plur":   x_subj_plur,
            "baseline_cat_plur":  b_cat_plur,
            "operated_cat_plur":  o_cat_plur,
            "oracle_cat_plur":    x_cat_plur,
            "baseline_subj_sing_kept": b_subj_sing,
            "operated_subj_sing_kept": o_subj_sing,
        })
        b_mark = "✓" if b_subj_plur else " "
        o_mark = "✓" if o_subj_plur else " "
        x_mark = "✓" if x_subj_plur else " "
        print(f"  {i+1:<2} {p['subj_sing']:<11} {p['cat_sing']:<10}   "
              f"{b_mark} {text_baseline[i]:<26.26} "
              f"{o_mark} {text_operated[i]:<26.26} "
              f"{x_mark} {text_oracle[i]:<26.26}")

    # ---- Roll-up + verdict --------------------------------------------
    n_baseline_subj = sum(1 for r in rows if r["baseline_subj_plur"])
    n_operated_subj = sum(1 for r in rows if r["operated_subj_plur"])
    n_oracle_subj   = sum(1 for r in rows if r["oracle_subj_plur"])
    n_baseline_cat  = sum(1 for r in rows if r["baseline_cat_plur"])
    n_operated_cat  = sum(1 for r in rows if r["operated_cat_plur"])
    n_oracle_cat    = sum(1 for r in rows if r["oracle_cat_plur"])
    n_operated_both = sum(1 for r in rows if r["operated_subj_plur"] and r["operated_cat_plur"])
    n_oracle_both   = sum(1 for r in rows if r["oracle_subj_plur"] and r["oracle_cat_plur"])

    print("\n" + "=" * 78)
    print("ROLL-UP")
    print("=" * 78)
    print(f"  subject-plural in output:")
    print(f"    baseline (no op):              {n_baseline_subj}/{n}")
    print(f"    operated (definitional∘plural): {n_operated_subj}/{n}  ← GATE METRIC")
    print(f"    oracle (encode plural-ref):     {n_oracle_subj}/{n}  (upper bound)")
    print(f"  category-plural in output:")
    print(f"    baseline:                      {n_baseline_cat}/{n}")
    print(f"    operated:                      {n_operated_cat}/{n}")
    print(f"    oracle:                        {n_oracle_cat}/{n}")
    print(f"  BOTH (subject + category plural):")
    print(f"    operated:                      {n_operated_both}/{n}")
    print(f"    oracle:                        {n_oracle_both}/{n}")

    rate_operated_subj = n_operated_subj / n
    rate_baseline_subj = n_baseline_subj / n
    print(f"\n  operated subject-plural rate: {rate_operated_subj:.4f}")
    print(f"  gate target:                  ≥ {args.subj_plural_min}")

    if rate_operated_subj >= args.subj_plural_min:
        verdict = "STAGE_3_5_PASS"
        message = (
            f"Cross-domain composition {n_operated_subj}/{n} "
            f"({rate_operated_subj:.0%}) meets the ≥{args.subj_plural_min:.0%} "
            f"gate. The Stage 0 plural operator (trained on isolated word "
            f"pairs) successfully shifts the ψ of definitional sentences "
            f"such that the Stage 3.1 decoder generates plural-form "
            f"surface text. Universal-pipeline thesis fully validated: "
            f"per-domain decoders + concept operators COMPOSE without "
            f"per-domain operator retraining."
        )
    elif rate_operated_subj > rate_baseline_subj + 0.10:
        verdict = "STAGE_3_5_BELOW_TARGET_BUT_LIFT"
        message = (
            f"Operated rate {rate_operated_subj:.4f} below the "
            f"≥{args.subj_plural_min} gate but ≥+0.10 above the no-op "
            f"baseline {rate_baseline_subj:.4f} — the operator IS doing "
            f"something. The gap is likely either (a) ψ-space delta "
            f"trained on single-word pairs doesn't fully transfer to "
            f"sentence-pooled ψ, or (b) the decoder needs to see plural-"
            f"form templates during training. Document as a v2/CSIL "
            f"follow-up: cross-domain operator-decoder co-tuning."
        )
    else:
        verdict = "STAGE_3_5_FAIL"
        message = (
            f"Operated rate {rate_operated_subj:.4f} not meaningfully "
            f"above baseline {rate_baseline_subj:.4f}. Cross-domain "
            f"composition does NOT work with this naive δ-broadcast "
            f"approach. Inspect the oracle (decode h_pref) — if oracle "
            f"itself is low, the decoder can't produce plural-form "
            f"definitional text regardless of operator quality (out-of-"
            f"distribution generation). If oracle is high, the bottleneck "
            f"is operator δ-transfer to sentence space."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    payload = {
        "task": "3.5",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "n_test_cases": n,
        "plural_op_train": {
            "n_pairs": len(plural_pairs),
            "epochs": args.epochs,
            "smoke_cos_op": cos_train,
            "smoke_cos_noop": cos_noop,
        },
        "psi_space_metrics": {
            "cos_op_pref_mean":   float(cos_op_pref.mean().item()),
            "cos_sing_pref_mean": float(cos_sing_pref.mean().item()),
            "lift": float((cos_op_pref - cos_sing_pref).mean().item()),
        },
        "subj_plural": {
            "baseline": n_baseline_subj,
            "operated": n_operated_subj,
            "oracle":   n_oracle_subj,
        },
        "cat_plural": {
            "baseline": n_baseline_cat,
            "operated": n_operated_cat,
            "oracle":   n_oracle_cat,
        },
        "both_plural": {
            "operated": n_operated_both,
            "oracle":   n_oracle_both,
        },
        "rate_operated_subj_plur": rate_operated_subj,
        "rate_baseline_subj_plur": rate_baseline_subj,
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
    raise SystemExit(0 if rate_operated_subj >= args.subj_plural_min else 1)


if __name__ == "__main__":
    main()
