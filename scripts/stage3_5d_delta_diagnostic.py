"""Stage 3 / Sub-task 3.5d — δ-source / δ-magnitude diagnostic.

3.5c trained the decoder co-trained on (h_sing + δ_gt → plural_target). Stream C
NLL converged to ~0 (memorized) but inference (apply OPERATOR's δ_op to a
novel subject and decode) hit 0/8 plural surface text. Oracle hit 7/8.

Open question: is the failure (a) decoder didn't GENERALIZE the transformation
beyond training subjects, or (b) decoder DID generalize but operator's δ_op
is too far from training-time δ_gt for novel inputs to trigger the
transformation behavior?

This diagnostic tests three δ-sources at inference, NO new training:

  delta_gt_novel  = ψ(plural_ref_novel) - ψ(singular_novel)   ← oracle δ
  delta_op_novel  = operator(ψ_sing) - ψ_sing                 ← inference δ
  delta_op × α    = scaled inference δ at α ∈ {1, 2, 3, 5}    ← magnitude sweep

Setup: load the 3.5c-trained decoder + the 3.5b sentence-trained operator.
For each of the 8 truly-novel subjects, run all five settings, decode, and
score subject-pluralization in output.

Verdict:
  - delta_gt PASSES (≥4/8) but delta_op fails:
      decoder generalized; bottleneck is operator δ being too noisy on
      novel inputs. Fix: train stream C with operator-predicted δ
      (matches inference distribution) — that's 3.5e.
  - delta_gt also FAILS:
      decoder didn't generalize, only memorized the 93 training transforms.
      Need more transformation training pairs OR a different recipe
      (per-position δ instead of broadcast, etc.).
  - Some α-scaled delta_op PASSES but α=1 fails:
      magnitude was the issue. Just rescale at inference.

~3 min on GPU (no training).

Run on the GPU box:
  python scripts/stage3_5d_delta_diagnostic.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.generator import PointerSeqCondDecoder

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn
from scripts.stage3_5_cross_domain_compose import (
    TEST_PAIRS,
    _words_in,
    has_any,
    encode_activations,
    decode_h,
)
from scripts.stage3_5b_sentence_op import (
    OP_TRAIN_PAIRS,
    OP_TRAIN_TEMPLATES_SING_PLUR,
    build_op_train_sentence_pairs,
    train_sentence_operator,
    assert_no_test_subject_leakage,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--op-epochs", type=int, default=3000)
    parser.add_argument("--op-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-ckpt",
                        default="data/explanations_v2/checkpoints/decoder_3_5c.pt")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--scales", type=str, default="1.0,1.5,2.0,3.0,5.0",
                        help="comma-separated δ-scales to sweep on operator δ")
    parser.add_argument("--out", default="results/stage3/cross_domain_diagnostic.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Stage 3 / Sub-task 3.5d — δ-source / δ-magnitude diagnostic")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    assert_no_test_subject_leakage()
    scales = [float(x) for x in args.scales.split(",")]
    print(f"  δ-scale sweep: {scales}")

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]
    encode_pooled = make_encode_fn(mdl, tok, args.device)

    # ---- Train sentence-op -------------------------------------------
    sent_pairs = build_op_train_sentence_pairs()
    print(f"\n[1] Train sentence-level plural operator ({args.op_epochs} epochs)")
    print("-" * 78)
    plural_op, op_history = train_sentence_operator(
        encode_pooled, sent_pairs, dim=DIM, device=args.device,
        seed=args.seed, epochs=args.op_epochs, lr=args.op_lr,
    )
    print(f"  final cos(op→tgt) on training: {op_history[-1]['cos']:.4f}")

    # ---- Load 3.5c decoder -------------------------------------------
    print(f"\n[2] Load 3.5c co-trained decoder")
    print("-" * 78)
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size,
        n_layers=args.n_layers, n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
    ).to(args.device)
    decoder_path = Path(args.decoder_ckpt)
    if not decoder_path.exists():
        raise SystemExit(f"FATAL: 3.5c decoder not found at {decoder_path}. "
                         f"Run scripts/stage3_5c_cotrained_decoder.py first.")
    sd = torch.load(str(decoder_path), map_location=args.device)
    decoder.load_state_dict(sd)
    decoder.eval()
    print(f"  loaded {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params")

    # ---- Build novel sing + plur-ref --------------------------------
    n = len(TEST_PAIRS)
    sing_sents = [f"{p['subj_sing']} is a {p['cat_sing']}" for p in TEST_PAIRS]
    plur_refs = [f"{p['subj_plur'][0]} are {p['cat_plur'][0]}" for p in TEST_PAIRS]
    h_sing, mask_sing, ids_sing = encode_activations(
        sing_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    h_pref, mask_pref, ids_pref = encode_activations(
        plur_refs, tok, mdl, args.device, t_max=args.t_max,
    )
    psi_sing = encode_pooled(sing_sents)
    psi_pref = encode_pooled(plur_refs)

    # Compute deltas
    with torch.no_grad():
        psi_op = plural_op(psi_sing)
    delta_op = psi_op - psi_sing
    delta_gt = psi_pref - psi_sing
    real_mask = (mask_sing > 0).unsqueeze(-1).float()

    # δ magnitude comparison
    mag_op = delta_op.norm(dim=-1)
    mag_gt = delta_gt.norm(dim=-1)
    print(f"\n[3] δ-magnitude comparison (per-novel-subject)")
    print("-" * 78)
    print(f"  {'subj':<11} {'‖δ_op‖':>8} {'‖δ_gt‖':>8} {'op/gt':>7}")
    for i, p in enumerate(TEST_PAIRS):
        print(f"  {p['subj_sing']:<11} {mag_op[i].item():>8.3f} "
              f"{mag_gt[i].item():>8.3f} "
              f"{(mag_op[i]/mag_gt[i]).item():>7.3f}")
    print(f"  mean ratio: {(mag_op/mag_gt).mean().item():.3f}")
    print(f"  cos(δ_op, δ_gt) per pair:")
    cos_d = F.cosine_similarity(delta_op, delta_gt, dim=-1)
    for i, p in enumerate(TEST_PAIRS):
        print(f"    {p['subj_sing']:<11}: {cos_d[i].item():.4f}")
    print(f"  mean cos(δ_op, δ_gt): {cos_d.mean().item():.4f}")

    # ---- Run diagnostic settings ------------------------------------
    print(f"\n[4] Decode under five settings")
    print("-" * 78)

    def apply_and_decode(delta: torch.Tensor) -> list[str]:
        h_perturbed = h_sing + delta.unsqueeze(1) * real_mask
        return decode_h(decoder, tok, h_perturbed, mask_sing, ids_sing)

    settings: dict[str, list[str]] = {}

    # Baseline (no operator)
    settings["baseline"] = decode_h(decoder, tok, h_sing, mask_sing, ids_sing)
    # Oracle (encode plural-ref directly)
    settings["oracle"] = decode_h(decoder, tok, h_pref, mask_pref, ids_pref)
    # Ground-truth δ on novel subjects
    settings["delta_gt_novel"] = apply_and_decode(delta_gt)
    # Operator's δ at α=1 (matches 3.5c)
    settings["delta_op_x1"] = apply_and_decode(delta_op)
    # Operator's δ at α-sweep
    for s in scales:
        if s == 1.0:
            continue
        settings[f"delta_op_x{s:g}"] = apply_and_decode(s * delta_op)

    # ---- Score per setting ------------------------------------------
    print(f"\n[5] Subject-plural counts per setting")
    print("-" * 78)
    summary = {}
    for name, texts in settings.items():
        n_subj = sum(
            1 for i, p in enumerate(TEST_PAIRS)
            if has_any(_words_in(texts[i]), p["subj_plur"])
        )
        n_cat = sum(
            1 for i, p in enumerate(TEST_PAIRS)
            if has_any(_words_in(texts[i]), p["cat_plur"])
        )
        summary[name] = {"subj_plur": n_subj, "cat_plur": n_cat}
        print(f"  {name:<22}  subj_plur: {n_subj}/{n}  cat_plur: {n_cat}/{n}")

    # ---- Per-pair table for the most informative settings ----------
    print(f"\n[6] Per-pair text comparison (key settings)")
    print("-" * 78)
    print(f"  {'#':<2} {'subj':<11} {'baseline':<26} {'delta_gt':<26} {'delta_op_x1':<26}")
    rows = []
    for i, p in enumerate(TEST_PAIRS):
        b = settings["baseline"][i]
        gt = settings["delta_gt_novel"][i]
        op1 = settings["delta_op_x1"][i]
        ok_gt = has_any(_words_in(gt), p["subj_plur"])
        ok_op1 = has_any(_words_in(op1), p["subj_plur"])
        rows.append({
            "subj_sing": p["subj_sing"],
            "subj_plur_options": p["subj_plur"],
            "all_settings": {name: settings[name][i] for name in settings},
            "delta_gt_subj_plur": ok_gt,
            "delta_op_subj_plur": ok_op1,
        })
        gtm = "✓" if ok_gt else " "
        opm = "✓" if ok_op1 else " "
        print(f"  {i+1:<2} {p['subj_sing']:<11} {b:<24.24}  "
              f"{gtm} {gt:<24.24}  {opm} {op1:<24.24}")

    # ---- Verdict ----------------------------------------------------
    n_gt = summary["delta_gt_novel"]["subj_plur"]
    n_op1 = summary["delta_op_x1"]["subj_plur"]
    best_op_scale_n = max(
        summary[k]["subj_plur"] for k in summary
        if k.startswith("delta_op_x")
    )
    best_op_scale = max(
        (k for k in summary if k.startswith("delta_op_x")),
        key=lambda k: summary[k]["subj_plur"],
    )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  baseline                : {summary['baseline']['subj_plur']}/{n}")
    print(f"  oracle (plural-ref h)   : {summary['oracle']['subj_plur']}/{n}")
    print(f"  delta_gt_novel          : {n_gt}/{n}  ← did decoder generalize?")
    print(f"  delta_op_x1             : {n_op1}/{n}")
    print(f"  best operator scale     : {best_op_scale} → {best_op_scale_n}/{n}")

    if n_gt >= 4:
        if best_op_scale_n >= 4:
            verdict = "STAGE_3_5_DIAGNOSTIC_OP_SCALING_FIXES_IT"
            message = (
                f"δ_gt_novel passes ({n_gt}/{n}) AND scaling operator δ "
                f"to {best_op_scale} also passes ({best_op_scale_n}/{n}). "
                f"Decoder DID generalize the transformation. The fix is "
                f"inference-time δ-scaling, no retraining needed. "
                f"Architectural lesson: operator-predicted δ has correct "
                f"direction but undersized magnitude on novel inputs."
            )
        else:
            verdict = "STAGE_3_5_DIAGNOSTIC_OP_DISTRIBUTION_GAP"
            message = (
                f"δ_gt_novel passes ({n_gt}/{n}) but operator δ at no scale "
                f"reaches the gate (best {best_op_scale_n}/{n} at "
                f"{best_op_scale}). Decoder generalized the transform but "
                f"operator's δ_op direction is too noisy. Fix: train stream "
                f"C with operator-predicted δ (matches inference dist) — "
                f"3.5e."
            )
    else:
        verdict = "STAGE_3_5_DIAGNOSTIC_DECODER_MEMORIZED_NOT_GENERALIZED"
        message = (
            f"Even with ground-truth δ_gt on novel subjects, decoder "
            f"produces only {n_gt}/{n} plural surface forms. The 3.5c "
            f"co-trained decoder MEMORIZED the 93 training transforms "
            f"but did NOT generalize the transformation rule. Fix needs "
            f"more transformation training pairs (more subjects), per-"
            f"position δ instead of broadcast, or a different recipe."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    payload = {
        "task": "3.5d",
        "encoder": args.encoder, "encoder_dim": DIM,
        "n_test_cases": n,
        "delta_magnitudes": {
            "mag_op": [float(x) for x in mag_op],
            "mag_gt": [float(x) for x in mag_gt],
            "ratio_op_over_gt": [float(x) for x in (mag_op/mag_gt)],
            "mean_ratio": float((mag_op/mag_gt).mean()),
        },
        "delta_directions": {
            "cos_op_gt": [float(x) for x in cos_d],
            "mean_cos_op_gt": float(cos_d.mean()),
        },
        "summary": summary,
        "best_op_scale": best_op_scale,
        "best_op_scale_count": best_op_scale_n,
        "verdict": verdict,
        "message": message,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")


if __name__ == "__main__":
    main()
