"""Stage 3 / Sub-task 3.5c — co-train the decoder on transformation pairs
to make `definitional ∘ plural` reach surface form.

3.5b (commit 6d54415) fixed the OPERATOR (sentence-trained, lift +0.0137
in ψ-space, near-perfect on training) but the decoder still produced 0/8
plural surface text. Diagnosis: the Pointer-Generator decoder copies
from input tokens; vocab branch was trained only on identity targets.
Without `apples` in the encoder input, the pointer cannot copy it, and
the vocab head was never trained to interpret a ψ-shift as "emit plural
form."

Fix: add a transformation-signal training stream to the decoder.
For each (sing, plur) sentence pair we know the ground-truth δ in
sentence-pooled-ψ space (psi_plur - psi_sing). Build three training
examples per pair:

  A. (h_sing,      target=ids_sing)   — identity singular (preserves 3.1)
  B. (h_plur,      target=ids_plur)   — identity plural (decoder sees plural-form
                                         encoder inputs as input, not just target)
  C. (h_perturbed, target=ids_plur)   — TRANSFORMATION: h_sing + δ_gt broadcast
                                         to non-pad positions → emit plural

The decoder learns: "uniform δ-shift in plural direction → use vocab branch
to emit plural surface form" (because pointer can't copy plural words from
an h that was built from singular).

At inference, we use the 3.5b sentence-trained operator's δ (approximation
of ground-truth δ). If the decoder generalizes from training-time δ_gt to
inference-time δ_op (≈δ_gt up to operator residual), the chain works.

Acceptance gate (same as 3.5/3.5b per plan §19.17):
  ≥ 50% of operated outputs contain plural-form subject (≥4/8).

Held-out test subjects (mango, peach, cabbage, penguin, bookcase,
trumpet, helicopter, church) are NOT in operator training (verified in
3.5b) AND NOT in decoder transformation training (verified here).
Genuinely novel transfer.

Run on the GPU box (~30-45 min):
  python scripts/stage3_5c_cotrained_decoder.py
  python scripts/stage3_5c_cotrained_decoder.py --steps 5000
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
from selflearnai.generator import (
    PointerSeqCondDecoder,
    perturb_h,
    mixture_nll,
)
from selflearnai.generator.loss import mse_activation_loss

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


# ---------------------------------------------------------------------------
# Training-data builder for the co-trained decoder.
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_cotraining_examples(
    sent_pairs: list[tuple[str, str]],
    *,
    tok, mdl, device, t_max,
) -> dict:
    """For each (sing, plur) pair build A/B/C training examples.

    Returns a dict of stacked tensors:
      h_inputs:        [3N, T, D] — h_sing | h_plur | h_perturbed
      h_masks:         [3N, T]    — mask_sing | mask_plur | mask_sing (perturbed
                                    keeps the singular shape since we add δ
                                    to h_sing; we use the singular mask)
      target_ids:      [3N, T]    — ids_sing | ids_plur | ids_plur
      target_h:        [3N, T, D] — what MSE loss reconstructs against
                                    (h_sing | h_plur | h_plur — perturbed
                                     should reconstruct the plural target h)
      stream_label:    [3N]       — 0/1/2 = identity_sing / identity_plur / transform
    """
    sing_sents = [p[0] for p in sent_pairs]
    plur_sents = [p[1] for p in sent_pairs]

    # Encode both directions
    h_s, mask_s, ids_s = encode_activations(sing_sents, tok, mdl, device, t_max=t_max)
    h_p, mask_p, ids_p = encode_activations(plur_sents, tok, mdl, device, t_max=t_max)

    # Pool to ψ
    rm_s = mask_s.unsqueeze(-1).float()
    rm_p = mask_p.unsqueeze(-1).float()
    psi_s = (h_s * rm_s).sum(dim=1) / rm_s.sum(dim=1).clamp(min=1.0)
    psi_p = (h_p * rm_p).sum(dim=1) / rm_p.sum(dim=1).clamp(min=1.0)

    # Ground-truth δ in sentence-pooled-ψ space.
    delta_gt = (psi_p - psi_s).unsqueeze(1)              # [N, 1, D]
    h_perturbed = h_s + delta_gt * rm_s                   # broadcast to non-pad

    h_inputs = torch.cat([h_s, h_p, h_perturbed], dim=0)
    h_masks = torch.cat([mask_s, mask_p, mask_s], dim=0)        # perturbed uses sing mask
    target_ids = torch.cat([ids_s, ids_p, ids_p], dim=0)
    target_h = torch.cat([h_s, h_p, h_p], dim=0)
    n = h_s.size(0)
    stream_label = torch.cat([
        torch.zeros(n, dtype=torch.long),
        torch.ones(n, dtype=torch.long),
        torch.full((n,), 2, dtype=torch.long),
    ], dim=0)
    return {
        "h_inputs": h_inputs,
        "h_masks": h_masks,
        "target_ids": target_ids,
        "target_h": target_h,
        "stream_label": stream_label,
        "n_per_stream": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # Operator training
    parser.add_argument("--op-epochs", type=int, default=3000)
    parser.add_argument("--op-lr", type=float, default=1e-3)
    # Decoder fine-tune
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="10x smaller than 3.1's 2e-4 — fine-tune, not from scratch")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--no-perturb", action="store_true",
                        help="skip perturbation aug (we're already perturbing intentionally)")
    parser.add_argument("--perturb-prob", type=float, default=0.15)
    parser.add_argument("--gaussian-delta", type=float, default=0.7)
    parser.add_argument("--mask-token-rate", type=float, default=0.3)
    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    # IO
    parser.add_argument("--decoder-init",
                        default="data/explanations_v2/checkpoints/decoder_3a1.pt",
                        help="Stage 3.1 decoder to fine-tune from")
    parser.add_argument("--decoder-out",
                        default="data/explanations_v2/checkpoints/decoder_3_5c.pt")
    parser.add_argument("--out", default="results/stage3/cross_domain_compose_v3.json")
    parser.add_argument("--subj-plural-min", type=float, default=0.50)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Stage 3 / Sub-task 3.5c — co-trained decoder for definitional ∘ plural")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    assert_no_test_subject_leakage()

    use_perturb = not args.no_perturb

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]
    encode_pooled = make_encode_fn(mdl, tok, args.device)

    # ---- Sentence pairs + sentence-trained operator ------------------
    sent_pairs = build_op_train_sentence_pairs()
    print(f"\n[1] Sentence pairs: {len(sent_pairs)} ({len(OP_TRAIN_PAIRS)} subjects "
          f"× {len(OP_TRAIN_TEMPLATES_SING_PLUR)} templates)")
    print("-" * 78)
    print(f"  example: {sent_pairs[0][0]!r}  →  {sent_pairs[0][1]!r}")
    test_subjects = {p["subj_sing"] for p in TEST_PAIRS}
    print(f"  held-out test subjects: {sorted(test_subjects)}")

    print(f"\n[2] Train sentence-level plural operator ({args.op_epochs} epochs)")
    print("-" * 78)
    plural_op, op_history = train_sentence_operator(
        encode_pooled, sent_pairs, dim=DIM, device=args.device,
        seed=args.seed, epochs=args.op_epochs, lr=args.op_lr,
    )
    print(f"  final cos(op→tgt) on training: {op_history[-1]['cos']:.4f}")

    # ---- Build co-training examples ----------------------------------
    print(f"\n[3] Build A/B/C co-training examples")
    print("-" * 78)
    train_data = build_cotraining_examples(
        sent_pairs, tok=tok, mdl=mdl, device=args.device, t_max=args.t_max,
    )
    h_inputs    = train_data["h_inputs"]
    h_masks     = train_data["h_masks"]
    target_ids  = train_data["target_ids"]
    target_h    = train_data["target_h"]
    stream      = train_data["stream_label"]
    N = h_inputs.size(0)
    print(f"  total examples: {N} ({train_data['n_per_stream']} per stream)")
    print(f"  stream A (identity singular): {(stream == 0).sum().item()}")
    print(f"  stream B (identity plural):   {(stream == 1).sum().item()}")
    print(f"  stream C (perturbed → plural): {(stream == 2).sum().item()}")

    # ---- Load Stage 3.1 decoder for fine-tuning ----------------------
    print(f"\n[4] Load Stage 3.1 decoder for fine-tuning")
    print("-" * 78)
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size,
        n_layers=args.n_layers, n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
    ).to(args.device)
    init_path = Path(args.decoder_init)
    if not init_path.exists():
        raise SystemExit(f"FATAL: decoder init not found at {init_path}")
    sd = torch.load(str(init_path), map_location=args.device)
    decoder.load_state_dict(sd)
    print(f"  loaded {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params from {init_path}")

    opt = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        return args.lr

    # ---- Fine-tune ---------------------------------------------------
    print(f"\n[5] Fine-tune decoder ({args.steps} steps, lr {args.lr})")
    print("-" * 78)
    decoder.train()
    loss_history = []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad()
        idx = torch.randint(0, N, (args.batch_size,), device=args.device)
        h_b = h_inputs[idx]
        mask_b = h_masks[idx]
        ids_b = target_ids[idx]
        target_h_b = target_h[idx]
        stream_b = stream[idx]

        if use_perturb:
            h_in = perturb_h(
                h_b, mask_b,
                apply_prob=args.perturb_prob,
                gaussian_delta=args.gaussian_delta,
                mask_token_rate=args.mask_token_rate,
            )
        else:
            h_in = h_b

        log_probs, hidden_out, p_gen = decoder(h_in, mask_b, ids_b)
        nll = mixture_nll(log_probs, ids_b)
        mse_loss = mse_activation_loss(
            hidden_out, decoder.mse_proj, target_h_b, mask_b,
        )
        loss = nll + args.mse_weight * mse_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                # Per-stream NLL break-down to see if all 3 streams converge.
                nll_per = {}
                for s in (0, 1, 2):
                    sm = (stream_b == s)
                    if sm.any():
                        nll_per[s] = float(mixture_nll(
                            log_probs[sm], ids_b[sm]
                        ).item())
            loss_history.append({
                "step": step, "loss": float(loss.item()),
                "nll": float(nll.item()), "mse": float(mse_loss.item()),
                "p_gen_mean": float(p_gen.mean().item()),
                "nll_stream_A": nll_per.get(0),
                "nll_stream_B": nll_per.get(1),
                "nll_stream_C": nll_per.get(2),
            })
            print(f"  step {step:>5}/{args.steps}  loss={loss.item():.4f}  "
                  f"nll={nll.item():.4f}  mse={mse_loss.item():.4f}  "
                  f"p_gen={p_gen.mean().item():.3f}  "
                  f"nll_A={nll_per.get(0)}  nll_B={nll_per.get(1)}  nll_C={nll_per.get(2)}")

    Path(args.decoder_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), args.decoder_out)
    print(f"  saved fine-tuned decoder to {args.decoder_out}")

    # ---- Held-out test on 8 truly-novel subjects --------------------
    decoder.eval()
    n = len(TEST_PAIRS)
    print(f"\n[6] Held-out test ({n} truly-novel subjects, never in op or decoder training)")
    print("-" * 78)
    sing_sents = [f"{p['subj_sing']} is a {p['cat_sing']}" for p in TEST_PAIRS]
    plur_refs = [f"{p['subj_plur'][0]} are {p['cat_plur'][0]}" for p in TEST_PAIRS]

    h_sing_t, mask_sing_t, ids_sing_t = encode_activations(
        sing_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    h_pref_t, mask_pref_t, ids_pref_t = encode_activations(
        plur_refs, tok, mdl, args.device, t_max=args.t_max,
    )
    psi_sing_t = encode_pooled(sing_sents)
    psi_pref_t = encode_pooled(plur_refs)

    # Apply OPERATOR (not ground-truth δ — this is the inference scenario).
    with torch.no_grad():
        psi_op = plural_op(psi_sing_t)
        delta_op = psi_op - psi_sing_t
    real_mask = (mask_sing_t > 0).unsqueeze(-1).float()
    h_op = h_sing_t + delta_op.unsqueeze(1) * real_mask

    cos_op_pref = F.cosine_similarity(psi_op, psi_pref_t, dim=-1)
    cos_sing_pref = F.cosine_similarity(psi_sing_t, psi_pref_t, dim=-1)
    print(f"  cos(op_psi, pref_psi):    {cos_op_pref.mean().item():.4f}")
    print(f"  cos(sing_psi, pref_psi):  {cos_sing_pref.mean().item():.4f}")
    print(f"  lift:                      {(cos_op_pref - cos_sing_pref).mean().item():+.4f}")

    text_baseline = decode_h(decoder, tok, h_sing_t, mask_sing_t, ids_sing_t)
    text_operated = decode_h(decoder, tok, h_op, mask_sing_t, ids_sing_t)
    text_oracle = decode_h(decoder, tok, h_pref_t, mask_pref_t, ids_pref_t)

    # Score
    print(f"\n[7] Per-case results")
    print("-" * 78)
    print(f"  {'#':<2} {'subj':<11} {'baseline':<28} {'operated':<28} {'oracle':<28}")
    rows = []
    for i, p in enumerate(TEST_PAIRS):
        wb = _words_in(text_baseline[i])
        wo = _words_in(text_operated[i])
        wx = _words_in(text_oracle[i])
        b_subj = has_any(wb, p["subj_plur"])
        o_subj = has_any(wo, p["subj_plur"])
        x_subj = has_any(wx, p["subj_plur"])
        b_cat = has_any(wb, p["cat_plur"])
        o_cat = has_any(wo, p["cat_plur"])
        x_cat = has_any(wx, p["cat_plur"])
        rows.append({
            "subj_sing": p["subj_sing"], "cat_sing": p["cat_sing"],
            "subj_plur_options": p["subj_plur"], "cat_plur_options": p["cat_plur"],
            "baseline_text": text_baseline[i],
            "operated_text": text_operated[i],
            "oracle_text":   text_oracle[i],
            "baseline_subj_plur": b_subj, "operated_subj_plur": o_subj, "oracle_subj_plur": x_subj,
            "baseline_cat_plur":  b_cat,  "operated_cat_plur":  o_cat,  "oracle_cat_plur":  x_cat,
        })
        bm = "✓" if b_subj else " "
        om = "✓" if o_subj else " "
        xm = "✓" if x_subj else " "
        print(f"  {i+1:<2} {p['subj_sing']:<11} "
              f"{bm} {text_baseline[i]:<26.26} "
              f"{om} {text_operated[i]:<26.26} "
              f"{xm} {text_oracle[i]:<26.26}")

    n_baseline = sum(1 for r in rows if r["baseline_subj_plur"])
    n_operated = sum(1 for r in rows if r["operated_subj_plur"])
    n_oracle = sum(1 for r in rows if r["oracle_subj_plur"])
    n_op_cat = sum(1 for r in rows if r["operated_cat_plur"])
    n_op_both = sum(1 for r in rows if r["operated_subj_plur"] and r["operated_cat_plur"])

    print("\n" + "=" * 78)
    print("ROLL-UP")
    print("=" * 78)
    print(f"  subject-plural in output:")
    print(f"    baseline (no op):                  {n_baseline}/{n}")
    print(f"    operated (sentence-op + co-trained decoder): {n_operated}/{n}  ← GATE")
    print(f"    oracle (encode plural-ref):         {n_oracle}/{n}  (upper bound)")
    print(f"  operated category-plural:            {n_op_cat}/{n}")
    print(f"  operated BOTH (subj + cat):          {n_op_both}/{n}")

    rate_op = n_operated / n
    print(f"\n  operated rate: {rate_op:.4f}")
    print(f"  gate target:   ≥ {args.subj_plural_min}")

    if rate_op >= args.subj_plural_min:
        verdict = "STAGE_3_5_PASS"
        message = (
            f"Co-trained decoder + sentence-trained operator hits "
            f"{n_operated}/{n} ({rate_op:.0%}) subject-plural. Cross-"
            f"domain composition VALIDATED with two principled fixes: "
            f"(1) train operators in destination encoder manifold, "
            f"(2) train decoders on transformation pairs (h_perturbed → "
            f"plural target) so the vocab branch learns to interpret "
            f"ψ-shifts as transformation signals. The architecture "
            f"composes — operator + decoder co-design is the unit of "
            f"transformation, not operator alone. Universal-pipeline "
            f"thesis fully validated end-to-end."
        )
    elif rate_op >= 0.30:
        verdict = "STAGE_3_5_BELOW_TARGET_PARTIAL"
        message = (
            f"Co-trained decoder lifted {rate_op:.0%} from baseline "
            f"{n_baseline/n:.0%} but below {args.subj_plural_min:.0%} gate. "
            f"Try: more transformation training pairs, more steps, or "
            f"larger lr."
        )
    else:
        verdict = "STAGE_3_5_FAIL_AFTER_COTRAIN"
        message = (
            f"Co-training did not lift held-out plural generation above "
            f"baseline. Inspect per-stream NLL: did stream C converge? "
            f"If stream C converged but inference still fails, the "
            f"OPERATOR's δ_op is too far from training-time δ_gt for "
            f"the decoder's transformation generalization to fire."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    payload = {
        "task": "3.5c",
        "fix_strategy": "decoder-cotrained-on-transformation-pairs",
        "encoder": args.encoder, "encoder_dim": DIM,
        "n_test_cases": n,
        "n_op_train_pairs": len(sent_pairs),
        "n_cotrain_examples": int(N),
        "op_train": {
            "epochs": args.op_epochs,
            "history": op_history,
        },
        "decoder_finetune": {
            "init_path": str(init_path),
            "out_path":  args.decoder_out,
            "steps": args.steps,
            "lr": args.lr,
            "loss_history": loss_history,
        },
        "psi_space_metrics": {
            "cos_op_pref_mean":   float(cos_op_pref.mean().item()),
            "cos_sing_pref_mean": float(cos_sing_pref.mean().item()),
            "lift": float((cos_op_pref - cos_sing_pref).mean().item()),
        },
        "subj_plural": {"baseline": n_baseline, "operated": n_operated, "oracle": n_oracle},
        "cat_plural_operated": n_op_cat,
        "both_plural_operated": n_op_both,
        "rate_operated_subj_plur": rate_op,
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
