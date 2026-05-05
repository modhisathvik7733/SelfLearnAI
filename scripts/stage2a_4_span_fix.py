"""Phase 2a / Sub-task 2a.4 (Path A) — multi-subword fix via fine-tune.

Quick experiment to lift the comparative concept's word-fidelity from
67% (72/108 in 2a.3) to ≥ 85% by adding a position-alignment bias on
the pointer attention. ~1.5 hr GPU.

Diagnosis (plan §19.14 documented limitation): comparative compounds
like `prettier` tokenize as `[pretty, ##ier]` (2 BERT WordPiece
tokens). The pointer copies one token at a time and gets confused on
the per-position alignment, producing outputs like `prettyistti`
instead of `pretty is prettier`. The vocab branch (`p_gen ≈ 0.6`) is
already low — the model wants to copy — but the copy targets the
wrong positions.

Fix (Path A — minimal architectural extension): add a learnable
T_out × T_in position bias matrix to the pointer's attention scores,
initialized as a soft identity (0.5 on the diagonal). For Phase 2a's
autoencoder-shaped setup (output sentence == encoder input), this
bias gives the pointer a structural prior: output position N likely
attends to encoder position N. Multi-subword cases benefit because
positions [4=pretty, 5=##ier] now have clearer per-position alignment.

Implementation:
  - selflearnai/generator/decoder.py: PointerSeqCondDecoder gains an
    optional `use_position_bias` flag (defaults False, preserves 2a.3
    behavior). When True, adds a 32×32 learnable parameter (1024
    floats) to ptr_scores.
  - This script: builds the decoder with use_position_bias=True,
    loads 2a.3's checkpoint with strict=False (lets the new
    position_bias param pick up its random init), fine-tunes for
    5K steps with reduced lr=1e-4, then evaluates on the SAME 432
    held-out sentences as 2a.3 for a clean comparison.

Acceptance gates (HARD, per-concept):
  - comparative both_in ≥ 92/108 (~85%, lifted from 72/108=67%)
  - other concepts maintain 2a.3 levels (no regression > 5%)
  - overall both_in ≥ 388/432 (90%, lifted from 372/432=86.1%)

If Path A passes: 2a.4 is closed and we proceed to 2a.5
(multi-candidate sampling) and 2a.7 (closing report).
If Path A fails: try Path B (clean implementation with span-copy
mechanism, full retrain) or Path C (vocabulary swap). Either way,
the diagnosis is empirical.

Run on the GPU box (~1-2 hr):
  python scripts/stage2a_4_span_fix.py
  python scripts/stage2a_4_span_fix.py --steps 10000  # longer if borderline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.generator import (
    PointerSeqCondDecoder,
    perturb_h,
    mixture_nll,
    GenerationVerdict,
    word_pair_fidelity,
    read_corpus_tsv,
)
from selflearnai.generator.eval import grammar_proxy, roll_up_gates
from selflearnai.generator.loss import mse_activation_loss

from scripts.stage1_planner_beam_smoke import ENCODERS

try:
    from scripts.stage2a_quick_probe import grammar_grade
except ImportError:
    def grammar_grade(text: str) -> tuple[int, bool]:
        proxy = grammar_proxy(text)
        return (0 if proxy >= 0.55 else 1, proxy >= 0.55)


@torch.no_grad()
def encode_pooled(sents, tok, mdl, device, max_length=64):
    inputs = tok(sents, padding=True, truncation=True, max_length=max_length,
                 return_tensors="pt").to(device)
    out = mdl(**inputs).last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).float()
    return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def encode_activations(sents, tok, mdl, device, t_max=32):
    inputs = tok(sents, padding="max_length", truncation=True,
                 max_length=t_max, return_tensors="pt").to(device)
    out = mdl(**inputs).last_hidden_state
    return out, inputs.attention_mask.float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus-dir", default="data/explanations_v2")
    parser.add_argument("--ckpt-2a3", default="data/explanations_v2/checkpoints/decoder_2a3.pt",
                        help="Path to the 2a.3 production checkpoint to fine-tune from.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Lower than 2a.3's 2e-4 — this is fine-tuning, not training.")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Architecture (locked from §19.14, plus 2a.4's position bias)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--position-bias-init", type=float, default=0.5,
                        help="Diagonal init scale for the new position bias.")
    # Loss recipe
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--no-mse", action="store_true")
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--perturb-prob", type=float, default=0.3)
    parser.add_argument("--gaussian-delta", type=float, default=0.7)
    parser.add_argument("--mask-token-rate", type=float, default=0.3)
    # Acceptance — comparative-focused, plus no-regression on others
    parser.add_argument("--cos-min", type=float, default=0.90)
    parser.add_argument("--grammar-pass-rate", type=float, default=0.95)
    parser.add_argument("--word-fidelity-min", type=float, default=0.80)
    parser.add_argument("--comparative-min", type=int, default=92,
                        help="Comparative both_in count target (lifted from 72/108).")
    parser.add_argument("--no-regression-margin", type=int, default=5,
                        help="Other concepts may drop by at most this many vs 2a.3.")
    parser.add_argument("--out", default="results/stage2a/span_fix.json")
    parser.add_argument("--ckpt", default="data/explanations_v2/checkpoints/decoder_2a4.pt")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    use_mse = not args.no_mse and args.mse_weight > 0.0
    use_perturb = not args.no_perturb

    enc_cfg = ENCODERS[args.encoder]
    print("Phase 2a / Sub-task 2a.4 (Path A) — multi-subword fix via fine-tune")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Loading 2a.3 checkpoint: {args.ckpt_2a3}")
    print(f"Adding learnable position bias (init {args.position_bias_init} on diagonal)")

    # ---- Load encoder + corpus ---------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]

    corpus_dir = Path(args.corpus_dir)
    train_rows = read_corpus_tsv(corpus_dir / "train.tsv")
    holdout_rows = read_corpus_tsv(corpus_dir / "holdout.tsv")
    train_sents = [r.sentence for r in train_rows]
    holdout_sents = [r.sentence for r in holdout_rows]
    n_train = len(train_sents)
    n_holdout = len(holdout_sents)
    print(f"\nCorpus: {n_train} train + {n_holdout} held-out (same as 2a.3)")

    # ---- Encode -------------------------------------------------------
    print("\nEncoding train + holdout ...")
    train_h, train_h_mask = encode_activations(
        train_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    holdout_h, holdout_h_mask = encode_activations(
        holdout_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    holdout_psi = encode_pooled(
        holdout_sents, tok, mdl, args.device, max_length=args.t_max,
    )
    train_tok = tok(train_sents, padding="max_length", truncation=True,
                    max_length=args.t_max, return_tensors="pt").to(args.device)
    train_ids = train_tok.input_ids
    holdout_tok = tok(holdout_sents, padding="max_length", truncation=True,
                      max_length=args.t_max, return_tensors="pt").to(args.device)
    holdout_ids = holdout_tok.input_ids

    # ---- Decoder with position bias ----------------------------------
    torch.manual_seed(args.seed)
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
        use_position_bias=True,
        position_bias_init=args.position_bias_init,
    ).to(args.device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder: {n_params/1e6:.2f}M params (locked + 2a.4 position bias)")

    # Load 2a.3 weights (strict=False because position_bias is new).
    ckpt_path = Path(args.ckpt_2a3)
    if not ckpt_path.exists():
        raise SystemExit(f"FATAL: 2a.3 checkpoint not found at {ckpt_path}. "
                         f"Run scripts/stage2a_3_train.py first.")
    state = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    print(f"\nLoaded 2a.3 checkpoint:")
    print(f"  missing keys (expected: position_bias only): {missing}")
    print(f"  unexpected keys (expected: empty): {unexpected}")
    if unexpected:
        raise SystemExit(f"FATAL: unexpected keys in checkpoint — architecture mismatch.")
    expected_missing = ["position_bias"]
    if sorted(missing) != sorted(expected_missing):
        print(f"  WARNING: missing keys differ from expected {expected_missing}.")

    # Sanity: confirm position_bias was init'd, not loaded from ckpt.
    pb_diag_mean = float(torch.diag(decoder.position_bias).mean().item())
    print(f"  position_bias diagonal mean: {pb_diag_mean:.3f}  "
          f"(expected ~{args.position_bias_init})")

    opt = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        return args.lr

    # ---- Fine-tune ----------------------------------------------------
    print(f"\nFine-tuning {args.steps} steps  batch={args.batch_size}  "
          f"lr={args.lr} (warmup {args.warmup_steps})")
    decoder.train()
    loss_history: list[dict] = []

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad()
        idx = torch.randint(0, n_train, (args.batch_size,), device=args.device)
        h_batch = train_h[idx]
        h_mask_batch = train_h_mask[idx]
        ids_batch = train_ids[idx]

        h_input = perturb_h(
            h_batch, h_mask_batch,
            apply_prob=(args.perturb_prob if use_perturb else 0.0),
            gaussian_delta=args.gaussian_delta,
            mask_token_rate=args.mask_token_rate,
        )

        log_probs, hidden_out, p_gen = decoder(
            h_input, h_mask_batch, ids_batch,
        )
        nll = mixture_nll(log_probs, ids_batch)
        if use_mse:
            mse_loss = mse_activation_loss(
                hidden_out, decoder.mse_proj, h_batch, h_mask_batch,
            )
            loss = nll + args.mse_weight * mse_loss
        else:
            mse_loss = torch.tensor(0.0, device=args.device)
            loss = nll

        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            pb_diag_mean = float(torch.diag(decoder.position_bias).mean().item())
            pb_off_mean = float(
                (decoder.position_bias.sum() - torch.diag(decoder.position_bias).sum()).item()
                / (decoder.position_bias.numel() - decoder.position_bias.size(0))
            )
            loss_history.append({
                "step": step,
                "loss": float(loss.item()),
                "nll": float(nll.item()),
                "mse": float(mse_loss.item()),
                "p_gen_mean": float(p_gen.mean().item()),
                "pb_diag_mean": pb_diag_mean,
                "pb_off_mean": pb_off_mean,
                "lr": float(opt.param_groups[0]["lr"]),
            })
            print(f"  step {step:>5}/{args.steps}  loss={loss.item():.4f}  "
                  f"nll={nll.item():.4f}  mse={mse_loss.item():.4f}  "
                  f"p_gen={p_gen.mean().item():.3f}  "
                  f"pb_diag={pb_diag_mean:+.3f}  pb_off={pb_off_mean:+.3f}")

    # Save checkpoint
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), args.ckpt)
    print(f"\nDecoder checkpoint saved to {args.ckpt}")

    # ---- Held-out eval -----------------------------------------------
    print("\n" + "=" * 78)
    print(f"HELD-OUT EVALUATION ({n_holdout} truly-novel sentences)")
    print("=" * 78)
    decoder.eval()
    eval_bs = 32
    all_gen_ids = []
    all_p_gen = []
    with torch.no_grad():
        for s in range(0, n_holdout, eval_bs):
            log_probs, _, p_gen = decoder(
                holdout_h[s:s + eval_bs],
                holdout_h_mask[s:s + eval_bs],
                holdout_ids[s:s + eval_bs],
            )
            all_gen_ids.append(log_probs.argmax(dim=-1))
            all_p_gen.append(p_gen.squeeze(-1))
    gen_ids = torch.cat(all_gen_ids, dim=0)
    p_gen_all = torch.cat(all_p_gen, dim=0)
    pad_id = tok.pad_token_id

    gen_texts = [tok.decode(gen_ids[i].tolist(), skip_special_tokens=True)
                 for i in range(n_holdout)]
    psi_gen = encode_pooled(gen_texts, tok, mdl, args.device, max_length=args.t_max)
    cos_recovered = F.cosine_similarity(psi_gen, holdout_psi, dim=-1)

    verdicts: list[GenerationVerdict] = []
    per_concept: dict[str, dict] = {}
    for i in range(n_holdout):
        row = holdout_rows[i]
        gen_text = gen_texts[i]
        cos = float(cos_recovered[i].item())
        n_errors, gpass = grammar_grade(gen_text)
        proxy = grammar_proxy(gen_text)
        src_in, tgt_in, both_in = word_pair_fidelity(row.src, row.tgt, gen_text)
        exact = gen_text.strip() == row.sentence.strip()
        real_mask = (holdout_ids[i] != pad_id).float()
        p_gen_mean = float((p_gen_all[i] * real_mask).sum().item() /
                           real_mask.sum().clamp(min=1.0).item())
        verdicts.append(GenerationVerdict(
            target=row.sentence, generated=gen_text,
            concept=row.concept, src_word=row.src, tgt_word=row.tgt,
            cos_recovered=cos, grammar_pass=gpass, grammar_n_errors=n_errors,
            grammar_proxy=proxy,
            src_in_gen=src_in, tgt_in_gen=tgt_in, both_in_gen=both_in,
            exact_match=exact, p_gen_mean=p_gen_mean,
        ))
        agg = per_concept.setdefault(
            row.concept,
            {"n": 0, "cos_pass": 0, "grammar_pass": 0,
             "src_in": 0, "tgt_in": 0, "both_in": 0, "exact": 0,
             "cos_sum": 0.0},
        )
        agg["n"] += 1
        agg["cos_sum"] += cos
        if cos >= args.cos_min:        agg["cos_pass"] += 1
        if gpass:                      agg["grammar_pass"] += 1
        if src_in:                     agg["src_in"] += 1
        if tgt_in:                     agg["tgt_in"] += 1
        if both_in:                    agg["both_in"] += 1
        if exact:                      agg["exact"] += 1

    # Per-concept comparison to 2a.3 baseline (hardcoded from logged result).
    BASELINE_2A3 = {
        "comparative": {"n": 108, "both_in": 72, "exact": 72},
        "opposite":    {"n": 108, "both_in": 108, "exact": 108},
        "past_tense":  {"n": 108, "both_in": 96, "exact": 95},
        "plural":      {"n": 108, "both_in": 96, "exact": 96},
    }

    print(f"\nPer-concept aggregate (vs 2a.3 baseline):")
    print(f"  {'concept':<12}  {'n':>4}  {'both_in':>10}  {'2a.3 baseline':>14}  {'Δ':>6}")
    for concept in sorted(per_concept.keys()):
        agg = per_concept[concept]
        n_c = agg["n"]
        baseline = BASELINE_2A3.get(concept, {"both_in": 0, "n": n_c})
        delta = agg["both_in"] - baseline["both_in"]
        delta_mark = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {concept:<12}  {n_c:>4}  "
              f"{agg['both_in']:>4}/{n_c:<4}  "
              f"{baseline['both_in']:>4}/{baseline['n']:<4}     "
              f"{delta_mark}{abs(delta):>3}")

    # Show first 5 comparative cases (the focus of this fix)
    print("\nComparative samples (first 5 — the multi-subword diagnostic):")
    comp_count = 0
    for i, v in enumerate(verdicts):
        if v.concept != "comparative":
            continue
        comp_count += 1
        if comp_count > 5:
            break
        cos_mark = "✓" if v.cos_recovered >= args.cos_min else "✗"
        wf_mark = "✓" if v.both_in_gen else (
            "≈" if v.src_in_gen or v.tgt_in_gen else "✗"
        )
        em = " (exact)" if v.exact_match else ""
        print(f"  [{comp_count}] {v.src_word!r}→{v.tgt_word!r}")
        print(f"      target:    {v.target!r}")
        print(f"      generated: {v.generated!r}{em}")
        print(f"      {cos_mark} cos={v.cos_recovered:.4f}  "
              f"{wf_mark} fidelity="
              f"{'BOTH' if v.both_in_gen else ('SOME' if v.src_in_gen or v.tgt_in_gen else 'NONE')}  "
              f"p_gen={v.p_gen_mean:.3f}")

    # ---- Verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT (Phase 2a §19.14 multi-subword fix)")
    print("=" * 78)
    gates = roll_up_gates(
        verdicts,
        cos_min=args.cos_min,
        grammar_pass_rate=args.grammar_pass_rate,
        word_fidelity_min=args.word_fidelity_min,
    )
    for line in gates.summary_lines():
        print(line)

    # Per-concept gates: comparative target + no-regression on others.
    comp_both = per_concept.get("comparative", {}).get("both_in", 0)
    comparative_pass = comp_both >= args.comparative_min
    print(f"\n  comparative both_in:           {comp_both}/108  "
          f"(target ≥ {args.comparative_min}/108)  "
          f"{'PASS' if comparative_pass else 'FAIL'}")

    no_regression = True
    for concept in ("opposite", "past_tense", "plural"):
        agg = per_concept.get(concept, {})
        baseline = BASELINE_2A3[concept]
        if agg.get("both_in", 0) < baseline["both_in"] - args.no_regression_margin:
            no_regression = False
            print(f"  {concept} REGRESSED: {agg.get('both_in', 0)}/{baseline['n']}  "
                  f"vs 2a.3 baseline {baseline['both_in']}/{baseline['n']}")
    if no_regression:
        print(f"  no regression (margin {args.no_regression_margin}) on other concepts: PASS")

    if comparative_pass and no_regression and gates.all_pass:
        verdict = "PATH_A_PASS"
        message = (
            f"Position-bias fine-tune lifted comparative from 72/108 to "
            f"{comp_both}/108 without regressing other concepts. The "
            f"position-alignment prior is the right inductive bias for "
            f"this stage's autoencoder-shaped task. 2a.4 closes; proceed "
            f"to 2a.5 (multi-candidate sampling) and 2a.7 (closing report)."
        )
    elif not comparative_pass and no_regression:
        verdict = "PATH_A_INSUFFICIENT"
        message = (
            f"Comparative improved to {comp_both}/108 but missed the "
            f"≥{args.comparative_min}/108 target. Position bias helps but "
            f"isn't enough on its own. Next: try Path B (span-copy "
            f"mechanism — pointer outputs a contiguous range, not single "
            f"tokens) or Path C (vocabulary swap)."
        )
    elif comparative_pass and not no_regression:
        verdict = "PATH_A_REGRESSION"
        message = (
            f"Comparative passed but other concepts regressed. The "
            f"position bias is too strong for templates that don't have "
            f"position alignment. Try lower position_bias_init or "
            f"layer-frozen fine-tuning."
        )
    else:
        verdict = "PATH_A_FAIL"
        message = (
            f"Path A didn't fix the multi-subword issue and may have "
            f"regressed. Move to Path B (span-copy) for a deeper fix."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ---------------------------------------------------
    payload = {
        "task": "2a.4-pathA",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "ckpt_2a3_loaded": str(ckpt_path),
        "ckpt_2a4_saved": args.ckpt,
        "decoder": {
            "n_params": n_params,
            "use_position_bias": True,
            "position_bias_init": args.position_bias_init,
            "hidden_dim": args.hidden_dim,
            "t_max": args.t_max,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "use_mse": use_mse,
            "mse_weight": args.mse_weight,
            "use_perturb": use_perturb,
        },
        "thresholds": {
            "cos_min": args.cos_min,
            "grammar_pass_rate": args.grammar_pass_rate,
            "word_fidelity_min": args.word_fidelity_min,
            "comparative_min": args.comparative_min,
            "no_regression_margin": args.no_regression_margin,
        },
        "loss_history": loss_history,
        "per_concept": per_concept,
        "baseline_2a3": BASELINE_2A3,
        "summary": {
            "median_cos": gates.median_cos,
            "n_cos_pass": gates.n_cos_pass,
            "n_grammar_pass": gates.n_grammar_pass,
            "n_both_in": gates.n_both_in,
            "n_exact_match": gates.n_exact_match,
            "comparative_both_in": comp_both,
            "p_gen_overall": gates.p_gen_overall,
        },
        "gates": {
            "cos_gate": gates.cos_gate,
            "grammar_gate": gates.grammar_gate,
            "word_fidelity_gate": gates.word_fidelity_gate,
            "comparative_pass": comparative_pass,
            "no_regression": no_regression,
            "all_pass": gates.all_pass and comparative_pass and no_regression,
        },
        "verdict": verdict,
        "message": message,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if (comparative_pass and no_regression and gates.all_pass) else 1)


if __name__ == "__main__":
    main()
