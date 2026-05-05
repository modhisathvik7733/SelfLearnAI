"""Phase 2a / Sub-task 2a.3 — full training run on production corpus.

The decisive Stage 2a gate (plan §19.14). ~6–8 hr GPU.

Trains the validated PointerSeqCondDecoder architecture (commits
bd4c183, d554c6c — 80% bit-exact on truly-novel held-out at small
scale) on the production corpus from 2a.1 (2064 train + 432 truly-
novel held-out across 4 concepts).

The first sub-task that uses the `selflearnai/generator/` package
(refactored in 2a.2). All training/eval logic lives in the package;
this script is orchestration only.

Acceptance gates (HARD, all three required per plan §19.14):
  - Median cos ≥ 0.90 on the 432 held-out (note: tighter than the
    smaller-scale 0.85 used in §19.13 — at this scale we can demand
    more)
  - ≥ 95% sentences pass grammar gate (LanguageTool errors == 0;
    falls back to wordfreq proxy with caveat if Java not installed)
  - ≥ 80% sentences contain BOTH target src + tgt words (tighter
    than §19.13's 70% — production target)

Outputs:
  results/stage2a/full_train.json    — training stats + held-out gates
  data/explanations_v2/checkpoints/decoder_2a3.pt  — saved decoder

Run on the GPU box (~6–8 hr):
  python scripts/stage2a_3_train.py
  python scripts/stage2a_3_train.py --steps 50000           # longer run
  python scripts/stage2a_3_train.py --no-mse --no-perturb   # ablation
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
    CorpusEntry,
)
from selflearnai.generator.eval import grammar_proxy, roll_up_gates
from selflearnai.generator.loss import mse_activation_loss

from scripts.stage1_planner_beam_smoke import ENCODERS

# Soft import for LanguageTool — falls back to proxy if Java absent.
try:
    from scripts.stage2a_quick_probe import grammar_grade
except ImportError:
    def grammar_grade(text: str) -> tuple[int, bool]:
        proxy = grammar_proxy(text)
        return (0 if proxy >= 0.55 else 1, proxy >= 0.55)


# ---------------------------------------------------------------------------
# Encoder helpers (mirror 2a.0e/0f shape — these used to live in scripts/)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_pooled(
    sents: list[str], tok, mdl, device: str, max_length: int = 64,
) -> torch.Tensor:
    inputs = tok(
        sents, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = mdl(**inputs).last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).float()
    return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def encode_activations(
    sents: list[str], tok, mdl, device: str, t_max: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = tok(
        sents, padding="max_length", truncation=True, max_length=t_max,
        return_tensors="pt",
    ).to(device)
    out = mdl(**inputs).last_hidden_state
    return out, inputs.attention_mask.float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # Corpus
    parser.add_argument("--corpus-dir", default="data/explanations_v2")
    # Training
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Architecture (locked from §19.14)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    # Loss recipe
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--no-mse", action="store_true")
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--perturb-prob", type=float, default=0.3)
    parser.add_argument("--gaussian-delta", type=float, default=0.7)
    parser.add_argument("--mask-token-rate", type=float, default=0.3)
    # Acceptance (production targets — tighter than §19.13 smoke)
    parser.add_argument("--cos-min", type=float, default=0.90)
    parser.add_argument("--grammar-pass-rate", type=float, default=0.95)
    parser.add_argument("--word-fidelity-min", type=float, default=0.80)
    # Output
    parser.add_argument("--out", default="results/stage2a/full_train.json")
    parser.add_argument("--ckpt", default="data/explanations_v2/checkpoints/decoder_2a3.pt")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5000,
                        help="Mid-training quick eval (subset of held-out).")
    args = parser.parse_args()

    use_mse = not args.no_mse and args.mse_weight > 0.0
    use_perturb = not args.no_perturb

    enc_cfg = ENCODERS[args.encoder]
    print("Phase 2a / Sub-task 2a.3 — production training run")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Loss: NLL on (vocab + copy) mixture | "
          f"MSE: {'on (λ=%.2f)' % args.mse_weight if use_mse else 'OFF'} | "
          f"Perturb: {'on (p=%.2f)' % args.perturb_prob if use_perturb else 'OFF'} | "
          f"FeatDropout: {args.feat_dropout}")

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]

    # ---- Corpus -------------------------------------------------------
    corpus_dir = Path(args.corpus_dir)
    train_tsv = corpus_dir / "train.tsv"
    holdout_tsv = corpus_dir / "holdout.tsv"
    if not train_tsv.exists() or not holdout_tsv.exists():
        raise SystemExit(
            f"FATAL: corpus files missing under {corpus_dir}. "
            f"Run scripts/stage2a_1_corpus.py first."
        )
    train_rows = read_corpus_tsv(train_tsv)
    holdout_rows = read_corpus_tsv(holdout_tsv)
    train_sents = [r.sentence for r in train_rows]
    holdout_sents = [r.sentence for r in holdout_rows]
    n_train = len(train_sents)
    n_holdout = len(holdout_sents)
    print(f"\nCorpus: {n_train} train + {n_holdout} held-out")

    # ---- Encode train + holdout (full activations for cross-attn) ----
    print("\nEncoding train + holdout sequences ...")
    train_h, train_h_mask = encode_activations(
        train_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    holdout_h, holdout_h_mask = encode_activations(
        holdout_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    holdout_psi = encode_pooled(
        holdout_sents, tok, mdl, args.device, max_length=args.t_max,
    )

    # Tokenize for both encoder input ids (pointer) and CE labels.
    train_tok = tok(
        train_sents, padding="max_length", truncation=True,
        max_length=args.t_max, return_tensors="pt",
    ).to(args.device)
    train_ids = train_tok.input_ids
    holdout_tok = tok(
        holdout_sents, padding="max_length", truncation=True,
        max_length=args.t_max, return_tensors="pt",
    ).to(args.device)
    holdout_ids = holdout_tok.input_ids
    longest_train = int(train_tok.attention_mask.sum(dim=-1).max().item())
    longest_holdout = int(holdout_tok.attention_mask.sum(dim=-1).max().item())
    print(f"  longest train tokenized:   {longest_train} (t_max={args.t_max})")
    print(f"  longest holdout tokenized: {longest_holdout}")

    # ---- Decoder (from package) --------------------------------------
    torch.manual_seed(args.seed)
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder: {n_params/1e6:.2f}M params (locked architecture from §19.14)")
    print(f"  layers={args.n_layers}  hidden={args.hidden_dim}  "
          f"heads={args.n_heads}  T_max={args.t_max}")

    opt = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        return args.lr

    # ---- Train --------------------------------------------------------
    print(f"\nTraining {args.steps} steps  batch={args.batch_size}  "
          f"lr={args.lr} (warmup {args.warmup_steps})")
    decoder.train()
    loss_history: list[dict] = []
    midtrain_evals: list[dict] = []

    @torch.no_grad()
    def quick_eval(step: int) -> dict:
        """Tiny held-out eval on first 64 sentences (cos only, fast)."""
        decoder.eval()
        n = min(64, n_holdout)
        log_probs, _, _ = decoder(
            holdout_h[:n], holdout_h_mask[:n], holdout_ids[:n],
        )
        gen_ids = log_probs.argmax(dim=-1)
        gen_texts = [
            tok.decode(row.tolist(), skip_special_tokens=True)
            for row in gen_ids
        ]
        psi_gen = encode_pooled(gen_texts, tok, mdl, args.device, max_length=args.t_max)
        cos = F.cosine_similarity(psi_gen, holdout_psi[:n], dim=-1)
        mean_cos = float(cos.mean().item())
        n_both = sum(
            1 for i in range(n)
            if word_pair_fidelity(
                holdout_rows[i].src, holdout_rows[i].tgt, gen_texts[i],
            )[2]
        )
        decoder.train()
        return {"step": step, "subset_n": n,
                "mean_cos": mean_cos, "n_both": n_both,
                "first_text": gen_texts[0]}

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
            loss_history.append({
                "step": step,
                "loss": float(loss.item()),
                "nll": float(nll.item()),
                "mse": float(mse_loss.item()),
                "p_gen_mean": float(p_gen.mean().item()),
                "lr": float(opt.param_groups[0]["lr"]),
            })
            print(f"  step {step:>5}/{args.steps}  "
                  f"loss={loss.item():.4f}  nll={nll.item():.4f}  "
                  f"mse={mse_loss.item():.4f}  "
                  f"p_gen={p_gen.mean().item():.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.6f}")

        if (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            qe = quick_eval(step + 1)
            midtrain_evals.append(qe)
            print(f"    [mid-eval @ step {qe['step']}] "
                  f"subset_n={qe['subset_n']}  mean_cos={qe['mean_cos']:.4f}  "
                  f"n_both={qe['n_both']}  first_text={qe['first_text']!r}")

    # ---- Save checkpoint ---------------------------------------------
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), ckpt_path)
    print(f"\nDecoder checkpoint saved to {ckpt_path}")

    # ---- Held-out eval -----------------------------------------------
    print("\n" + "=" * 78)
    print(f"HELD-OUT EVALUATION ({n_holdout} truly-novel sentences)")
    print("=" * 78)
    decoder.eval()
    eval_bs = 32
    all_gen_ids: list[torch.Tensor] = []
    all_p_gen: list[torch.Tensor] = []
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

    # Per-sentence: re-encode generated, score, build verdict.
    gen_texts: list[str] = []
    for i in range(n_holdout):
        gen_texts.append(tok.decode(gen_ids[i].tolist(), skip_special_tokens=True))
    psi_gen = encode_pooled(gen_texts, tok, mdl, args.device, max_length=args.t_max)
    cos_recovered = F.cosine_similarity(psi_gen, holdout_psi, dim=-1)

    verdicts: list[GenerationVerdict] = []
    per_concept_stats: dict[str, dict] = {}
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
        agg = per_concept_stats.setdefault(
            row.concept,
            {"n": 0, "cos_pass": 0, "grammar_pass": 0,
             "src_in": 0, "tgt_in": 0, "both_in": 0, "exact": 0,
             "cos_sum": 0.0},
        )
        agg["n"] += 1
        agg["cos_sum"] += cos
        if cos >= args.cos_min:
            agg["cos_pass"] += 1
        if gpass:
            agg["grammar_pass"] += 1
        if src_in:
            agg["src_in"] += 1
        if tgt_in:
            agg["tgt_in"] += 1
        if both_in:
            agg["both_in"] += 1
        if exact:
            agg["exact"] += 1

    # Per-concept aggregate
    print(f"\nPer-concept aggregate (across {n_holdout} held-out):")
    print(f"  {'concept':<12}  {'n':>4}  {'cos_avg':>8}  "
          f"{'cos_pass':>10}  {'grammar':>8}  {'both_in':>8}  {'exact':>8}")
    for concept in sorted(per_concept_stats.keys()):
        agg = per_concept_stats[concept]
        n_c = agg["n"]
        cos_avg = agg["cos_sum"] / n_c
        print(f"  {concept:<12}  {n_c:>4}  {cos_avg:>8.4f}  "
              f"{agg['cos_pass']:>4}/{n_c:<4}  "
              f"{agg['grammar_pass']:>4}/{n_c:<4}  "
              f"{agg['both_in']:>4}/{n_c:<4}  "
              f"{agg['exact']:>4}/{n_c:<4}")

    # First 10 samples for qualitative inspection
    print("\nPer-sentence sample (first 10):")
    for i, v in enumerate(verdicts[:10]):
        cos_mark = "✓" if v.cos_recovered >= args.cos_min else "✗"
        gr_mark = "✓" if v.grammar_pass else "✗"
        wf_mark = "✓" if v.both_in_gen else (
            "≈" if v.src_in_gen or v.tgt_in_gen else "✗"
        )
        em = " (exact)" if v.exact_match else ""
        print(f"  [{i+1}] {v.concept:<11} {v.src_word!r:>10}→{v.tgt_word!r:<11}")
        print(f"      target:    {v.target!r}")
        print(f"      generated: {v.generated!r}{em}")
        print(f"      {cos_mark} cos={v.cos_recovered:.4f}  "
              f"{gr_mark} LT={v.grammar_n_errors}  "
              f"{wf_mark} fidelity="
              f"{'BOTH' if v.both_in_gen else ('SOME' if v.src_in_gen or v.tgt_in_gen else 'NONE')}")

    # ---- Verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT (Phase 2a §19.14 closing gates)")
    print("=" * 78)
    gates = roll_up_gates(
        verdicts,
        cos_min=args.cos_min,
        grammar_pass_rate=args.grammar_pass_rate,
        word_fidelity_min=args.word_fidelity_min,
    )
    for line in gates.summary_lines():
        print(line)

    if gates.all_pass:
        verdict = "PHASE_2A_PASS"
        message = (
            "All three §19.14 closing gates passed at production scale. "
            "The locked architecture (sequence conditioning + Pointer-"
            "Generator + paper recipe) generalizes to truly-novel words. "
            "Next: 2a.4 (multi-subword copy fix) → 2a.5 (multi-candidate "
            "ablation) → 2a.7 (closing report + Phase 2a closure)."
        )
    elif gates.cos_gate and gates.grammar_gate and not gates.word_fidelity_gate:
        verdict = "WORD_FIDELITY_BELOW_GATE"
        message = (
            "Word-fidelity below 80% on held-out at production scale "
            "(was 81% in 2a.0f at smaller scale). Per-concept breakdown "
            "above will show which categories drag the average down — "
            "likely comparative (multi-subword issue, addressed in 2a.4)."
        )
    elif gates.cos_gate and not gates.grammar_gate:
        verdict = "GRAMMAR_BELOW_GATE"
        message = (
            "Grammar gate failed. If LanguageTool isn't installed, this "
            "is a wordfreq-proxy false-negative — install Java + LT and "
            "re-eval. If LT IS installed and grammar still fails, the "
            "model needs more training or a span-copy fix for compounds."
        )
    elif not gates.cos_gate:
        verdict = "COS_BELOW_GATE"
        message = (
            "Cos below 0.90 production target. May need more training "
            "steps, larger model, or this is a corpus-difficulty issue."
        )
    else:
        verdict = "MIXED_FAIL"
        message = "Multiple gates failed. See per-concept breakdown."

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ---------------------------------------------------
    payload = {
        "task": "2a.3",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "corpus_dir": str(corpus_dir),
        "n_train": n_train,
        "n_holdout": n_holdout,
        "decoder": {
            "n_params": n_params,
            "hidden_dim": args.hidden_dim,
            "t_max": args.t_max,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "ffn_mult": args.ffn_mult,
            "feat_dropout": args.feat_dropout,
            "attn_dropout": args.attn_dropout,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "use_mse": use_mse,
            "mse_weight": args.mse_weight,
            "use_perturb": use_perturb,
            "perturb_prob": args.perturb_prob if use_perturb else 0.0,
        },
        "thresholds": {
            "cos_min": args.cos_min,
            "grammar_pass_rate": args.grammar_pass_rate,
            "word_fidelity_min": args.word_fidelity_min,
        },
        "loss_history": loss_history,
        "midtrain_evals": midtrain_evals,
        "per_concept": per_concept_stats,
        "summary": {
            "median_cos": gates.median_cos,
            "n_cos_pass": gates.n_cos_pass,
            "n_grammar_pass": gates.n_grammar_pass,
            "n_src_in": gates.n_src_in,
            "n_tgt_in": gates.n_tgt_in,
            "n_both_in": gates.n_both_in,
            "n_exact_match": gates.n_exact_match,
            "p_gen_overall": gates.p_gen_overall,
        },
        "gates": {
            "cos_gate": gates.cos_gate,
            "grammar_gate": gates.grammar_gate,
            "word_fidelity_gate": gates.word_fidelity_gate,
            "all_pass": gates.all_pass,
        },
        "verdict": verdict,
        "message": message,
        "checkpoint_path": str(ckpt_path),
        "results": [
            {
                "target": v.target, "generated": v.generated,
                "concept": v.concept, "src_word": v.src_word, "tgt_word": v.tgt_word,
                "cos_recovered": v.cos_recovered,
                "grammar_pass": v.grammar_pass, "lt_n_errors": v.grammar_n_errors,
                "grammar_proxy": v.grammar_proxy,
                "src_in_gen": v.src_in_gen, "tgt_in_gen": v.tgt_in_gen,
                "both_in_gen": v.both_in_gen,
                "exact_match": v.exact_match,
                "p_gen_mean": v.p_gen_mean,
            }
            for v in verdicts
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if gates.all_pass else 1)


if __name__ == "__main__":
    main()
