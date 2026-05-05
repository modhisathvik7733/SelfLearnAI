"""Phase 2a / Sub-task 2a.5 — multi-candidate sampling ablation.

Validates plan §9.2's claim that multi-candidate sampling + ψ-fidelity
reranking provides meaningful lift over greedy decoding. ~30 min GPU.

Setup:
  - Load 2a.3 production checkpoint
  - Eval on the same 432 truly-novel held-out as 2a.3
  - K=1 (greedy argmax decode) vs K=5 (Gumbel-perturbed samples,
    pick best by cos(encode(candidate), ψ_target))
  - Compare per-item, overall, and per-concept

Acceptance gate (HARD per plan §19.14):
  - Median cos lift with K=5 vs K=1 ≥ 0.02

If lift is < 0.005, the multi-candidate machinery isn't pulling its
weight and §9.2's claim ("exploration without next-token sampling")
weakens — we'd surface that in 2a.7's closing report. If lift ≥ 0.02,
the mechanism is validated and the per-concept breakdown will show
WHERE the lift concentrates (likely comparative, which had the most
failed greedy decodes — Gumbel diversity helps when greedy is wrong).

Run on the GPU box (~30 min):
  python scripts/stage2a_5_multi_candidate.py
  python scripts/stage2a_5_multi_candidate.py --k 10 --temperature 0.8
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
    multi_candidate_sample,
    word_pair_fidelity,
    read_corpus_tsv,
)
from selflearnai.generator.eval import grammar_proxy

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
    parser.add_argument("--ckpt", default="data/explanations_v2/checkpoints/decoder_2a3.pt",
                        help="Production checkpoint to evaluate.")
    # K + sampling
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    # Architecture (must match the loaded checkpoint — locked from §19.14)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    # Acceptance
    parser.add_argument("--lift-min", type=float, default=0.02,
                        help="Minimum median cos lift K=5 vs K=1 (plan §19.14).")
    parser.add_argument("--out", default="results/stage2a/multi_candidate.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Phase 2a / Sub-task 2a.5 — multi-candidate sampling ablation")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Checkpoint: {args.ckpt}")
    print(f"K={args.k}, temperature={args.temperature}")

    # ---- Encoder + corpus + checkpoint -------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]

    corpus_dir = Path(args.corpus_dir)
    holdout_rows = read_corpus_tsv(corpus_dir / "holdout.tsv")
    holdout_sents = [r.sentence for r in holdout_rows]
    n_holdout = len(holdout_sents)
    print(f"\nHeld-out: {n_holdout} truly-novel sentences (same as 2a.3)")

    print("\nEncoding holdout ...")
    holdout_h, holdout_h_mask = encode_activations(
        holdout_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    holdout_psi = encode_pooled(
        holdout_sents, tok, mdl, args.device, max_length=args.t_max,
    )
    holdout_tok = tok(holdout_sents, padding="max_length", truncation=True,
                      max_length=args.t_max, return_tensors="pt").to(args.device)
    holdout_ids = holdout_tok.input_ids

    # Decoder + checkpoint load
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
    ).to(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"FATAL: checkpoint not found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    decoder.load_state_dict(state)
    decoder.eval()
    print(f"Loaded checkpoint ({sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params)")

    # ---- Encode-text closure for multi_candidate_sample --------------
    # multi_candidate_sample re-encodes each candidate to score by cos.
    @torch.no_grad()
    def encode_text_for_scoring(texts: list[str]) -> torch.Tensor:
        return encode_pooled(texts, tok, mdl, args.device, max_length=args.t_max)

    # ---- K=1 (greedy) baseline ---------------------------------------
    print("\n[1] K=1 greedy decode")
    print("-" * 78)
    eval_bs = 32
    greedy_texts: list[str] = []
    with torch.no_grad():
        for s in range(0, n_holdout, eval_bs):
            log_probs, _, _ = decoder(
                holdout_h[s:s + eval_bs],
                holdout_h_mask[s:s + eval_bs],
                holdout_ids[s:s + eval_bs],
            )
            ids = log_probs.argmax(dim=-1)
            for row in ids:
                greedy_texts.append(tok.decode(row.tolist(), skip_special_tokens=True))
    greedy_psi = encode_pooled(greedy_texts, tok, mdl, args.device, max_length=args.t_max)
    greedy_cos = F.cosine_similarity(greedy_psi, holdout_psi, dim=-1)
    print(f"  greedy median cos: {float(greedy_cos.median().item()):.4f}")

    # ---- K=k sampling + ψ-fidelity rerank ----------------------------
    print(f"\n[2] K={args.k} Gumbel-perturbed sampling + ψ-fidelity rerank")
    print("-" * 78)
    torch.manual_seed(args.seed)
    sampled_texts: list[str] = []
    sampled_cos_per_item: list[float] = []
    all_candidates: list[list[str]] = []
    all_cos: list[list[float]] = []

    # Process in batches to control memory.
    for s in range(0, n_holdout, eval_bs):
        e = min(s + eval_bs, n_holdout)
        best_t, best_c, all_c, all_co = multi_candidate_sample(
            decoder,
            holdout_h[s:e],
            holdout_h_mask[s:e],
            holdout_ids[s:e],
            psi_target=holdout_psi[s:e],
            encode_text_fn=encode_text_for_scoring,
            tokenizer=tok,
            k=args.k,
            temperature=args.temperature,
            seed=args.seed + s,
        )
        sampled_texts.extend(best_t)
        sampled_cos_per_item.extend([float(c) for c in best_c])
        all_candidates.extend(all_c)
        all_cos.extend(all_co)
    sampled_cos = torch.tensor(sampled_cos_per_item)
    print(f"  K={args.k} median cos: {float(sampled_cos.median().item()):.4f}")

    # ---- Compare -----------------------------------------------------
    print("\n[3] Lift analysis")
    print("-" * 78)
    lift_per_item = sampled_cos - greedy_cos.cpu()
    median_lift = float(lift_per_item.median().item())
    mean_lift = float(lift_per_item.mean().item())
    n_lifted = int((lift_per_item > 1e-4).sum().item())
    n_dropped = int((lift_per_item < -1e-4).sum().item())
    n_unchanged = n_holdout - n_lifted - n_dropped
    print(f"  median lift:  {median_lift:+.4f}  (target ≥ {args.lift_min:+.4f})")
    print(f"  mean lift:    {mean_lift:+.4f}")
    print(f"  items lifted: {n_lifted}/{n_holdout}")
    print(f"  items dropped:{n_dropped}/{n_holdout}")
    print(f"  unchanged:    {n_unchanged}/{n_holdout}")

    # Word-fidelity comparison.
    pad_id = tok.pad_token_id
    n_both_greedy = 0
    n_both_sampled = 0
    per_concept = {}
    for i, row in enumerate(holdout_rows):
        _, _, both_g = word_pair_fidelity(row.src, row.tgt, greedy_texts[i])
        _, _, both_s = word_pair_fidelity(row.src, row.tgt, sampled_texts[i])
        if both_g:
            n_both_greedy += 1
        if both_s:
            n_both_sampled += 1
        agg = per_concept.setdefault(row.concept,
            {"n": 0, "lift_sum": 0.0,
             "n_lifted": 0, "n_dropped": 0,
             "both_greedy": 0, "both_sampled": 0,
             "greedy_cos_sum": 0.0, "sampled_cos_sum": 0.0})
        agg["n"] += 1
        item_lift = float(lift_per_item[i].item())
        agg["lift_sum"] += item_lift
        if item_lift > 1e-4: agg["n_lifted"] += 1
        if item_lift < -1e-4: agg["n_dropped"] += 1
        if both_g: agg["both_greedy"] += 1
        if both_s: agg["both_sampled"] += 1
        agg["greedy_cos_sum"] += float(greedy_cos[i].item())
        agg["sampled_cos_sum"] += float(sampled_cos[i].item())

    print(f"\n  word-fidelity (BOTH src+tgt present):")
    print(f"    greedy:  {n_both_greedy}/{n_holdout} ({100*n_both_greedy/n_holdout:.1f}%)")
    print(f"    K={args.k}:    {n_both_sampled}/{n_holdout} ({100*n_both_sampled/n_holdout:.1f}%)")
    print(f"    Δ:       {n_both_sampled - n_both_greedy:+d}")

    # Per-concept lift
    print("\n[4] Per-concept lift")
    print("-" * 78)
    print(f"  {'concept':<12}  {'n':>4}  {'avg_lift':>10}  {'lifted':>8}  "
          f"{'fid_g':>10}  {'fid_K':>10}  {'fid_Δ':>6}")
    for concept in sorted(per_concept.keys()):
        agg = per_concept[concept]
        avg_lift = agg["lift_sum"] / agg["n"]
        fid_diff = agg["both_sampled"] - agg["both_greedy"]
        fid_diff_mark = "+" if fid_diff > 0 else ""
        print(f"  {concept:<12}  {agg['n']:>4}  {avg_lift:>+10.4f}  "
              f"{agg['n_lifted']:>4}/{agg['n']:<4}  "
              f"{agg['both_greedy']:>4}/{agg['n']:<4}  "
              f"{agg['both_sampled']:>4}/{agg['n']:<4}  "
              f"{fid_diff_mark}{fid_diff}")

    # Show a few concrete examples of K=k beating greedy
    print(f"\n[5] Sample wins (top 5 items by lift, where K={args.k} > greedy)")
    print("-" * 78)
    lift_indices = lift_per_item.topk(min(5, n_holdout)).indices.tolist()
    for rank, i in enumerate(lift_indices, start=1):
        if lift_per_item[i] <= 0:
            break
        row = holdout_rows[i]
        g = greedy_texts[i]
        s = sampled_texts[i]
        print(f"  [{rank}] {row.concept} {row.src!r}→{row.tgt!r}")
        print(f"      target:  {row.sentence!r}")
        print(f"      greedy:  {g!r} (cos={float(greedy_cos[i]):.4f})")
        print(f"      K={args.k}:    {s!r} (cos={float(sampled_cos[i]):.4f})  "
              f"lift={float(lift_per_item[i]):+.4f}")

    # ---- Verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    lift_ok = median_lift >= args.lift_min
    print(f"  median cos lift K={args.k} vs K=1:  {median_lift:+.4f}  "
          f"(target ≥ {args.lift_min:+.4f})  "
          f"{'PASS' if lift_ok else 'FAIL'}")
    print(f"  word-fidelity gain:                   "
          f"{n_both_sampled - n_both_greedy:+d}/{n_holdout}")
    print(f"  no items dropped beyond noise (<5%):  "
          f"{'PASS' if n_dropped < 0.05 * n_holdout else 'FAIL'} "
          f"(dropped {n_dropped})")

    if lift_ok and n_dropped < 0.05 * n_holdout:
        verdict = "MULTI_CANDIDATE_PASS"
        message = (
            f"Multi-candidate sampling provides meaningful lift over "
            f"greedy decoding. Median cos +{median_lift:.4f}, "
            f"word-fidelity +{n_both_sampled - n_both_greedy} sentences. "
            f"Plan §9.2's claim about exploration via stochastic sampling "
            f"is empirically validated. 2a.5 closes; proceed to 2a.7 "
            f"(closing report + RESULTS.md addendum + Phase 2a closure)."
        )
    elif lift_ok and n_dropped >= 0.05 * n_holdout:
        verdict = "MULTI_CANDIDATE_NOISY"
        message = (
            f"K={args.k} sampling improves the median but drops too many "
            f"items beyond noise ({n_dropped}/{n_holdout}). Try lower "
            f"temperature ({args.temperature * 0.5:.2f}) to reduce "
            f"variance, or pick the candidate by cos AGAINST greedy as "
            f"a tiebreaker."
        )
    else:
        verdict = "MULTI_CANDIDATE_NO_LIFT"
        message = (
            f"K={args.k} sampling didn't materially improve over greedy "
            f"(median lift {median_lift:+.4f} < {args.lift_min:+.4f}). "
            f"This suggests the model's distribution is highly peaked "
            f"on a single best output — Gumbel sampling can't find a "
            f"better candidate when greedy is already at the mode. "
            f"Plan §9.2's exploration claim is weakened. Document in "
            f"2a.7's closing report; proceed to closure."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ----------------------------------------------------
    payload = {
        "task": "2a.5",
        "encoder": args.encoder,
        "checkpoint": str(ckpt_path),
        "n_holdout": n_holdout,
        "k": args.k,
        "temperature": args.temperature,
        "thresholds": {
            "lift_min": args.lift_min,
            "max_dropped_frac": 0.05,
        },
        "summary": {
            "greedy_median_cos": float(greedy_cos.median().item()),
            "sampled_median_cos": float(sampled_cos.median().item()),
            "median_lift": median_lift,
            "mean_lift": mean_lift,
            "n_lifted": n_lifted,
            "n_dropped": n_dropped,
            "n_unchanged": n_unchanged,
            "n_both_greedy": n_both_greedy,
            "n_both_sampled": n_both_sampled,
            "word_fidelity_gain": n_both_sampled - n_both_greedy,
        },
        "per_concept": per_concept,
        "verdict": verdict,
        "message": message,
        "per_item": [
            {
                "target": holdout_rows[i].sentence,
                "concept": holdout_rows[i].concept,
                "src": holdout_rows[i].src,
                "tgt": holdout_rows[i].tgt,
                "greedy_text": greedy_texts[i],
                "greedy_cos": float(greedy_cos[i].item()),
                "sampled_text": sampled_texts[i],
                "sampled_cos": float(sampled_cos[i].item()),
                "lift": float(lift_per_item[i].item()),
                "candidate_texts": all_candidates[i],
                "candidate_coses": all_cos[i],
            }
            for i in range(n_holdout)
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if (lift_ok and n_dropped < 0.05 * n_holdout) else 1)


if __name__ == "__main__":
    main()
