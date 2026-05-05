"""Phase 2a sub-task 2a.0f — pointer-generator on truly novel words.

Sanity check on top of 2a.0e's POINTER_PASS (commit bd4c183).
2a.0e tested generalization to held-out word pairs that came from
text_pairs_held_out.tsv files — these are similar in distribution
to training (plural held-out: book/pig/key, training: cat/dog/car).
Strong result (151/168 exact match, 152/168 both words present)
but the held-out distribution was friendly.

This sub-task tests TRULY NOVEL word pairs that don't appear in
ANY existing concept TSV. Specifically:

  Plural (irregulars):
    mouse→mice, child→children, foot→feet, tooth→teeth,
    goose→geese, man→men, woman→women

  Past tense (strong irregulars):
    eat→ate, write→wrote, swim→swam, drink→drank,
    sing→sang, speak→spoke, catch→caught

  Comparative:
    pretty→prettier, simple→simpler, gentle→gentler,
    heavy→heavier, narrow→narrower, clever→cleverer, lonely→lonelier

  Opposite:
    rich→poor, safe→dangerous, friend→enemy,
    love→hate, awake→asleep, clean→dirty, truth→lie

7 pairs × 4 concepts × 7 templates = 196 truly-novel held-out sents.

If pointer-generator's COPY mechanism is the architectural fix — not
just a memorization trick that happened to work on near-distribution
data — the model should reproduce these word pairs even though it
has never seen them. The pointer copies from the encoder's input
sequence; word familiarity from training is irrelevant to the copy
operation itself.

Same train corpus as 2a.0e (1036 sentences from existing TSVs).
Same architecture (PointerSeqCondDecoder). Same training procedure.
Only the eval set changes.

Acceptance gates (HARD, all three):
  - Median cos ≥ 0.85 on the 196 novel held-out
  - ≥ 95% sentences pass grammar gate
  - ≥ 70% sentences contain BOTH target src + tgt words

Run on the GPU box (~40-60 min, same as 2a.0e):
  python scripts/stage2a_pointer_novel.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn, read_pairs
from scripts.stage2a_quick_probe import grammar_grade, grammar_proxy
from scripts.stage2a_generalization import (
    CONCEPT_DATA_DIRS,
    TEMPLATES,
    encode_pooled,
    encode_activations,
)
from scripts.stage2a_seq_conditioning import (
    perturb_h,
    word_pair_fidelity,
)
from scripts.stage2a_pointer import PointerSeqCondDecoder


# ---------------------------------------------------------------------------
# TRULY NOVEL word pairs (verified absent from existing concept TSVs)
# ---------------------------------------------------------------------------
# Sanity-checked against:
#   data/plurality/text_pairs_{train,held_out}.tsv (44 + 6 pairs)
#   data/past_tense/text_pairs_{train,held_out}.tsv (43 + 6)
#   data/comparative/text_pairs_{train,held_out}.tsv (31 + 6)
#   data/opposite_v2/text_pairs_{train,held_out}.tsv (30 + 6)
# These 28 pairs do NOT appear in any of the above (manual audit;
# the script also verifies at runtime — FATAL on any leak).

NOVEL_PAIRS: dict[str, list[tuple[str, str]]] = {
    "plural": [
        ("mouse", "mice"),
        ("child", "children"),
        ("foot",  "feet"),
        ("tooth", "teeth"),
        ("goose", "geese"),
        ("man",   "men"),
        ("woman", "women"),
    ],
    "past_tense": [
        ("eat",    "ate"),
        ("write",  "wrote"),
        ("swim",   "swam"),
        ("drink",  "drank"),
        ("sing",   "sang"),
        ("speak",  "spoke"),
        ("catch",  "caught"),
    ],
    "comparative": [
        ("pretty",  "prettier"),
        ("simple",  "simpler"),
        ("gentle",  "gentler"),
        ("heavy",   "heavier"),
        ("narrow",  "narrower"),
        ("clever",  "cleverer"),
        ("lonely",  "lonelier"),
    ],
    "opposite": [
        ("rich",   "poor"),
        ("safe",   "dangerous"),
        ("friend", "enemy"),
        ("love",   "hate"),
        ("awake",  "asleep"),
        ("clean",  "dirty"),
        ("truth",  "lie"),
    ],
}


def assert_no_overlap_with_existing_tsvs() -> None:
    """Verify NOVEL_PAIRS truly don't appear in any concept TSV. FATAL on leak."""
    leaks: list[tuple[str, tuple[str, str], str]] = []
    for concept, pairs in NOVEL_PAIRS.items():
        ddir = CONCEPT_DATA_DIRS[concept]
        existing: set[tuple[str, str]] = set()
        for fname in ("text_pairs_train.tsv", "text_pairs_held_out.tsv"):
            p = Path(ddir) / fname
            if p.exists():
                for src, tgt in read_pairs(p):
                    existing.add((src, tgt))
        for pair in pairs:
            if pair in existing:
                leaks.append((concept, pair, str(ddir)))
    if leaks:
        print("FATAL: NOVEL_PAIRS leak with existing TSVs:")
        for concept, pair, ddir in leaks:
            print(f"  {concept}: {pair} appears in {ddir}/")
        raise SystemExit(1)


def build_corpus_with_novel_holdout() -> tuple[list[str], list[str], list[tuple[str, str, str]], dict]:
    """Train: same as 2a.0c/d/e (existing concept TSVs × shared templates).
    Held-out: NOVEL_PAIRS × shared templates. Held-out word pairs do
    NOT appear in train (verified by assert_no_overlap_with_existing_tsvs).

    Returns (train_sents, holdout_sents, holdout_pair_index, stats).
    holdout_pair_index aligns 1:1 with holdout_sents, gives
    (concept, src, tgt) per row.
    """
    train_sents: list[str] = []
    holdout_sents: list[str] = []
    holdout_pair_index: list[tuple[str, str, str]] = []
    stats: dict = {"per_concept": {}}
    for concept, ddir in CONCEPT_DATA_DIRS.items():
        train_pairs = read_pairs(Path(ddir) / "text_pairs_train.tsv")
        novel_pairs = NOVEL_PAIRS[concept]
        templates = TEMPLATES[concept]
        for src, tgt in train_pairs:
            for tpl in templates:
                train_sents.append(tpl.format(src=src, tgt=tgt))
        for src, tgt in novel_pairs:
            for tpl in templates:
                holdout_sents.append(tpl.format(src=src, tgt=tgt))
                holdout_pair_index.append((concept, src, tgt))
        stats["per_concept"][concept] = {
            "n_train_pairs": len(train_pairs),
            "n_novel_pairs": len(novel_pairs),
            "n_templates": len(templates),
            "n_train_sents": len(train_pairs) * len(templates),
            "n_novel_sents": len(novel_pairs) * len(templates),
        }
    stats["total_train_sents"] = len(train_sents)
    stats["total_novel_sents"] = len(holdout_sents)
    return train_sents, holdout_sents, holdout_pair_index, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--no-mse", action="store_true")
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--perturb-prob", type=float, default=0.3)
    parser.add_argument("--gaussian-delta", type=float, default=0.7)
    parser.add_argument("--mask-token-rate", type=float, default=0.3)
    parser.add_argument("--cos-min", type=float, default=0.85)
    parser.add_argument("--grammar-pass-rate", type=float, default=0.95)
    parser.add_argument("--word-fidelity-min", type=float, default=0.70)
    parser.add_argument("--out", default="results/stage2a/pointer_novel.json")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    use_mse = not args.no_mse and args.mse_weight > 0.0
    use_perturb = not args.no_perturb

    enc_cfg = ENCODERS[args.encoder]
    print("Phase 2a sub-task 2a.0f — Pointer-Generator on TRULY NOVEL words")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    print("\nVerifying NOVEL_PAIRS don't leak into existing TSVs ...")
    assert_no_overlap_with_existing_tsvs()
    print("  OK: 28 truly-novel pairs × 7 templates = 196 unseen held-out sentences")

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]

    # ---- Corpus -------------------------------------------------------
    train_sents, holdout_sents, holdout_pairs_idx, corpus_stats = (
        build_corpus_with_novel_holdout()
    )
    n_train = len(train_sents)
    n_holdout = len(holdout_sents)
    print(f"\nCorpus:")
    print(f"  train (same as 2a.0e):  {n_train} sentences")
    print(f"  truly-novel held-out:   {n_holdout} sentences")
    for concept, st in corpus_stats["per_concept"].items():
        print(f"    {concept:<12}  train={st['n_train_sents']}  "
              f"novel={st['n_novel_sents']}  templates={st['n_templates']}")
    print(f"  novel held-out samples:")
    for s in holdout_sents[:6]:
        print(f"    {s!r}")

    # Encode train + held-out (full activations).
    print("\nEncoding train + truly-novel held-out ...")
    train_h, train_h_mask = encode_activations(
        train_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    holdout_h, holdout_h_mask = encode_activations(
        holdout_sents, tok, mdl, args.device, t_max=args.t_max,
    )
    holdout_psi = encode_pooled(
        holdout_sents, tok, mdl, args.device, max_length=args.t_max,
    )

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
    longest = int(train_tok.attention_mask.sum(dim=-1).max().item())
    print(f"  longest train tokenized length: {longest} (t_max={args.t_max})")

    # ---- Decoder ------------------------------------------------------
    torch.manual_seed(args.seed)
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nPointerSeqCondDecoder: {n_params/1e6:.2f}M params (same as 2a.0e)")

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
    pad_id = tok.pad_token_id
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad()
        idx = torch.randint(0, n_train, (args.batch_size,), device=args.device)
        h_batch = train_h[idx]
        h_mask_batch = train_h_mask[idx]
        ids_batch = train_ids[idx]

        if use_perturb:
            h_input = perturb_h(
                h_batch, h_mask_batch,
                apply_prob=args.perturb_prob,
                gaussian_delta=args.gaussian_delta,
                mask_token_rate=args.mask_token_rate,
            )
        else:
            h_input = h_batch

        log_probs, hidden_out, p_gen = decoder(
            h_input, h_mask_batch, ids_batch,
        )

        nll = F.nll_loss(
            log_probs.reshape(-1, log_probs.size(-1)),
            ids_batch.reshape(-1),
        )
        if use_mse:
            h_target = h_batch
            mask = h_mask_batch.unsqueeze(-1)
            h_pred = decoder.mse_proj(hidden_out)
            mse_per = (h_pred - h_target).pow(2)
            mse_loss = (mse_per * mask).sum() / (mask.sum() * DIM).clamp(min=1.0)
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
                  f"p_gen_mean={p_gen.mean().item():.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.6f}")

    # ---- Held-out eval (truly-novel) ---------------------------------
    print("\n" + "=" * 78)
    print("TRULY-NOVEL HELD-OUT EVALUATION")
    print("=" * 78)
    decoder.eval()
    eval_bs = 32
    all_gen_ids = []
    all_p_gen = []
    with torch.no_grad():
        for s in range(0, n_holdout, eval_bs):
            h_chunk = holdout_h[s:s + eval_bs]
            mask_chunk = holdout_h_mask[s:s + eval_bs]
            ids_chunk = holdout_ids[s:s + eval_bs]
            log_probs, _, p_gen = decoder(h_chunk, mask_chunk, ids_chunk)
            all_gen_ids.append(log_probs.argmax(dim=-1))
            all_p_gen.append(p_gen.squeeze(-1))
    gen_ids = torch.cat(all_gen_ids, dim=0)
    p_gen_all = torch.cat(all_p_gen, dim=0)

    results: list[dict] = []
    n_cos_pass = 0
    n_grammar_pass = 0
    n_exact = 0
    n_src_in = 0
    n_tgt_in = 0
    n_both_in = 0
    # Per-concept breakdowns since irregulars are concept-specific.
    per_concept = {c: {"n": 0, "exact": 0, "both_in": 0, "src_in": 0, "tgt_in": 0}
                   for c in NOVEL_PAIRS}
    for i in range(n_holdout):
        target_text = holdout_sents[i]
        concept, src_word, tgt_word = holdout_pairs_idx[i]
        gen_text = tok.decode(gen_ids[i].tolist(), skip_special_tokens=True)
        psi_gen = encode_pooled([gen_text], tok, mdl, args.device,
                                max_length=args.t_max)[0]
        cos_recovered = float(F.cosine_similarity(
            psi_gen, holdout_psi[i], dim=0,
        ).item())
        n_errors, grammar_pass = grammar_grade(gen_text)
        proxy = grammar_proxy(gen_text)
        exact = gen_text.strip() == target_text.strip()
        src_in, tgt_in, both_in = word_pair_fidelity(src_word, tgt_word, gen_text)

        if cos_recovered >= args.cos_min:
            n_cos_pass += 1
        if grammar_pass:
            n_grammar_pass += 1
        if exact:
            n_exact += 1
        if src_in:
            n_src_in += 1
        if tgt_in:
            n_tgt_in += 1
        if both_in:
            n_both_in += 1

        per_concept[concept]["n"] += 1
        per_concept[concept]["exact"] += int(exact)
        per_concept[concept]["both_in"] += int(both_in)
        per_concept[concept]["src_in"] += int(src_in)
        per_concept[concept]["tgt_in"] += int(tgt_in)

        real_mask = (holdout_ids[i] != pad_id).float()
        p_gen_mean = float((p_gen_all[i] * real_mask).sum().item() /
                           real_mask.sum().clamp(min=1.0).item())

        results.append({
            "target": target_text,
            "generated": gen_text,
            "concept": concept,
            "src_word": src_word,
            "tgt_word": tgt_word,
            "src_in_gen": src_in,
            "tgt_in_gen": tgt_in,
            "both_in_gen": both_in,
            "cos_recovered": cos_recovered,
            "lt_n_errors": n_errors,
            "grammar_pass": grammar_pass,
            "grammar_proxy": proxy,
            "exact_match": exact,
            "p_gen_mean": p_gen_mean,
        })

    # First 1 sentence per truly-novel pair (28 total) for clear visibility.
    print(f"\nPer-pair sample (one sentence per truly-novel pair):")
    seen_pairs: set[tuple[str, str, str]] = set()
    for i, r in enumerate(results):
        key = (r["concept"], r["src_word"], r["tgt_word"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        cos_mark = "✓" if r["cos_recovered"] >= args.cos_min else "✗"
        gr_mark = "✓" if r["grammar_pass"] else "✗"
        wf_mark = "✓" if r["both_in_gen"] else (
            "≈" if r["src_in_gen"] or r["tgt_in_gen"] else "✗"
        )
        em = " (exact)" if r["exact_match"] else ""
        print(f"  [{r['concept']:<11} | {r['src_word']!r:>10}→{r['tgt_word']!r:<11}] "
              f"target:    {r['target']!r}")
        print(f"      generated: {r['generated']!r}{em}")
        print(f"      src_in={r['src_in_gen']}  tgt_in={r['tgt_in_gen']}  "
              f"p_gen={r['p_gen_mean']:.3f}  "
              f"{cos_mark} cos={r['cos_recovered']:.3f}  "
              f"{wf_mark} fidelity={'BOTH' if r['both_in_gen'] else ('SOME' if r['src_in_gen'] or r['tgt_in_gen'] else 'NONE')}")

    print(f"\nPer-concept aggregate (across all 7 templates):")
    print(f"  {'concept':<12}  {'n':>4}  {'exact':>10}  {'both_in':>10}  "
          f"{'src_in':>10}  {'tgt_in':>10}")
    for concept, agg in per_concept.items():
        print(f"  {concept:<12}  {agg['n']:>4}  "
              f"{agg['exact']:>3}/{agg['n']:>3}    "
              f"{agg['both_in']:>3}/{agg['n']:>3}    "
              f"{agg['src_in']:>3}/{agg['n']:>3}    "
              f"{agg['tgt_in']:>3}/{agg['n']:>3}")

    # ---- Verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    coses = sorted(r["cos_recovered"] for r in results)
    median_cos = coses[len(coses) // 2]
    grammar_target = int(args.grammar_pass_rate * n_holdout)
    word_fid_target = int(args.word_fidelity_min * n_holdout)
    cos_gate = median_cos >= args.cos_min
    grammar_gate = n_grammar_pass >= grammar_target
    word_fid_gate = n_both_in >= word_fid_target
    p_gen_overall = float(sum(r["p_gen_mean"] for r in results) / len(results))

    print(f"Median cos:                      {median_cos:.4f}  "
          f"(target ≥ {args.cos_min})")
    print(f"Sentences passing cos:           {n_cos_pass}/{n_holdout}")
    print(f"Sentences passing grammar:       {n_grammar_pass}/{n_holdout}  "
          f"(target ≥ {grammar_target})")
    print(f"Sentences with src word in gen:  {n_src_in}/{n_holdout}")
    print(f"Sentences with tgt word in gen:  {n_tgt_in}/{n_holdout}")
    print(f"Sentences with BOTH (gate):      {n_both_in}/{n_holdout}  "
          f"(target ≥ {word_fid_target})")
    print(f"Exact target matches:            {n_exact}/{n_holdout}")
    print(f"Mean p_gen:                      {p_gen_overall:.3f}")

    if cos_gate and grammar_gate and word_fid_gate:
        verdict = "TRULY_NOVEL_PASS"
        message = (
            "Pointer-Generator generalizes to TRULY NOVEL word pairs that "
            "the model never saw during training. The copy mechanism is a "
            "structural fix, not a memorization trick. Approach B with "
            "sequence conditioning + pointer-generator is locked. Next: "
            "revise plan §19.12 to reflect this architecture and proceed "
            "to the production Phase 2a build."
        )
    elif cos_gate and grammar_gate and not word_fid_gate:
        verdict = "TRULY_NOVEL_FAIL"
        message = (
            "On truly-novel pairs (irregulars + new vocabulary), word "
            "fidelity drops below the gate. Pointer engaged on near-"
            "distribution data (2a.0e PASS) but doesn't generalize to "
            "this distribution. Diagnose: (a) does p_gen behave the same "
            "as 2a.0e? (b) is the copy attention attending to the right "
            "input positions for novel words? (c) does encoder vocab "
            "include these tokens cleanly?"
        )
    else:
        verdict = "OTHER_GATES_FAILED"
        message = (
            "Cos or grammar gate failed on truly-novel set. Investigate "
            "before drawing conclusions about the architecture."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    payload = {
        "task": "2a.0f",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "novel_pairs": {c: pairs for c, pairs in NOVEL_PAIRS.items()},
        "corpus": corpus_stats,
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
        },
        "thresholds": {
            "cos_min": args.cos_min,
            "grammar_pass_rate": args.grammar_pass_rate,
            "word_fidelity_min": args.word_fidelity_min,
        },
        "loss_history": loss_history,
        "summary": {
            "median_cos": median_cos,
            "n_cos_pass": n_cos_pass,
            "n_grammar_pass": n_grammar_pass,
            "n_src_in": n_src_in,
            "n_tgt_in": n_tgt_in,
            "n_both_in": n_both_in,
            "n_exact_match": n_exact,
            "p_gen_overall": p_gen_overall,
            "n_holdout": n_holdout,
        },
        "per_concept": per_concept,
        "verdict": verdict,
        "message": message,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if (cos_gate and grammar_gate and word_fid_gate) else 1)


if __name__ == "__main__":
    main()
