"""Stage 3 / Sub-task 3.1 — cross-domain validation on definitional sentences.

The first concrete experiment of Stage 3 (universal domain ingestion).
~3-5 hr GPU.

Question: does Phase 2a's locked architecture (sequence conditioning +
Pointer-Generator + paper recipe) transfer to a NEW domain with
DIFFERENT semantic structure than Phase 2a's morphological transformations?

Phase 2a's domains were morphological:
  - plural:      cat → cats         (suffix +s)
  - past_tense:  walk → walked      (suffix +ed)
  - comparative: big → bigger       (suffix +er)
  - opposite:    hot → cold         (semantic inversion)

This sub-task tests a NEW kind of relation:
  - definitional: subject → category    (is-a relation)
    apple is a fruit
    tiger is an animal
    chair is a piece of furniture

Different semantic relation. Same sentence-shape. Same architecture.

Setup:
  - 50 training pairs across 10 categories (fruit, vegetable, animal,
    furniture, instrument, color, vehicle, building, sport, weather)
    × 10 templates → 500 train sentences
  - 12 truly-novel held-out pairs (mango, penguin, helicopter, ...) from
    same categories × 10 templates → 120 held-out sentences
  - Train decoder FROM SCRATCH (not fine-tune from Phase 2a's checkpoint)
    so we test the recipe's reproducibility, not just transfer

Acceptance gates (slightly more lenient than 2a.3's production
gates since this is a first cross-domain test):
  - Median cos ≥ 0.85 on held-out (vs 2a.3's 0.90)
  - Sentences passing grammar gate ≥ 95% (same as 2a.3)
  - Sentences with BOTH subject + category words ≥ 70% (vs 2a.3's 80%)

If 3.1 passes: architectural-transfer claim is empirically validated.
Proceed to 3.2 (per-domain energy model) and 3.3 (domain registry).

If 3.1 fails: diagnose. Likely candidates: corpus too small, templates
too narrow, semantic relation harder than morphology, or the recipe
needs domain-specific tuning. Plan §19.16 will be revised based on
the failure mode.

Run on the GPU box (~3-5 hr):
  python scripts/stage3_1_definitional.py
  python scripts/stage3_1_definitional.py --steps 20000  # longer run
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

from selflearnai.generator import (
    PointerSeqCondDecoder,
    perturb_h,
    mixture_nll,
    GenerationVerdict,
    word_pair_fidelity,
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


# ---------------------------------------------------------------------------
# Definitional corpus — 50 train pairs across 10 categories
# ---------------------------------------------------------------------------

TRAIN_PAIRS: list[tuple[str, str]] = [
    # fruit (5)
    ("apple",     "fruit"),
    ("banana",    "fruit"),
    ("orange",    "fruit"),
    ("grape",     "fruit"),
    ("lemon",     "fruit"),
    # vegetable (5)
    ("carrot",    "vegetable"),
    ("spinach",   "vegetable"),
    ("potato",    "vegetable"),
    ("broccoli",  "vegetable"),
    ("lettuce",   "vegetable"),
    # animal (5)
    ("tiger",     "animal"),
    ("dolphin",   "animal"),
    ("eagle",     "animal"),
    ("rabbit",    "animal"),
    ("snake",     "animal"),
    # furniture (5)
    ("chair",     "furniture"),
    ("table",     "furniture"),
    ("sofa",      "furniture"),
    ("desk",      "furniture"),
    ("bed",       "furniture"),
    # instrument (5)
    ("piano",     "instrument"),
    ("guitar",    "instrument"),
    ("drum",      "instrument"),
    ("violin",    "instrument"),
    ("flute",     "instrument"),
    # color (5)
    ("red",       "color"),
    ("blue",      "color"),
    ("green",     "color"),
    ("yellow",    "color"),
    ("purple",    "color"),
    # vehicle (5)
    ("car",       "vehicle"),
    ("bus",       "vehicle"),
    ("plane",     "vehicle"),
    ("train",     "vehicle"),
    ("bicycle",   "vehicle"),
    # building (5)
    ("house",     "building"),
    ("school",    "building"),
    ("hospital",  "building"),
    ("library",   "building"),
    ("museum",    "building"),
    # sport (5)
    ("tennis",    "sport"),
    ("soccer",    "sport"),
    ("basketball","sport"),
    ("swimming",  "sport"),
    ("running",   "sport"),
    # weather (5)
    ("rain",      "weather"),
    ("snow",      "weather"),
    ("wind",      "weather"),
    ("sunshine",  "weather"),
    ("fog",       "weather"),
]

# 12 truly-novel pairs — same categories, different items, NEVER in train.
TRULY_NOVEL_PAIRS: list[tuple[str, str]] = [
    ("mango",      "fruit"),
    ("peach",      "fruit"),
    ("cabbage",    "vegetable"),
    ("penguin",    "animal"),
    ("bookcase",   "furniture"),
    ("trumpet",    "instrument"),
    ("white",      "color"),
    ("black",      "color"),
    ("helicopter", "vehicle"),
    ("church",     "building"),
    ("cricket",    "sport"),
    ("thunder",    "weather"),
]

TEMPLATES: list[str] = [
    "{subject} is a {category}",
    "a {subject} is a {category}",
    "{subject} is a type of {category}",
    "{subject} is a kind of {category}",
    "we call {subject} a {category}",
    "the {subject} is a {category}",
    "{subject} belongs to {category}",
    "{subject} is an example of {category}",
    "every {subject} is a {category}",
    "{subject} is part of {category}",
]


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

def audit_no_overlap() -> None:
    """FATAL if any TRULY_NOVEL pair overlaps with TRAIN_PAIRS. Mirrors
    the 2a.0f / 2a.1 audit shape."""
    train_set = set(TRAIN_PAIRS)
    leaks = [p for p in TRULY_NOVEL_PAIRS if p in train_set]
    if leaks:
        print(f"FATAL: TRULY_NOVEL pairs leak into train: {leaks}")
        raise SystemExit(1)
    # Also check train pairs unique
    if len(set(TRAIN_PAIRS)) != len(TRAIN_PAIRS):
        print(f"FATAL: train pairs contain duplicates")
        raise SystemExit(1)


def render(pairs: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for subj, cat in pairs:
        for ti, tpl in enumerate(TEMPLATES):
            rows.append({
                "subject":      subj,
                "category":     cat,
                "template_idx": ti,
                "sentence":     tpl.format(subject=subj, category=cat),
            })
    return rows


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # Training
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
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
    # Acceptance — slightly more lenient than 2a.3 (first cross-domain test)
    parser.add_argument("--cos-min", type=float, default=0.85)
    parser.add_argument("--grammar-pass-rate", type=float, default=0.95)
    parser.add_argument("--word-fidelity-min", type=float, default=0.70)
    parser.add_argument("--out", default="results/stage3/definitional.json")
    parser.add_argument("--ckpt", default="data/explanations_v2/checkpoints/decoder_3a1.pt")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    use_mse = not args.no_mse and args.mse_weight > 0.0
    use_perturb = not args.no_perturb

    enc_cfg = ENCODERS[args.encoder]
    print("Stage 3 / Sub-task 3.1 — cross-domain validation (definitional)")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"NEW domain: definitional 'X is a Y' relations (different from Phase 2a)")

    # ---- Audits + corpus build ---------------------------------------
    print("\n[1] Leakage audit")
    print("-" * 78)
    audit_no_overlap()
    print(f"  ✓ {len(TRAIN_PAIRS)} train pairs, {len(TRULY_NOVEL_PAIRS)} truly-novel held-out, no overlap")

    train_rows = render(TRAIN_PAIRS)
    holdout_rows = render(TRULY_NOVEL_PAIRS)
    train_sents = [r["sentence"] for r in train_rows]
    holdout_sents = [r["sentence"] for r in holdout_rows]
    n_train = len(train_sents)
    n_holdout = len(holdout_sents)
    print(f"\n  corpus: {n_train} train + {n_holdout} held-out")
    print(f"  templates per pair: {len(TEMPLATES)}")
    print(f"  samples (train):")
    for s in train_sents[:3]:
        print(f"    {s!r}")
    print(f"  samples (held-out — truly novel):")
    for s in holdout_sents[:3]:
        print(f"    {s!r}")

    # ---- Encoder ------------------------------------------------------
    print(f"\n[2] Loading {enc_cfg['model']} ...")
    print("-" * 78)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]

    # ---- Encode -------------------------------------------------------
    print(f"\n[3] Encoding train + holdout")
    print("-" * 78)
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
    longest = int(train_tok.attention_mask.sum(dim=-1).max().item())
    print(f"  longest train tokenized: {longest} (t_max={args.t_max})")

    # ---- Decoder (FROM SCRATCH, locked architecture) -----------------
    torch.manual_seed(args.seed)
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n[4] Decoder: {n_params/1e6:.2f}M params (locked architecture, fresh init)")
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
    print(f"\n[5] Training {args.steps} steps  batch={args.batch_size}  "
          f"lr={args.lr} (warmup {args.warmup_steps})")
    print("-" * 78)
    decoder.train()
    loss_history = []
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

        log_probs, hidden_out, p_gen = decoder(h_input, h_mask_batch, ids_batch)
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
                "step": step, "loss": float(loss.item()),
                "nll": float(nll.item()), "mse": float(mse_loss.item()),
                "p_gen_mean": float(p_gen.mean().item()),
                "lr": float(opt.param_groups[0]["lr"]),
            })
            print(f"  step {step:>5}/{args.steps}  loss={loss.item():.4f}  "
                  f"nll={nll.item():.4f}  mse={mse_loss.item():.4f}  "
                  f"p_gen={p_gen.mean().item():.3f}")

    # Save checkpoint
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), args.ckpt)
    print(f"\nDecoder checkpoint saved to {args.ckpt}")

    # ---- Held-out eval -----------------------------------------------
    print("\n" + "=" * 78)
    print(f"[6] HELD-OUT EVALUATION ({n_holdout} truly-novel sentences)")
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

    verdicts = []
    per_category = {}
    for i in range(n_holdout):
        row = holdout_rows[i]
        gen_text = gen_texts[i]
        cos = float(cos_recovered[i].item())
        n_errors, gpass = grammar_grade(gen_text)
        proxy = grammar_proxy(gen_text)
        # In definitional, "src" = subject, "tgt" = category
        src_in, tgt_in, both_in = word_pair_fidelity(row["subject"], row["category"], gen_text)
        exact = gen_text.strip() == row["sentence"].strip()
        real_mask = (holdout_ids[i] != pad_id).float()
        p_gen_mean = float((p_gen_all[i] * real_mask).sum().item() /
                           real_mask.sum().clamp(min=1.0).item())
        verdicts.append(GenerationVerdict(
            target=row["sentence"], generated=gen_text,
            concept="definitional",
            src_word=row["subject"], tgt_word=row["category"],
            cos_recovered=cos, grammar_pass=gpass, grammar_n_errors=n_errors,
            grammar_proxy=proxy,
            src_in_gen=src_in, tgt_in_gen=tgt_in, both_in_gen=both_in,
            exact_match=exact, p_gen_mean=p_gen_mean,
        ))
        cat = row["category"]
        agg = per_category.setdefault(
            cat, {"n": 0, "cos_pass": 0, "grammar_pass": 0,
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

    # Per-category breakdown
    print(f"\nPer-category aggregate (across {n_holdout} held-out):")
    print(f"  {'category':<12}  {'n':>3}  {'cos_avg':>8}  "
          f"{'cos_pass':>8}  {'grammar':>7}  {'both_in':>8}  {'exact':>7}")
    for cat in sorted(per_category.keys()):
        agg = per_category[cat]
        n_c = agg["n"]
        cos_avg = agg["cos_sum"] / n_c
        print(f"  {cat:<12}  {n_c:>3}  {cos_avg:>8.4f}  "
              f"{agg['cos_pass']:>3}/{n_c:<3}  "
              f"{agg['grammar_pass']:>3}/{n_c:<3}  "
              f"{agg['both_in']:>3}/{n_c:<3}  "
              f"{agg['exact']:>3}/{n_c:<3}")

    # Show all 12 truly-novel pairs (one sample each)
    print(f"\nPer-pair sample (one sentence per truly-novel pair):")
    seen_pairs = set()
    for v in verdicts:
        key = (v.src_word, v.tgt_word)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        cos_mark = "✓" if v.cos_recovered >= args.cos_min else "✗"
        wf_mark = "✓" if v.both_in_gen else (
            "≈" if v.src_in_gen or v.tgt_in_gen else "✗"
        )
        em = " (exact)" if v.exact_match else ""
        print(f"  [{v.src_word!r:>12}→{v.tgt_word!r:<11}] target:    {v.target!r}")
        print(f"      generated: {v.generated!r}{em}")
        print(f"      {cos_mark} cos={v.cos_recovered:.4f}  "
              f"{wf_mark} fidelity="
              f"{'BOTH' if v.both_in_gen else ('SOME' if v.src_in_gen or v.tgt_in_gen else 'NONE')}  "
              f"p_gen={v.p_gen_mean:.3f}")

    # ---- Verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT (Stage 3.1 — cross-domain validation)")
    print("=" * 78)
    gates = roll_up_gates(
        verdicts,
        cos_min=args.cos_min,
        grammar_pass_rate=args.grammar_pass_rate,
        word_fidelity_min=args.word_fidelity_min,
    )
    for line in gates.summary_lines():
        print(line)

    # Comparison to Phase 2a 2a.3 production numbers (for context)
    print(f"\n  Phase 2a (2a.3) reference: median cos 1.000, grammar 99.8%, "
          f"word-fidelity 86.1%")

    if gates.all_pass:
        verdict = "STAGE_3_1_PASS"
        message = (
            f"The Phase 2a recipe trained from scratch on a NEW domain "
            f"(definitional 'is-a' relations) hits comparable numbers "
            f"to Phase 2a's morphological domains. The architecture "
            f"transfers — same locked architecture, same loss, different "
            f"semantic relation, comparable result. Universal-domain-"
            f"ingestion thesis empirically supported. Next: 3.2 "
            f"(per-domain energy model) and 3.3 (domain registry)."
        )
    elif gates.cos_gate and gates.grammar_gate and not gates.word_fidelity_gate:
        verdict = "STAGE_3_1_WORD_FIDELITY_BELOW"
        message = (
            f"Cos and grammar pass but word-fidelity below 70%. The "
            f"definitional domain may need more training data or longer "
            f"training. Per-category breakdown above will show which "
            f"categories drag the average down. Diagnose before "
            f"committing to 3.2."
        )
    elif not gates.cos_gate:
        verdict = "STAGE_3_1_COS_BELOW"
        message = (
            f"Cos below 0.85 — model isn't generating semantically-"
            f"matching text. Check: (a) is the corpus too small (500 "
            f"sentences vs 2a's 2064), (b) is the loss converging, "
            f"(c) does the architecture need different hyperparams "
            f"for non-morphological domains?"
        )
    else:
        verdict = "STAGE_3_1_MIXED_FAIL"
        message = (
            f"Multiple gates fail. Likely the recipe needs domain-"
            f"specific tuning or the corpus is too small. Diagnose "
            f"before committing to the universal pipeline."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ---------------------------------------------------
    payload = {
        "task": "3.1",
        "domain": "definitional",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "n_train_pairs": len(TRAIN_PAIRS),
        "n_truly_novel_pairs": len(TRULY_NOVEL_PAIRS),
        "n_templates": len(TEMPLATES),
        "n_train_sents": n_train,
        "n_holdout_sents": n_holdout,
        "decoder": {
            "n_params": n_params,
            "hidden_dim": args.hidden_dim,
            "t_max": args.t_max,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "ffn_mult": args.ffn_mult,
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
        "per_category": per_category,
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
        "checkpoint_path": args.ckpt,
        "results": [
            {
                "subject": v.src_word, "category": v.tgt_word,
                "target": v.target, "generated": v.generated,
                "cos_recovered": v.cos_recovered,
                "grammar_pass": v.grammar_pass,
                "lt_n_errors": v.grammar_n_errors,
                "src_in_gen": v.src_in_gen,
                "tgt_in_gen": v.tgt_in_gen,
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
