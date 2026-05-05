"""Phase 2b / Sub-task 2b.1 — multi-subword fix via subword-chaining loss.

Promotes plan §19.14's documented limitation to a concrete fix after
Stage 3.1 confirmed the issue is multi-domain (Phase 2a comparative
+ Stage 3.1 furniture/vegetable). ~1.5-2 hr GPU.

Diagnosis (combined from 2a.0f / 2a.3 / 3.1):
  Phase 2a comparative: prettier → 'isttier' (drops 'pretty' subword)
  Stage 3.1 furniture:  bookcase → 'book is a furniture' (drops '##case')
  Stage 3.1 vegetable:  cabbage  → 'cabbage is cabbage vegetable'
                                    (pointer attends wrong position)

Fix (subword-chaining auxiliary loss):
  At training positions where target token is a WordPiece continuation
  (starts with '##'), explicitly supervise the pointer attention to
  attend to the SAME position in the encoder input. For autoencoder-
  shaped tasks (target sentence == encoder input), this means
  ptr_attn[N, N] should be peaked at continuation positions.

  The vocab/MSE losses already implicitly encourage this at the OUTPUT
  level, but at continuation positions the model often gets confused
  by the template-position majority pattern. Explicit supervision
  resolves the confusion.

  Implementation:
    1. PointerSeqCondDecoder.forward(return_ptr_attn=True) exposes
       attention weights to the training loop.
    2. build_continuation_mask(target_ids, tokenizer) → bool mask.
    3. subword_chain_loss(ptr_attn, mask) = NLL of ptr_attn at the
       diagonal, restricted to continuation positions.
    4. Total loss = mixture_nll + λ_MSE·MSE + λ_chain·subword_chain.

Strategy:
  - Load Phase 2a's 2a.3 production checkpoint
  - Fine-tune for 5K steps with the new auxiliary loss
  - Evaluate on BOTH Phase 2a's 432 holdout (focus on comparative)
    AND Stage 3.1's 120 holdout (focus on furniture)
  - Acceptance: both improve materially without regressing other
    concepts/categories

Acceptance gates (HARD, all required):
  - Phase 2a comparative: 72/108 → ≥ 92/108 (+20)
  - Stage 3.1 furniture:  0/10  → ≥  8/10  (+8)
  - No regression > 5% on other concepts/categories
  - Overall §19.14 closing gates remain green on Phase 2a holdout

If 2b.1 passes:
  - Update plan §19.14's documented limitation to "RESOLVED via 2b.1"
  - Re-train Stage 3.1 with the fix and confirm furniture lifts
  - Then proceed to Stage 3.2 (energy model)

If 2b.1 fails:
  - The subword-chaining auxiliary loss isn't enough on its own
  - Next experiment: span-copy mechanism (variable-length copy spans)
    or vocabulary swap (BPE → larger units)

Run on the GPU box (~1.5-2 hr):
  python scripts/stage2b_1_subword_fix.py
  python scripts/stage2b_1_subword_fix.py --steps 10000  # longer
  python scripts/stage2b_1_subword_fix.py --chain-weight 2.0  # stronger
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
    subword_chain_loss,
    build_continuation_mask,
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


# Hardcoded reference numbers from prior runs for no-regression checks.
PHASE_2A_BASELINE = {
    "comparative": 72,    # /108
    "opposite":    108,   # /108
    "past_tense":  96,    # /108
    "plural":      96,    # /108
}
STAGE_3_1_BASELINE = {
    "animal":     10, "building":   10, "color":      20, "fruit":      20,
    "furniture":   0, "instrument": 10, "sport":      10, "vegetable":   8,
    "vehicle":    10, "weather":    10,
}


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


def build_definitional_holdout() -> tuple[list[dict], list[str]]:
    """Reconstruct Stage 3.1's truly-novel holdout for joint evaluation.
    Mirrors scripts/stage3_1_definitional.py:TRULY_NOVEL_PAIRS."""
    pairs = [
        ("mango", "fruit"), ("peach", "fruit"),
        ("cabbage", "vegetable"),
        ("penguin", "animal"),
        ("bookcase", "furniture"),
        ("trumpet", "instrument"),
        ("white", "color"), ("black", "color"),
        ("helicopter", "vehicle"),
        ("church", "building"),
        ("cricket", "sport"),
        ("thunder", "weather"),
    ]
    templates = [
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
    rows = []
    sents = []
    for subj, cat in pairs:
        for tpl in templates:
            sent = tpl.format(subject=subj, category=cat)
            rows.append({"subject": subj, "category": cat, "sentence": sent})
            sents.append(sent)
    return rows, sents


def evaluate(
    decoder, holdout_h, holdout_h_mask, holdout_ids, holdout_psi,
    holdout_pair_index,    # list of (concept_or_category, src, tgt, target_sentence)
    tok, mdl, device, t_max,
    cos_min: float, label: str,
) -> tuple[list[GenerationVerdict], dict]:
    """Run held-out eval; return (verdicts, per_group_stats)."""
    decoder.eval()
    eval_bs = 32
    n = holdout_h.size(0)
    all_gen_ids = []
    all_p_gen = []
    with torch.no_grad():
        for s in range(0, n, eval_bs):
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
                 for i in range(n)]
    psi_gen = encode_pooled(gen_texts, tok, mdl, device, max_length=t_max)
    cos_recovered = F.cosine_similarity(psi_gen, holdout_psi, dim=-1)

    verdicts: list[GenerationVerdict] = []
    per_group: dict[str, dict] = {}
    for i in range(n):
        group, src, tgt, target_sent = holdout_pair_index[i]
        gen = gen_texts[i]
        cos = float(cos_recovered[i].item())
        n_errors, gpass = grammar_grade(gen)
        proxy = grammar_proxy(gen)
        src_in, tgt_in, both_in = word_pair_fidelity(src, tgt, gen)
        exact = gen.strip() == target_sent.strip()
        real_mask = (holdout_ids[i] != pad_id).float()
        p_gen_mean = float((p_gen_all[i] * real_mask).sum().item() /
                           real_mask.sum().clamp(min=1.0).item())
        verdicts.append(GenerationVerdict(
            target=target_sent, generated=gen,
            concept=group, src_word=src, tgt_word=tgt,
            cos_recovered=cos, grammar_pass=gpass, grammar_n_errors=n_errors,
            grammar_proxy=proxy,
            src_in_gen=src_in, tgt_in_gen=tgt_in, both_in_gen=both_in,
            exact_match=exact, p_gen_mean=p_gen_mean,
        ))
        agg = per_group.setdefault(group, {"n": 0, "both_in": 0, "exact": 0})
        agg["n"] += 1
        if both_in:
            agg["both_in"] += 1
        if exact:
            agg["exact"] += 1
    print(f"\n  {label} per-group both_in:")
    for grp in sorted(per_group.keys()):
        a = per_group[grp]
        print(f"    {grp:<12}  {a['both_in']:>3}/{a['n']:<3}  exact={a['exact']}/{a['n']}")
    return verdicts, per_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt-2a3", default="data/explanations_v2/checkpoints/decoder_2a3.pt")
    parser.add_argument("--corpus-dir", default="data/explanations_v2",
                        help="Phase 2a corpus to fine-tune on (with multi-subword pairs).")
    # Training
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Architecture (must match 2a.3 checkpoint)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    # Loss recipe
    parser.add_argument("--mse-weight", type=float, default=0.5)
    parser.add_argument("--chain-weight", type=float, default=1.0,
                        help="λ for the subword-chaining auxiliary loss.")
    parser.add_argument("--no-mse", action="store_true")
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--perturb-prob", type=float, default=0.3)
    parser.add_argument("--gaussian-delta", type=float, default=0.7)
    parser.add_argument("--mask-token-rate", type=float, default=0.3)
    # Acceptance
    parser.add_argument("--phase2a-comparative-min", type=int, default=92)
    parser.add_argument("--stage3-furniture-min", type=int, default=8)
    parser.add_argument("--no-regression-margin", type=int, default=5)
    parser.add_argument("--cos-min", type=float, default=0.85)
    parser.add_argument("--out", default="results/stage2a/subword_fix.json")
    parser.add_argument("--ckpt", default="data/explanations_v2/checkpoints/decoder_2b1.pt")
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    use_mse = not args.no_mse and args.mse_weight > 0.0
    use_perturb = not args.no_perturb
    use_chain = args.chain_weight > 0.0

    enc_cfg = ENCODERS[args.encoder]
    print("Phase 2b / Sub-task 2b.1 — multi-subword fix via subword-chaining loss")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Loading 2a.3 checkpoint: {args.ckpt_2a3}")
    print(f"Loss: NLL + {args.mse_weight}·MSE + {args.chain_weight}·subword_chain")

    # ---- Encoder + corpus --------------------------------------------
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
    holdout_sents_phase2a = [r.sentence for r in holdout_rows]
    n_train = len(train_sents)
    n_holdout_phase2a = len(holdout_sents_phase2a)
    print(f"\nTraining corpus: {n_train} train sents (Phase 2a)")
    print(f"Phase 2a holdout: {n_holdout_phase2a} truly-novel sentences")

    # Stage 3.1 holdout for joint eval
    stage3_rows, stage3_sents = build_definitional_holdout()
    n_holdout_stage3 = len(stage3_sents)
    print(f"Stage 3.1 holdout: {n_holdout_stage3} definitional sentences")

    # ---- Encode + tokenize --------------------------------------------
    print("\nEncoding train + Phase 2a holdout + Stage 3.1 holdout ...")
    train_h, train_h_mask = encode_activations(train_sents, tok, mdl, args.device, t_max=args.t_max)
    h2a, m2a = encode_activations(holdout_sents_phase2a, tok, mdl, args.device, t_max=args.t_max)
    psi2a = encode_pooled(holdout_sents_phase2a, tok, mdl, args.device, max_length=args.t_max)
    h31, m31 = encode_activations(stage3_sents, tok, mdl, args.device, t_max=args.t_max)
    psi31 = encode_pooled(stage3_sents, tok, mdl, args.device, max_length=args.t_max)

    train_tok = tok(train_sents, padding="max_length", truncation=True,
                    max_length=args.t_max, return_tensors="pt").to(args.device)
    train_ids = train_tok.input_ids
    holdout_ids_2a = tok(holdout_sents_phase2a, padding="max_length", truncation=True,
                         max_length=args.t_max, return_tensors="pt").to(args.device).input_ids
    holdout_ids_3 = tok(stage3_sents, padding="max_length", truncation=True,
                        max_length=args.t_max, return_tensors="pt").to(args.device).input_ids

    # Index for evaluation: (group, src, tgt, target_sentence)
    pair_index_2a = [(r.concept, r.src, r.tgt, r.sentence) for r in holdout_rows]
    pair_index_3 = [(r["category"], r["subject"], r["category"], r["sentence"])
                    for r in stage3_rows]

    # Pre-compute continuation mask on training data (one-time cost).
    print("\nDetecting WordPiece continuation tokens in training data ...")
    train_continuation_mask = build_continuation_mask(train_ids, tok)
    n_continuation = int(train_continuation_mask.sum().item())
    n_total_tokens = int((train_ids != tok.pad_token_id).sum().item())
    print(f"  continuation tokens: {n_continuation}/{n_total_tokens} "
          f"({100*n_continuation/max(n_total_tokens,1):.1f}% of non-pad tokens)")

    # ---- Decoder + checkpoint load -----------------------------------
    torch.manual_seed(args.seed)
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim, t_max=args.t_max,
        vocab_size=tok.vocab_size, n_layers=args.n_layers,
        n_heads=args.n_heads, ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout, attn_dropout=args.attn_dropout,
    ).to(args.device)
    ckpt_path = Path(args.ckpt_2a3)
    state = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    decoder.load_state_dict(state)
    print(f"\nLoaded 2a.3 checkpoint ({sum(p.numel() for p in decoder.parameters())/1e6:.2f}M params)")

    # ---- Pre-fix BASELINE eval (sanity-check 2a.3 reproduces) ---------
    print("\n" + "=" * 78)
    print("BASELINE (2a.3 checkpoint, before fine-tuning)")
    print("=" * 78)
    _, baseline_2a = evaluate(
        decoder, h2a, m2a, holdout_ids_2a, psi2a, pair_index_2a,
        tok, mdl, args.device, args.t_max, args.cos_min,
        label="Phase 2a holdout",
    )
    _, baseline_3 = evaluate(
        decoder, h31, m31, holdout_ids_3, psi31, pair_index_3,
        tok, mdl, args.device, args.t_max, args.cos_min,
        label="Stage 3.1 holdout",
    )

    # ---- Fine-tune ----------------------------------------------------
    opt = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        return args.lr

    print(f"\n" + "=" * 78)
    print(f"FINE-TUNING with subword-chaining loss")
    print("=" * 78)
    print(f"  steps={args.steps}  batch={args.batch_size}  lr={args.lr}  "
          f"warmup={args.warmup_steps}  chain_weight={args.chain_weight}")
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
        cont_mask_batch = train_continuation_mask[idx]

        h_input = perturb_h(
            h_batch, h_mask_batch,
            apply_prob=(args.perturb_prob if use_perturb else 0.0),
            gaussian_delta=args.gaussian_delta,
            mask_token_rate=args.mask_token_rate,
        )

        log_probs, hidden_out, p_gen, ptr_attn = decoder(
            h_input, h_mask_batch, ids_batch, return_ptr_attn=True,
        )
        nll = mixture_nll(log_probs, ids_batch)
        if use_mse:
            mse_loss = mse_activation_loss(
                hidden_out, decoder.mse_proj, h_batch, h_mask_batch,
            )
        else:
            mse_loss = torch.tensor(0.0, device=args.device)
        if use_chain:
            chain_loss = subword_chain_loss(ptr_attn, cont_mask_batch)
        else:
            chain_loss = torch.tensor(0.0, device=args.device)
        loss = nll + args.mse_weight * mse_loss + args.chain_weight * chain_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            loss_history.append({
                "step": step, "loss": float(loss.item()),
                "nll": float(nll.item()), "mse": float(mse_loss.item()),
                "chain": float(chain_loss.item()),
                "p_gen_mean": float(p_gen.mean().item()),
                "lr": float(opt.param_groups[0]["lr"]),
            })
            print(f"  step {step:>4}/{args.steps}  "
                  f"loss={loss.item():.4f}  nll={nll.item():.4f}  "
                  f"mse={mse_loss.item():.4f}  chain={chain_loss.item():.4f}  "
                  f"p_gen={p_gen.mean().item():.3f}")

    # Save fine-tuned checkpoint
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), args.ckpt)
    print(f"\nFine-tuned checkpoint saved to {args.ckpt}")

    # ---- Post-fix eval ------------------------------------------------
    print("\n" + "=" * 78)
    print("POST-FIX EVALUATION")
    print("=" * 78)
    verdicts_2a, after_2a = evaluate(
        decoder, h2a, m2a, holdout_ids_2a, psi2a, pair_index_2a,
        tok, mdl, args.device, args.t_max, args.cos_min,
        label="Phase 2a holdout",
    )
    verdicts_3, after_3 = evaluate(
        decoder, h31, m31, holdout_ids_3, psi31, pair_index_3,
        tok, mdl, args.device, args.t_max, args.cos_min,
        label="Stage 3.1 holdout",
    )

    # ---- Comparison + verdict ----------------------------------------
    print("\n" + "=" * 78)
    print("DELTAS (Phase 2a)")
    print("=" * 78)
    for grp in sorted(set(after_2a) | set(PHASE_2A_BASELINE)):
        a = after_2a.get(grp, {"n": 0, "both_in": 0})
        before = PHASE_2A_BASELINE.get(grp, 0)
        delta = a["both_in"] - before
        print(f"  {grp:<12}  before={before:>3}  after={a['both_in']:>3}  Δ={delta:+d}")

    print("\n" + "=" * 78)
    print("DELTAS (Stage 3.1)")
    print("=" * 78)
    for grp in sorted(set(after_3) | set(STAGE_3_1_BASELINE)):
        a = after_3.get(grp, {"n": 0, "both_in": 0})
        before = STAGE_3_1_BASELINE.get(grp, 0)
        delta = a["both_in"] - before
        print(f"  {grp:<12}  before={before:>3}  after={a['both_in']:>3}  Δ={delta:+d}")

    # Acceptance checks
    comp_after = after_2a.get("comparative", {}).get("both_in", 0)
    furn_after = after_3.get("furniture", {}).get("both_in", 0)
    comp_pass = comp_after >= args.phase2a_comparative_min
    furn_pass = furn_after >= args.stage3_furniture_min

    no_regression = True
    regress_msgs = []
    for grp in PHASE_2A_BASELINE:
        a = after_2a.get(grp, {"n": 0, "both_in": 0})
        before = PHASE_2A_BASELINE[grp]
        if a["both_in"] < before - args.no_regression_margin:
            no_regression = False
            regress_msgs.append(f"Phase 2a {grp} regressed: {a['both_in']} < {before - args.no_regression_margin}")
    for grp in STAGE_3_1_BASELINE:
        a = after_3.get(grp, {"n": 0, "both_in": 0})
        before = STAGE_3_1_BASELINE[grp]
        if a["both_in"] < before - args.no_regression_margin:
            no_regression = False
            regress_msgs.append(f"Stage 3.1 {grp} regressed: {a['both_in']} < {before - args.no_regression_margin}")

    print("\n" + "=" * 78)
    print("VERDICT (Phase 2b.1 — multi-subword fix)")
    print("=" * 78)
    print(f"  comparative: {comp_after}/108 (target ≥ {args.phase2a_comparative_min})  "
          f"{'PASS' if comp_pass else 'FAIL'}")
    print(f"  furniture:   {furn_after}/10  (target ≥ {args.stage3_furniture_min})   "
          f"{'PASS' if furn_pass else 'FAIL'}")
    print(f"  no-regression (margin {args.no_regression_margin}): "
          f"{'PASS' if no_regression else 'FAIL'}")
    if regress_msgs:
        for m in regress_msgs:
            print(f"    · {m}")

    if comp_pass and furn_pass and no_regression:
        verdict = "STAGE_2B_1_PASS"
        message = (
            f"Subword-chaining loss fixed the multi-subword issue across "
            f"BOTH Phase 2a (comparative {72} → {comp_after}) AND "
            f"Stage 3.1 (furniture {0} → {furn_after}) without regressing "
            f"other groups. Plan §19.14's documented limitation is RESOLVED. "
            f"Update the package to enable subword-chaining by default in "
            f"production training, then proceed to 3.2 (energy model)."
        )
    elif comp_pass and not furn_pass:
        verdict = "PHASE_2A_FIXED_STAGE3_PARTIAL"
        message = (
            f"Phase 2a fixed but Stage 3.1 furniture didn't recover. "
            f"Either chain-weight is too low for that domain, or fine-"
            f"tuning on Phase 2a's corpus alone doesn't transfer the fix "
            f"to Stage 3.1. Try retraining Stage 3.1 with the new loss "
            f"from scratch."
        )
    elif not comp_pass and furn_pass:
        verdict = "STAGE3_FIXED_PHASE_2A_NOT"
        message = (
            f"Unusual: Stage 3.1 furniture fixed but Phase 2a comparative "
            f"didn't. Diagnose by inspecting per-pair outputs."
        )
    else:
        verdict = "STAGE_2B_1_INSUFFICIENT"
        message = (
            f"Subword-chaining loss didn't materially fix either domain. "
            f"Try: (a) higher chain-weight, (b) longer training, (c) the "
            f"more invasive span-copy mechanism in 2b.2."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ----------------------------------------------------
    payload = {
        "task": "2b.1",
        "encoder": args.encoder,
        "ckpt_2a3_loaded": str(ckpt_path),
        "ckpt_2b1_saved": args.ckpt,
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "chain_weight": args.chain_weight,
            "use_chain": use_chain,
            "use_mse": use_mse,
            "use_perturb": use_perturb,
        },
        "thresholds": {
            "phase2a_comparative_min": args.phase2a_comparative_min,
            "stage3_furniture_min": args.stage3_furniture_min,
            "no_regression_margin": args.no_regression_margin,
        },
        "loss_history": loss_history,
        "phase2a_baseline": PHASE_2A_BASELINE,
        "phase2a_after": after_2a,
        "stage3_baseline": STAGE_3_1_BASELINE,
        "stage3_after": after_3,
        "summary": {
            "comparative_after": comp_after,
            "furniture_after": furn_after,
            "comp_pass": comp_pass,
            "furn_pass": furn_pass,
            "no_regression": no_regression,
        },
        "verdict": verdict,
        "message": message,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if (comp_pass and furn_pass and no_regression) else 1)


if __name__ == "__main__":
    main()
