"""Phase 2a sub-task 2a.0d — sequence-conditioning fix + word-fidelity gate.

Empirical fix for the 2a.0c failure mode. ~1-2 hr GPU.

Diagnosis from 2a.0c (commit 89f3216): the decoder learned TEMPLATE
structure but NOT word-pair fidelity. Held-out generated 'the plural
of leg is hands' when target was 'the plural of book is books' —
0/168 exact matches. The cos≥0.85 gate passed (median 0.91) because
two sentences with the same template but different word pairs have
similar pooled ψ. The cos gate measures template+domain similarity,
not content fidelity.

Research findings (this commit's PR):
  - Morris et al 2023 (vec2text): 92% bit-exact reconstruction from
    a single sentence embedding via iterative refinement + AR decoder.
    The iterative-refinement IDEA is architecture-agnostic but their
    T5 decoder violates our no-AR rule.
  - Cosmos (Meshchaninov 2025): does NOT pool. Their Perceiver
    Resampler queries the FULL sequence of encoder activations
    h ∈ R^{L×768}. Token-level information preserved by design.
  - Conclusion: pooling ψ to a single vector was the v1 bottleneck.
    Sequence conditioning preserves the word-level info that
    Approach A's CAPACITY_PASS proved E5 can carry.

The fix: replace single-pooled-ψ conditioning with the full encoder
activation sequence h. Cross-attention from learnable output
positions to h. Same loss recipe (CE + activation MSE + perturbation
+ feature dropout), same corpus, same training schedule.

The eval: HARDENED with word-pair fidelity gate. Tracks not just cos
and grammar, but also whether the held-out source AND target words
appear in the generated text. Without this gate, 2a.0c would have
been declared PASS despite producing wrong words.

Acceptance gates (HARD, all three required):
  - Median cos ≥ 0.85 on held-out
  - ≥ 95% sentences pass grammar gate
  - **NEW**: ≥ 70% of sentences contain BOTH target source word AND
    target word (the actual word-pair fidelity metric we should have
    had from the start).

Run on the GPU box:
  python scripts/stage2a_seq_conditioning.py
  python scripts/stage2a_seq_conditioning.py --steps 30000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn, read_pairs
from scripts.stage2a_quick_probe import grammar_grade, grammar_proxy
from scripts.stage2a_generalization import (
    CONCEPT_DATA_DIRS,
    TEMPLATES,
    build_corpus,
    encode_pooled,
    encode_activations,
)


# ---------------------------------------------------------------------------
# Sequence-conditioned decoder (replaces 2a.0c's single-ψ projection)
# ---------------------------------------------------------------------------

class SeqCondDecoder(nn.Module):
    """Non-AR transformer decoder conditioned on the FULL encoder
    activation sequence h ∈ R^{B×T_in×D_enc} via cross-attention.

    Forward pass:
      1. Project h: D_enc → hidden_dim per token.
      2. Optional feature dropout on the projected memory (paper §4.2.2).
      3. T_out learnable output position embeddings.
      4. nn.TransformerDecoder layers — each layer has:
           - bidirectional self-attention on output positions
             (NO causal mask — this is non-autoregressive)
           - cross-attention from output positions → memory tokens
      5. Linear head → vocab logits per output position.

    The cross-attention is the key piece: every output position can
    attend to every encoder activation, so per-token information
    (specific words like "book", "books") flows directly into the
    decoder's working representation.
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        hidden_dim: int = 512,
        t_max: int = 32,
        vocab_size: int = 30522,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_mult: int = 4,
        feat_dropout: float = 0.4,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.t_max = t_max
        self.vocab_size = vocab_size
        # Per-token projection of encoder activations into decoder space.
        self.cond_proj = nn.Linear(encoder_dim, hidden_dim)
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        # Output position embeddings (learned).
        self.output_pos_emb = nn.Parameter(
            torch.randn(t_max, hidden_dim) * 0.02
        )
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            batch_first=True,
            activation="gelu",
            dropout=attn_dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.token_head = nn.Linear(hidden_dim, vocab_size)
        # Project decoder hidden → encoder dim for activation-MSE loss.
        self.mse_proj = nn.Linear(hidden_dim, encoder_dim)

    def forward(
        self,
        encoder_h: torch.Tensor,                       # [B, T_in, D_enc]
        encoder_mask: torch.Tensor,                    # [B, T_in], 1 for real tokens
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = encoder_h.size(0)
        memory = self.feat_dropout(self.cond_proj(encoder_h))   # [B, T_in, h]
        memory_pad_mask = (encoder_mask < 0.5)                  # True for pads
        out_seeds = self.output_pos_emb.unsqueeze(0).expand(B, -1, -1)
        # tgt_mask=None → no causal masking → bidirectional self-attn.
        out = self.transformer(
            tgt=out_seeds,
            memory=memory,
            memory_key_padding_mask=memory_pad_mask,
        )                                                       # [B, T_out, h]
        logits = self.token_head(out)
        return logits, out


# ---------------------------------------------------------------------------
# Sequence-level perturbation augmentation (paper §4.2.2)
# ---------------------------------------------------------------------------

def perturb_h(
    h: torch.Tensor,                              # [B, T_in, D_enc]
    h_mask: torch.Tensor,                          # [B, T_in]
    *,
    apply_prob: float = 0.5,
    gaussian_delta: float = 0.7,
    mask_token_rate: float = 0.3,
) -> torch.Tensor:
    """Per-sample perturbation:
      - 50% of samples get perturbed
      - When perturbed, 50/50 between Gaussian-noise on every real
        position OR token-level masking (set entire token vector to
        zero) at mask_token_rate=30% of real positions.
    Padding positions are never perturbed.
    """
    B, T, D = h.shape
    device = h.device
    apply = torch.rand(B, device=device) < apply_prob
    if not apply.any():
        return h
    mode_a = torch.rand(B, device=device) < 0.5
    out = h.clone()
    # MODE A: Gaussian noise on all real positions of selected samples.
    a_idx = (apply & mode_a).nonzero(as_tuple=True)[0]
    if a_idx.numel() > 0:
        # Sample noise on the same shape as those rows.
        noise = torch.randn_like(out[a_idx])
        # Apply only to real positions (mask out padding).
        m = h_mask[a_idx].unsqueeze(-1)        # [b, T, 1]
        new = gaussian_delta * out[a_idx] + (1.0 - gaussian_delta ** 2) ** 0.5 * noise
        out[a_idx] = m * new + (1.0 - m) * out[a_idx]
    # MODE B: token-level masking — zero out 30% of real positions.
    b_idx = (apply & ~mode_a).nonzero(as_tuple=True)[0]
    if b_idx.numel() > 0:
        # Random mask: 1 = keep, 0 = drop. Only consider real positions.
        keep = (torch.rand(b_idx.numel(), T, device=device) >= mask_token_rate).float()
        keep = keep * h_mask[b_idx]            # never mask padding (it's already 0)
        # For padding positions, keep_factor stays mask_factor; multiplying
        # only the kept tokens.
        out[b_idx] = out[b_idx] * keep.unsqueeze(-1) * h_mask[b_idx].unsqueeze(-1) \
                     + out[b_idx] * (1.0 - h_mask[b_idx].unsqueeze(-1))
    return out


# ---------------------------------------------------------------------------
# Word fidelity metric — the 2a.0c-blind-spot fix
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z]+")


def words_in_text(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def word_pair_fidelity(
    src: str, tgt: str, generated_text: str,
) -> tuple[bool, bool, bool]:
    """Returns (src_in_gen, tgt_in_gen, both_in_gen).

    A correct held-out generation should contain BOTH the source and
    target words from the held-out word pair. 2a.0c had 0/168 exact
    matches and the per-sentence outputs showed it was producing
    valid templates with wrong word pairs — exactly what this metric
    catches.
    """
    gen_words = words_in_text(generated_text)
    src_in = src.lower() in gen_words
    tgt_in = tgt.lower() in gen_words
    return src_in, tgt_in, (src_in and tgt_in)


def build_held_out_pair_index(holdout_sents: list[str]) -> list[tuple[str, str, str]]:
    """For each held-out sentence, recover the (concept, src_word, tgt_word).

    We re-walk the same templates × pairs construction used in
    build_corpus() so the index aligns 1:1 with holdout_sents.
    """
    rows: list[tuple[str, str, str]] = []
    for concept, ddir in CONCEPT_DATA_DIRS.items():
        holdout_pairs = read_pairs(Path(ddir) / "text_pairs_held_out.tsv")
        templates = TEMPLATES[concept]
        for src, tgt in holdout_pairs:
            for _tpl in templates:
                rows.append((concept, src, tgt))
    if len(rows) != len(holdout_sents):
        raise SystemExit(
            f"FATAL: held-out index mismatch: {len(rows)} pair-rows vs "
            f"{len(holdout_sents)} sentences. build_corpus order may have changed."
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # Training
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # Architecture (smaller defaults than 2a.0c since cross-attention is more expressive)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.4)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    # Loss recipe
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--no-mse", action="store_true")
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--perturb-prob", type=float, default=0.5)
    parser.add_argument("--gaussian-delta", type=float, default=0.7)
    parser.add_argument("--mask-token-rate", type=float, default=0.3)
    # Acceptance
    parser.add_argument("--cos-min", type=float, default=0.85)
    parser.add_argument("--grammar-pass-rate", type=float, default=0.95)
    parser.add_argument("--word-fidelity-min", type=float, default=0.70,
                        help="Min fraction of held-out sentences with BOTH "
                             "src + tgt words present (the 2a.0c blind-spot fix).")
    # Output
    parser.add_argument("--out", default="results/stage2a/seq_conditioning.json")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    use_mse = not args.no_mse and args.mse_weight > 0.0
    use_perturb = not args.no_perturb

    enc_cfg = ENCODERS[args.encoder]
    print("Phase 2a sub-task 2a.0d — sequence-conditioning fix + word-fidelity gate")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Loss: CE | MSE: {'on (λ=%.2f)' % args.mse_weight if use_mse else 'OFF'} "
          f"| Perturb: {'on' if use_perturb else 'OFF'} "
          f"| FeatDropout: {args.feat_dropout}")

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]

    # ---- Corpus -------------------------------------------------------
    train_sents, holdout_sents, corpus_stats = build_corpus()
    n_train = len(train_sents)
    n_holdout = len(holdout_sents)
    print(f"\nCorpus: {n_train} train + {n_holdout} held-out (same as 2a.0c)")

    # Build (concept, src, tgt) index for held-out word fidelity scoring.
    holdout_pairs_idx = build_held_out_pair_index(holdout_sents)
    print(f"  held-out pairs index built: {len(holdout_pairs_idx)} rows")

    # ---- Encode train + holdout (full activations for both) ----------
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
    print(f"  train_h: {tuple(train_h.shape)}  holdout_h: {tuple(holdout_h.shape)}")

    # Tokenize for token-CE labels.
    train_tok = tok(
        train_sents, padding="max_length", truncation=True,
        max_length=args.t_max, return_tensors="pt",
    ).to(args.device)
    train_ids = train_tok.input_ids
    holdout_tok = tok(
        holdout_sents, padding="max_length", truncation=True,
        max_length=args.t_max, return_tensors="pt",
    ).to(args.device)
    longest = int(train_tok.attention_mask.sum(dim=-1).max().item())
    print(f"  longest train tokenized length: {longest} (t_max={args.t_max})")

    # ---- Decoder ------------------------------------------------------
    torch.manual_seed(args.seed)
    decoder = SeqCondDecoder(
        encoder_dim=DIM,
        hidden_dim=args.hidden_dim,
        t_max=args.t_max,
        vocab_size=tok.vocab_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_mult=args.ffn_mult,
        feat_dropout=args.feat_dropout,
        attn_dropout=args.attn_dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder (sequence-conditioned): {n_params/1e6:.2f}M params")
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
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad()
        idx = torch.randint(0, n_train, (args.batch_size,), device=args.device)
        h_batch = train_h[idx]                             # [B, T_in, D_enc]
        h_mask_batch = train_h_mask[idx]                   # [B, T_in]
        ids_batch = train_ids[idx]                         # [B, T_out]

        if use_perturb:
            h_input = perturb_h(
                h_batch, h_mask_batch,
                apply_prob=args.perturb_prob,
                gaussian_delta=args.gaussian_delta,
                mask_token_rate=args.mask_token_rate,
            )
        else:
            h_input = h_batch

        logits, hidden_out = decoder(h_input, h_mask_batch)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids_batch.reshape(-1),
        )
        if use_mse:
            # MSE between projected decoder hidden states and the
            # ORIGINAL (unperturbed) encoder activations of the target
            # text. Mask out padding positions in the target so we
            # don't fit pad activations.
            h_target = h_batch                              # [B, T, D_enc]
            mask = h_mask_batch.unsqueeze(-1)               # [B, T, 1]
            h_pred = decoder.mse_proj(hidden_out)           # [B, T, D_enc]
            mse_per = (h_pred - h_target).pow(2)
            mse_loss = (mse_per * mask).sum() / (mask.sum() * DIM).clamp(min=1.0)
            loss = ce_loss + args.mse_weight * mse_loss
        else:
            mse_loss = torch.tensor(0.0, device=args.device)
            loss = ce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            loss_history.append({
                "step": step,
                "loss": float(loss.item()),
                "ce": float(ce_loss.item()),
                "mse": float(mse_loss.item()),
                "lr": float(opt.param_groups[0]["lr"]),
            })
            print(f"  step {step:>5}/{args.steps}  "
                  f"loss={loss.item():.4f}  ce={ce_loss.item():.4f}  "
                  f"mse={mse_loss.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.6f}")

    # ---- Held-out eval -----------------------------------------------
    print("\n" + "=" * 78)
    print("HELD-OUT EVALUATION (with word-fidelity gate)")
    print("=" * 78)
    decoder.eval()
    eval_bs = 32
    all_gen_ids = []
    with torch.no_grad():
        for s in range(0, n_holdout, eval_bs):
            h_chunk = holdout_h[s:s + eval_bs]
            mask_chunk = holdout_h_mask[s:s + eval_bs]
            logits, _ = decoder(h_chunk, mask_chunk)
            all_gen_ids.append(logits.argmax(dim=-1))
    gen_ids = torch.cat(all_gen_ids, dim=0)

    results: list[dict] = []
    n_cos_pass = 0
    n_grammar_pass = 0
    n_exact = 0
    n_src_in = 0
    n_tgt_in = 0
    n_both_in = 0
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
        })

    # Print first 15 for inspection.
    print(f"\nPer-sentence (first 15 of {n_holdout}):")
    for i, r in enumerate(results[:15]):
        cos_mark = "✓" if r["cos_recovered"] >= args.cos_min else "✗"
        gr_mark = "✓" if r["grammar_pass"] else "✗"
        wf_mark = "✓" if r["both_in_gen"] else (
            "≈" if r["src_in_gen"] or r["tgt_in_gen"] else "✗"
        )
        em = " (exact)" if r["exact_match"] else ""
        print(f"  [{i+1}] target:    {r['target']!r}")
        print(f"      generated: {r['generated']!r}{em}")
        print(f"      pair: ({r['src_word']!r}, {r['tgt_word']!r})  "
              f"src_in={r['src_in_gen']}  tgt_in={r['tgt_in_gen']}")
        print(f"      {cos_mark} cos={r['cos_recovered']:.4f}  "
              f"{gr_mark} LT={r['lt_n_errors']}  "
              f"{wf_mark} word-fidelity={'BOTH' if r['both_in_gen'] else ('SOME' if r['src_in_gen'] or r['tgt_in_gen'] else 'NONE')}")

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

    if cos_gate and grammar_gate and word_fid_gate:
        verdict = "GENERALIZATION_PASS"
        message = (
            "All three gates passed: cos, grammar, AND word-pair fidelity. "
            "Sequence conditioning fixed the 2a.0c failure mode. Approach "
            "B with sequence conditioning is empirically viable for the "
            "full Phase 2a build. Next: revise plan §19.12 to lock in "
            "this architecture (replace the single-pooled-ψ decoder with "
            "the SeqCondDecoder defined here) and proceed to the production "
            "Phase 2a build."
        )
    elif cos_gate and grammar_gate and not word_fid_gate:
        verdict = "WORD_FIDELITY_FAIL"
        message = (
            "Same failure mode as 2a.0c: cos and grammar pass but the "
            "decoder still doesn't reproduce the target's specific word "
            "pair on held-out. Sequence conditioning didn't help enough "
            "at this scale. Next: try (a) more training steps, (b) larger "
            "model, (c) iterative refinement loop (vec2text-style: "
            "generate → re-encode → refine), or (d) escalate to §13.9 "
            "templated structural backbone."
        )
    elif cos_gate and not grammar_gate:
        verdict = "GRAMMAR_BELOW_GATE"
        message = (
            "Cos passes but text isn't grammatical. Increase training "
            "steps and/or model capacity."
        )
    elif not cos_gate and (grammar_gate or word_fid_gate):
        verdict = "COS_BELOW_GATE"
        message = (
            "Generated text recovers some target features (grammar or "
            "word-pair) but pooled ψ similarity is below 0.85. Investigate "
            "the conditioning-to-output flow."
        )
    else:
        verdict = "GENERALIZATION_FAIL"
        message = (
            "Multiple gates fail. The architecture or recipe needs deeper "
            "rework. Likely paths: more data, iterative refinement, "
            "or §13.9 fallback."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ---------------------------------------------------
    payload = {
        "task": "2a.0d",
        "encoder": args.encoder,
        "encoder_dim": DIM,
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
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
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
            "n_holdout": n_holdout,
        },
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
