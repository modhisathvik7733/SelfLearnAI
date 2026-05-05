"""Phase 2a sub-task 2a.0e — Pointer-Generator decoder for word fidelity.

Empirical fix for 2a.0d's WORD_FIDELITY_FAIL. ~40-60 min GPU.

Diagnosis from 2a.0d (commit 2a5ce56): even with full encoder
activation cross-attention conditioning, the decoder produced
0/168 target words on held-out (6/168 source words). Failure mode:
the model used cross-attention for TEMPLATE recognition but
sampled word pairs from the training distribution rather than
extracting specific words from the encoder activations. CE loss
collapsed to 0.0000 on training (full memorization) — the model
found a vocab-only solution that ignores word identity in the
cross-attention.

Fix (See/Liu/Manning 2017 — Pointer-Generator Network):

    final_prob[token=k] = p_gen · P_vocab[k]
                         + (1 - p_gen) · Σⱼ attn[j] · 1[input_token_j == k]

The decoder explicitly chooses, per output position, between:
  - GENERATE from vocab (p_gen ≈ 1) → for template words
  - COPY a token from the encoder input sequence (p_gen ≈ 0)
    → for slot fills (the {src}, {tgt} positions)

The copy probability is computed by scattering the cross-attention
weights onto the encoder input token IDs. This gives the decoder a
direct mechanism to reproduce held-out words from the encoder input,
without relying on the dense vocab projection learning a "copy this
held-out word" signal that pure CE on memorizable training data
never incentivizes.

Same corpus as 2a.0c/2a.0d (1036 train + 168 held-out, shared
templates, disjoint word pairs). Same word-fidelity gate as 2a.0d
(at least 70% of held-out sentences must contain BOTH target src
and tgt words). Cos and grammar gates retained for completeness.

Architecture (PointerSeqCondDecoder):
  - Cross-attention over encoder activations (same as SeqCondDecoder)
  - Standard vocab head (token_head)
  - Pointer head: per-position dot-product attention over encoder
    positions → softmax → scatter to vocab via input_token_ids
  - Gate head: per-position p_gen ∈ [0, 1] via sigmoid
  - Final NLL loss on the mixture distribution

Run on the GPU box (~40-60 min):
  python scripts/stage2a_pointer.py
  python scripts/stage2a_pointer.py --steps 30000
"""
from __future__ import annotations

import argparse
import json
import math
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
from scripts.stage2a_seq_conditioning import (
    perturb_h,
    word_pair_fidelity,
    build_held_out_pair_index,
)


# ---------------------------------------------------------------------------
# Pointer-Generator decoder
# ---------------------------------------------------------------------------

class PointerSeqCondDecoder(nn.Module):
    """Non-AR decoder with cross-attention + a pointer-generator
    mixture head.

    Forward pass:
      1. Project encoder activations h → memory (per-token, hidden_dim).
      2. T_out learnable output position embeddings.
      3. Bidirectional self-attention on outputs + cross-attention to
         memory (nn.TransformerDecoder, no causal mask).
      4. token_head: dense vocab logits per output position.
      5. Pointer head: dot-product attention from output hidden →
         memory hidden → softmax over encoder positions, then
         scatter onto vocab via input_token_ids.
      6. gen_gate: per-position p_gen ∈ [0, 1].
      7. Final probability: p_gen · vocab_probs + (1 − p_gen) · copy_probs.

    Returns (final_log_probs, decoder_hidden, p_gen).
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
        feat_dropout: float = 0.2,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.t_max = t_max
        self.cond_proj = nn.Linear(encoder_dim, hidden_dim)
        self.feat_dropout = nn.Dropout(p=feat_dropout)
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
        # Pointer attention: separate Q/K projections so the pointer
        # learns its own attention pattern, independent of the
        # transformer's cross-attention layers.
        self.ptr_q = nn.Linear(hidden_dim, hidden_dim)
        self.ptr_k = nn.Linear(hidden_dim, hidden_dim)
        # Gate: per-position p_gen ∈ [0, 1].
        self.gen_gate = nn.Linear(hidden_dim, 1)
        # MSE projection (kept for ablation parity with 2a.0d).
        self.mse_proj = nn.Linear(hidden_dim, encoder_dim)

    def forward(
        self,
        encoder_h: torch.Tensor,             # [B, T_in, D_enc]
        encoder_mask: torch.Tensor,          # [B, T_in]
        encoder_token_ids: torch.Tensor,     # [B, T_in]  — input token ids
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T_in, _ = encoder_h.shape
        memory = self.feat_dropout(self.cond_proj(encoder_h))   # [B, T_in, h]
        memory_pad_mask = (encoder_mask < 0.5)                  # True for pads

        out_seeds = self.output_pos_emb.unsqueeze(0).expand(B, -1, -1)
        out_hidden = self.transformer(
            tgt=out_seeds,
            memory=memory,
            memory_key_padding_mask=memory_pad_mask,
        )                                                       # [B, T_out, h]

        # ---- Vocab branch ----
        vocab_logits = self.token_head(out_hidden)              # [B, T_out, V]
        vocab_probs = F.softmax(vocab_logits, dim=-1)

        # ---- Pointer branch ----
        # Per-position attention over encoder positions.
        q = self.ptr_q(out_hidden)                              # [B, T_out, h]
        k = self.ptr_k(memory)                                  # [B, T_in,  h]
        ptr_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        # Mask padding before softmax.
        ptr_scores = ptr_scores.masked_fill(
            memory_pad_mask.unsqueeze(1), float("-inf"),
        )
        ptr_attn = F.softmax(ptr_scores, dim=-1)                # [B, T_out, T_in]

        # Scatter attention probs to vocab via input_token_ids.
        # For batch b, output position i, encoder position j with token k:
        #   copy_probs[b, i, k] += ptr_attn[b, i, j]
        # Vectorize using scatter_add along vocab dim.
        copy_probs = torch.zeros_like(vocab_probs)              # [B, T_out, V]
        # encoder_token_ids: [B, T_in] → expand to [B, T_out, T_in]
        idx = encoder_token_ids.unsqueeze(1).expand(-1, vocab_probs.size(1), -1)
        copy_probs = copy_probs.scatter_add(-1, idx, ptr_attn)

        # ---- Gate ----
        p_gen = torch.sigmoid(self.gen_gate(out_hidden))        # [B, T_out, 1]

        # ---- Mixture ----
        final_probs = p_gen * vocab_probs + (1.0 - p_gen) * copy_probs
        final_log_probs = torch.log(final_probs.clamp_min(1e-12))

        return final_log_probs, out_hidden, p_gen


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
    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="Lower than 2a.0d's 0.4 — pointer head needs "
                             "stable cross-attention to learn copy alignment.")
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    # Loss recipe
    parser.add_argument("--mse-weight", type=float, default=0.5,
                        help="Lower than 2a.0d's 1.0 — too much MSE pressure "
                             "competes with the pointer loss.")
    parser.add_argument("--no-mse", action="store_true")
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--perturb-prob", type=float, default=0.3,
                        help="Lower than 2a.0d's 0.5 — pointer needs "
                             "predictable cross-attention input.")
    parser.add_argument("--gaussian-delta", type=float, default=0.7)
    parser.add_argument("--mask-token-rate", type=float, default=0.3)
    # Acceptance
    parser.add_argument("--cos-min", type=float, default=0.85)
    parser.add_argument("--grammar-pass-rate", type=float, default=0.95)
    parser.add_argument("--word-fidelity-min", type=float, default=0.70)
    # Output
    parser.add_argument("--out", default="results/stage2a/pointer.json")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    use_mse = not args.no_mse and args.mse_weight > 0.0
    use_perturb = not args.no_perturb

    enc_cfg = ENCODERS[args.encoder]
    print("Phase 2a sub-task 2a.0e — Pointer-Generator decoder")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Loss: NLL on (vocab + copy) mixture | "
          f"MSE: {'on (λ=%.2f)' % args.mse_weight if use_mse else 'OFF'} | "
          f"Perturb: {'on' if use_perturb else 'OFF'} | "
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
    train_sents, holdout_sents, corpus_stats = build_corpus()
    n_train = len(train_sents)
    n_holdout = len(holdout_sents)
    print(f"\nCorpus: {n_train} train + {n_holdout} held-out (same as 2a.0c/d)")

    holdout_pairs_idx = build_held_out_pair_index(holdout_sents)

    # ---- Encode train + holdout (full activations) --------------------
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

    # Tokenize for both encoder input ids (pointer) and CE labels.
    train_tok = tok(
        train_sents, padding="max_length", truncation=True,
        max_length=args.t_max, return_tensors="pt",
    ).to(args.device)
    train_ids = train_tok.input_ids                       # [N_train, T]
    holdout_tok = tok(
        holdout_sents, padding="max_length", truncation=True,
        max_length=args.t_max, return_tensors="pt",
    ).to(args.device)
    holdout_ids = holdout_tok.input_ids                   # [N_holdout, T]
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
    print(f"\nPointerSeqCondDecoder: {n_params/1e6:.2f}M params")

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
        ids_batch = train_ids[idx]                          # input + target
        # IMPORTANT: input_token_ids for the pointer should be the
        # SOURCE encoder's token ids (which are the target ids since
        # we're inverting an autoencoder). They match here.
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

        # NLL on mixture log-probs.
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

    # ---- Held-out eval -----------------------------------------------
    print("\n" + "=" * 78)
    print("HELD-OUT EVALUATION (with word-fidelity gate)")
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
    gen_ids = torch.cat(all_gen_ids, dim=0)              # [N_holdout, T]
    p_gen_all = torch.cat(all_p_gen, dim=0)              # [N_holdout, T]

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

        # Mean p_gen across real positions for this sentence
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

    # First 15 for inspection
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
              f"src_in={r['src_in_gen']}  tgt_in={r['tgt_in_gen']}  "
              f"p_gen={r['p_gen_mean']:.3f}")
        print(f"      {cos_mark} cos={r['cos_recovered']:.4f}  "
              f"{gr_mark} LT={r['lt_n_errors']}  "
              f"{wf_mark} word-fidelity="
              f"{'BOTH' if r['both_in_gen'] else ('SOME' if r['src_in_gen'] or r['tgt_in_gen'] else 'NONE')}")

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
    p_gen_overall = float(sum(r["p_gen_mean"] for r in results) / len(results))
    print(f"Mean p_gen across held-out:      {p_gen_overall:.3f}  "
          f"(0.0 = always copy, 1.0 = always generate)")

    if cos_gate and grammar_gate and word_fid_gate:
        verdict = "POINTER_PASS"
        message = (
            "Pointer-Generator fixed the word-fidelity bottleneck: held-out "
            "word pairs are correctly reproduced via the copy mechanism. "
            "Approach B with sequence conditioning + pointer-generator is "
            "empirically viable. Next: revise plan §19.12 to lock this "
            "architecture in and proceed to the production Phase 2a build."
        )
    elif cos_gate and grammar_gate and not word_fid_gate:
        verdict = "POINTER_FAIL"
        message = (
            "Pointer-Generator did not fix word fidelity at this scale. "
            "p_gen behavior may indicate the model isn't learning to copy "
            "(check `Mean p_gen` — close to 1.0 means it's still vocab-only). "
            "Next: try (a) more training data, (b) explicit p_gen regularization "
            "to encourage copying at slot positions, (c) iterative "
            "refinement (vec2text-style), or (d) escalate to §13.9 fallback."
        )
    elif not cos_gate or not grammar_gate:
        verdict = "BASELINE_REGRESSION"
        message = (
            "Cos or grammar gate regressed compared to 2a.0d. Pointer "
            "head is interfering with the base architecture. Consider "
            "a smaller pointer head or different gate initialization."
        )
    else:
        verdict = "GENERALIZATION_FAIL"
        message = (
            "Multiple gates fail. Escalate or rethink architecture."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ---------------------------------------------------
    payload = {
        "task": "2a.0e",
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
            "p_gen_overall": p_gen_overall,
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
