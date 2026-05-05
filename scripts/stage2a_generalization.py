"""Phase 2a sub-task 2a.0c — Cosmos-style generalization test.

Decisive test of the paper-informed Approach B at scale. ~1–2 hr GPU.

Question: does the architecture that PASSED capacity (2a.0b on 50
sentences) GENERALIZE to held-out word pairs under the full paper
recipe? Pass → revise §19.12 with the paper's recipe + proceed to
Phase 2a's main sub-tasks. Fail → either iterate (more data,
diffusion, etc.) or escalate per plan §13.9.

Setup:
  - Corpus: 4 strong concepts (plural, past_tense, comparative,
    opposite) × 7 templates × ~37 train word pairs ≈ 1036 train
    sentences + 168 held-out sentences. Held-out word pairs do
    NOT appear in train; templates are SHARED (per Plan-agent
    guidance: stratified template family).
  - Decoder: same architecture as 2a.0b, scaled to ~25M params
    (4 layers, 768 hidden, 8 heads, N=8 cond tokens, T=32).
  - Loss: parallel position CE + λ_MSE · MSE on encoder activations
    + perturbation augmentation on ψ + feature dropout on
    conditioning tokens.
  - Training: ~20K steps, batch 32, AdamW lr=2e-4 + warmup,
    grad clip 1.0.

Acceptance gates (HARD):
  - Median cos(encode(generated), ψ_target) ≥ 0.85 on 168 held-out.
  - ≥ 95% sentences pass grammar gate (LanguageTool errors == 0,
    falls back to wordfreq proxy with caveat if Java not installed).

Run:
  python scripts/stage2a_generalization.py
  python scripts/stage2a_generalization.py --steps 30000 --no-mse
  python scripts/stage2a_generalization.py --no-perturb --no-feat-dropout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn, read_pairs
from scripts.stage2a_quick_probe import grammar_grade, grammar_proxy


# ---------------------------------------------------------------------------
# Corpus templates (extended from 2a.0b)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, list[str]] = {
    "plural": [
        "the plural of {src} is {tgt}",
        "{src} becomes {tgt} in plural form",
        "{tgt} are the plural of {src}",
        "we say {tgt} when there are many {src}",
        "{tgt} is what we call multiple {src}",
        "to make {src} plural we say {tgt}",
        "more than one {src} becomes {tgt}",
    ],
    "past_tense": [
        "the past tense of {src} is {tgt}",
        "{src} becomes {tgt} in the past",
        "{tgt} is the past tense of {src}",
        "to say {src} happened we use {tgt}",
        "{tgt} is what {src} becomes in the past",
        "yesterday i {tgt} after i {src}",
        "the past form of {src} is {tgt}",
    ],
    "comparative": [
        "the comparative of {src} is {tgt}",
        "more {src} is {tgt}",
        "something more {src} is {tgt}",
        "to compare we say {tgt} instead of {src}",
        "{tgt} is the comparative form of {src}",
        "when something is more {src} it is {tgt}",
        "the comparative form of {src} is {tgt}",
    ],
    "opposite": [
        "the opposite of {src} is {tgt}",
        "{src} and {tgt} are opposites",
        "{tgt} is the opposite of {src}",
        "{tgt} is the antonym of {src}",
        "the antonym of {src} is {tgt}",
        "{src} means the opposite of {tgt}",
        "if something is not {src} it might be {tgt}",
    ],
}

CONCEPT_DATA_DIRS = {
    "plural":      "data/plurality",
    "past_tense":  "data/past_tense",
    "comparative": "data/comparative",
    "opposite":    "data/opposite_v2",
}


def build_corpus() -> tuple[list[str], list[str], dict]:
    """Build train + held-out corpora using SHARED templates and DISJOINT
    word-pair sets (held-out pairs come from text_pairs_held_out.tsv;
    train pairs come from text_pairs_train.tsv).

    Returns (train_sents, holdout_sents, stats).
    """
    train_sents: list[str] = []
    holdout_sents: list[str] = []
    stats: dict = {"per_concept": {}}
    for concept, ddir in CONCEPT_DATA_DIRS.items():
        train_pairs = read_pairs(Path(ddir) / "text_pairs_train.tsv")
        holdout_pairs = read_pairs(Path(ddir) / "text_pairs_held_out.tsv")
        # Sanity: no overlap of (src, tgt) pairs across the two splits.
        overlap = set(train_pairs) & set(holdout_pairs)
        if overlap:
            raise SystemExit(
                f"FATAL: {concept} train/holdout overlap on pairs: {overlap}"
            )
        templates = TEMPLATES[concept]
        for src, tgt in train_pairs:
            for tpl in templates:
                train_sents.append(tpl.format(src=src, tgt=tgt))
        for src, tgt in holdout_pairs:
            for tpl in templates:
                holdout_sents.append(tpl.format(src=src, tgt=tgt))
        stats["per_concept"][concept] = {
            "n_train_pairs": len(train_pairs),
            "n_holdout_pairs": len(holdout_pairs),
            "n_templates": len(templates),
            "n_train_sents": len(train_pairs) * len(templates),
            "n_holdout_sents": len(holdout_pairs) * len(templates),
        }
    stats["total_train_sents"] = len(train_sents)
    stats["total_holdout_sents"] = len(holdout_sents)
    return train_sents, holdout_sents, stats


# ---------------------------------------------------------------------------
# Decoder (same shape as 2a.0b, scaled, exposes hidden states for MSE loss)
# ---------------------------------------------------------------------------

class NonARDecoder(nn.Module):
    """Non-autoregressive transformer decoder with single-ψ conditioning.

    Forward pass:
      1. ψ ∈ R^{B×D_enc} → R^{B×N×h} via learned projection.
      2. T learnable output position embeddings.
      3. Concat [cond_tokens, output_pos_emb] = N+T.
      4. Optional feature dropout on conditioning tokens (paper §4.2.2).
      5. Add global positional embeddings.
      6. L bidirectional transformer encoder layers.
      7. Slice off output positions → linear head → vocab logits.

    Returns (logits, decoder_hidden_at_output_positions). The hidden
    states are projected to encoder dim by the caller for the MSE loss
    against E5's last_hidden_state on ground truth.
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        hidden_dim: int = 768,
        n_cond: int = 8,
        t_max: int = 32,
        vocab_size: int = 30522,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_mult: int = 4,
        feat_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.n_cond = n_cond
        self.t_max = t_max
        self.vocab_size = vocab_size
        self.psi_proj = nn.Linear(encoder_dim, n_cond * hidden_dim)
        # Feature-level dropout on conditioning tokens (paper p=0.4).
        self.feat_dropout = nn.Dropout(p=feat_dropout)
        self.output_pos_emb = nn.Parameter(
            torch.randn(t_max, hidden_dim) * 0.02
        )
        self.pos_emb = nn.Parameter(
            torch.randn(n_cond + t_max, hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            batch_first=True, activation="gelu",
            dropout=0.0, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.token_head = nn.Linear(hidden_dim, vocab_size)
        # MSE-loss projection: decoder hidden_dim → encoder_dim, used to
        # match E5's last_hidden_state for the activation-MSE loss.
        self.mse_proj = nn.Linear(hidden_dim, encoder_dim)

    def forward(
        self, psi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """psi: [B, encoder_dim] → (logits [B, T, V], hidden [B, T, h])"""
        B = psi.size(0)
        cond = self.psi_proj(psi).view(B, self.n_cond, self.hidden_dim)
        cond = self.feat_dropout(cond)                  # paper §4.2.2
        out_seeds = self.output_pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cond, out_seeds], dim=1)
        x = x + self.pos_emb.unsqueeze(0)
        x = self.transformer(x)
        out_hidden = x[:, self.n_cond:, :]              # [B, T, h]
        logits = self.token_head(out_hidden)            # [B, T, V]
        return logits, out_hidden


# ---------------------------------------------------------------------------
# Perturbation augmentation (paper §4.2.2)
# ---------------------------------------------------------------------------

def perturb_psi(
    psi: torch.Tensor,                 # [B, D]
    *,
    apply_prob: float = 0.5,
    gaussian_delta: float = 0.7,
    mask_feature_rate: float = 0.3,
) -> torch.Tensor:
    """Two-mode perturbation, applied with prob `apply_prob` per sample,
    50/50 between modes when applied:

      MODE A (Gaussian noise):  ψ' = δ·ψ + sqrt(1−δ²)·ε,  ε ~ N(0, I)
      MODE B (feature masking): zero out `mask_feature_rate` fraction
                                of ψ's features (BERT-vector analog of
                                the paper's 30%-of-vectors masking,
                                adapted because we have one pooled
                                vector instead of a sequence).
    """
    B = psi.size(0)
    device = psi.device
    apply_mask = torch.rand(B, device=device) < apply_prob
    if not apply_mask.any():
        return psi
    mode_a = torch.rand(B, device=device) < 0.5
    out = psi.clone()
    # Mode A: Gaussian noise
    a_idx = apply_mask & mode_a
    if a_idx.any():
        noise = torch.randn_like(out[a_idx])
        out[a_idx] = (
            gaussian_delta * out[a_idx]
            + (1.0 - gaussian_delta ** 2) ** 0.5 * noise
        )
    # Mode B: feature masking
    b_idx = apply_mask & ~mode_a
    if b_idx.any():
        feat_mask = (torch.rand_like(out[b_idx]) >= mask_feature_rate).float()
        out[b_idx] = out[b_idx] * feat_mask
    return out


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_pooled(
    sents: list[str], tok, mdl, device: str, max_length: int = 64,
) -> torch.Tensor:
    """Standard mean-pooled encode (for ψ targets)."""
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
    """Return (last_hidden_state, attention_mask) for the MSE loss.

    Pads/truncates to t_max so shapes line up with the decoder's output.
    """
    inputs = tok(
        sents, padding="max_length", truncation=True, max_length=t_max,
        return_tensors="pt",
    ).to(device)
    out = mdl(**inputs).last_hidden_state            # [B, T, D_enc]
    return out, inputs.attention_mask.float()        # [B, T]


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
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--n-cond", type=int, default=8)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    # Loss recipe (paper §4.2.2)
    parser.add_argument("--mse-weight", type=float, default=1.0,
                        help="λ for activation-MSE loss. 0 disables.")
    parser.add_argument("--no-mse", action="store_true",
                        help="Disable MSE loss entirely.")
    parser.add_argument("--no-perturb", action="store_true",
                        help="Disable ψ perturbation augmentation.")
    parser.add_argument("--perturb-prob", type=float, default=0.5)
    parser.add_argument("--gaussian-delta", type=float, default=0.7)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--feat-dropout", type=float, default=0.4)
    # Acceptance
    parser.add_argument("--cos-min", type=float, default=0.85)
    parser.add_argument("--grammar-pass-rate", type=float, default=0.95)
    # Output
    parser.add_argument("--out", default="results/stage2a/generalization.json")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    use_mse = not args.no_mse and args.mse_weight > 0.0
    use_perturb = not args.no_perturb

    enc_cfg = ENCODERS[args.encoder]
    print("Phase 2a sub-task 2a.0c — generalization test (full paper recipe)")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Loss recipe: CE always | MSE: {'on (λ=%.2f)' % args.mse_weight if use_mse else 'OFF'} | "
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
    print(f"\nCorpus:")
    print(f"  train:    {n_train} sentences")
    print(f"  holdout:  {n_holdout} sentences (held-out word pairs, shared templates)")
    for concept, st in corpus_stats["per_concept"].items():
        print(f"    {concept:<12}  train={st['n_train_sents']}  "
              f"holdout={st['n_holdout_sents']}  templates={st['n_templates']}")
    print(f"  samples (train):")
    for s in train_sents[:3]:
        print(f"    {s!r}")
    print(f"  samples (holdout):")
    for s in holdout_sents[:3]:
        print(f"    {s!r}")

    # Tokenize once
    print("\nTokenizing + encoding train and holdout ...")
    train_tok = tok(
        train_sents, padding="max_length", truncation=True,
        max_length=args.t_max, return_tensors="pt",
    ).to(args.device)
    holdout_tok = tok(
        holdout_sents, padding="max_length", truncation=True,
        max_length=args.t_max, return_tensors="pt",
    ).to(args.device)
    train_ids = train_tok.input_ids                           # [N_train, T]
    holdout_ids = holdout_tok.input_ids
    train_lengths = train_tok.attention_mask.sum(dim=-1)
    longest_train = int(train_lengths.max().item())
    print(f"  longest train tokenized length: {longest_train} (t_max={args.t_max})")
    if longest_train >= args.t_max:
        print(f"  WARNING: at least one train sentence reached t_max — increase --t-max")

    # Encode pooled ψ for both splits, plus full activations for train
    # (only train activations are needed for MSE; holdout uses pooled ψ
    # for cos eval).
    train_psi = encode_pooled(train_sents, tok, mdl, args.device, max_length=args.t_max)
    holdout_psi = encode_pooled(holdout_sents, tok, mdl, args.device, max_length=args.t_max)
    if use_mse:
        train_h, train_h_mask = encode_activations(
            train_sents, tok, mdl, args.device, t_max=args.t_max,
        )
        print(f"  train pooled ψ: {tuple(train_psi.shape)}, "
              f"activations: {tuple(train_h.shape)}")
    else:
        train_h = None
        train_h_mask = None
        print(f"  train pooled ψ: {tuple(train_psi.shape)}  (MSE disabled)")

    # ---- Decoder ------------------------------------------------------
    torch.manual_seed(args.seed)
    decoder = NonARDecoder(
        encoder_dim=DIM, hidden_dim=args.hidden_dim,
        n_cond=args.n_cond, t_max=args.t_max,
        vocab_size=tok.vocab_size,
        n_layers=args.n_layers, n_heads=args.n_heads,
        ffn_mult=args.ffn_mult, feat_dropout=args.feat_dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder: {n_params/1e6:.2f}M params")
    print(f"  layers={args.n_layers}  hidden={args.hidden_dim}  "
          f"heads={args.n_heads}  N_cond={args.n_cond}  T_max={args.t_max}")

    # ---- Optimizer + LR schedule -------------------------------------
    opt = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        return args.lr      # constant after warmup (paper does this too)

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
        psi_batch = train_psi[idx]
        ids_batch = train_ids[idx]

        # Perturbation augmentation BEFORE psi enters the decoder.
        if use_perturb:
            psi_input = perturb_psi(
                psi_batch,
                apply_prob=args.perturb_prob,
                gaussian_delta=args.gaussian_delta,
                mask_feature_rate=args.mask_rate,
            )
        else:
            psi_input = psi_batch

        logits, hidden_out = decoder(psi_input)         # [B, T, V], [B, T, h]
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids_batch.reshape(-1),
        )
        if use_mse:
            h_target = train_h[idx]                       # [B, T, D_enc]
            h_mask = train_h_mask[idx].unsqueeze(-1)      # [B, T, 1]
            h_pred = decoder.mse_proj(hidden_out)         # [B, T, D_enc]
            mse_per = (h_pred - h_target).pow(2)          # [B, T, D_enc]
            # Mask out pad positions in the MSE target (we don't want
            # to fit pad activations).
            mse_loss = (mse_per * h_mask).sum() / (h_mask.sum() * DIM).clamp(min=1.0)
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
                  f"loss={loss.item():.4f}  "
                  f"ce={ce_loss.item():.4f}  "
                  f"mse={mse_loss.item():.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.6f}")

    # ---- Held-out eval -----------------------------------------------
    print("\n" + "=" * 78)
    print("HELD-OUT EVALUATION")
    print("=" * 78)
    decoder.eval()
    eval_batch_size = 64
    all_gen_ids = []
    with torch.no_grad():
        for s in range(0, n_holdout, eval_batch_size):
            psi_chunk = holdout_psi[s:s + eval_batch_size]
            logits, _ = decoder(psi_chunk)
            all_gen_ids.append(logits.argmax(dim=-1))
    gen_ids = torch.cat(all_gen_ids, dim=0)        # [N_holdout, T]

    results: list[dict] = []
    n_cos_pass = 0
    n_grammar_pass = 0
    n_exact = 0
    for i in range(n_holdout):
        target_text = holdout_sents[i]
        gen_text = tok.decode(gen_ids[i].tolist(), skip_special_tokens=True)
        psi_gen = encode_pooled([gen_text], tok, mdl, args.device,
                                max_length=args.t_max)[0]
        cos_recovered = float(F.cosine_similarity(
            psi_gen, holdout_psi[i], dim=0,
        ).item())
        n_errors, grammar_pass = grammar_grade(gen_text)
        proxy = grammar_proxy(gen_text)
        exact = gen_text.strip() == target_text.strip()

        if cos_recovered >= args.cos_min:
            n_cos_pass += 1
        if grammar_pass:
            n_grammar_pass += 1
        if exact:
            n_exact += 1

        results.append({
            "target": target_text,
            "generated": gen_text,
            "cos_recovered": cos_recovered,
            "lt_n_errors": n_errors,
            "grammar_pass": grammar_pass,
            "grammar_proxy": proxy,
            "exact_match": exact,
        })

    # Print first 15 for inspection
    print(f"\nPer-sentence (first 15 of {n_holdout}):")
    for i, r in enumerate(results[:15]):
        cos_mark = "✓" if r["cos_recovered"] >= args.cos_min else "✗"
        gr_mark = "✓" if r["grammar_pass"] else "✗"
        em = " (exact)" if r["exact_match"] else ""
        print(f"  [{i+1}] target:    {r['target']!r}")
        print(f"      generated: {r['generated']!r}{em}")
        print(f"      {cos_mark} cos={r['cos_recovered']:.4f}  "
              f"{gr_mark} LT={r['lt_n_errors']}  proxy={r['grammar_proxy']:.3f}")

    # ---- Verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    coses = sorted(r["cos_recovered"] for r in results)
    median_cos = coses[len(coses) // 2]
    grammar_target = int(args.grammar_pass_rate * n_holdout)
    cos_gate = median_cos >= args.cos_min
    grammar_gate = n_grammar_pass >= grammar_target

    print(f"Median cos (held-out):           {median_cos:.4f}  "
          f"(target ≥ {args.cos_min})")
    print(f"Sentences passing cos:           {n_cos_pass}/{n_holdout}")
    print(f"Sentences passing grammar gate:  {n_grammar_pass}/{n_holdout}  "
          f"(target ≥ {grammar_target})")
    print(f"Exact target matches:            {n_exact}/{n_holdout}")

    if cos_gate and grammar_gate:
        verdict = "GENERALIZATION_PASS"
        message = (
            "The architecture generalizes to held-out word pairs under "
            "the paper's full recipe. Phase 2a Approach B is empirically "
            "viable. Next: revise plan §19.12 to reflect this recipe and "
            "begin the production Phase 2a build (corpus generator with "
            "more domains, full-scale training, multi-candidate sampling)."
        )
    elif cos_gate and not grammar_gate:
        verdict = "GRAMMAR_BELOW_GATE"
        message = (
            "Held-out cos passes but grammaticality is below 95%. The "
            "model recovers ψ-fidelity but text is structurally off. "
            "Investigate which templates / concepts fail. Try (a) more "
            "training steps, (b) larger model, (c) larger MSE weight, "
            "(d) introduce a token-bigram prior (templated grammar "
            "doesn't need diffusion-quality fluency for v1)."
        )
    elif not cos_gate and grammar_gate:
        verdict = "COS_BELOW_GATE"
        message = (
            "Text is fluent but doesn't recover ψ on held-out. The "
            "decoder generalizes its template patterns but loses faithfulness "
            "to specific word pairs. Check: are held-out word pairs in "
            "encoder vocabulary? Does the conditioning bottleneck need "
            "more capacity (try N_cond=16, hidden_dim=1024)?"
        )
    else:
        verdict = "GENERALIZATION_FAIL"
        message = (
            "Held-out fails on both metrics. Two follow-ups: (a) increase "
            "training data and/or model capacity; (b) escalate to plan "
            "§13.9 fallback (templated structural backbone). The data "
            "we shipped to the decoder may be too narrow for "
            "generalization at this scale."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ---------------------------------------------------
    payload = {
        "task": "2a.0c",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "corpus": corpus_stats,
        "decoder": {
            "n_params": n_params,
            "hidden_dim": args.hidden_dim,
            "n_cond": args.n_cond,
            "t_max": args.t_max,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "ffn_mult": args.ffn_mult,
            "feat_dropout": args.feat_dropout,
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
            "gaussian_delta": args.gaussian_delta,
            "mask_rate": args.mask_rate,
        },
        "thresholds": {
            "cos_min": args.cos_min,
            "grammar_pass_rate": args.grammar_pass_rate,
            "grammar_pass_target_count": grammar_target,
        },
        "loss_history": loss_history,
        "summary": {
            "median_cos": median_cos,
            "n_cos_pass": n_cos_pass,
            "n_grammar_pass": n_grammar_pass,
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
    raise SystemExit(0 if (cos_gate and grammar_gate) else 1)


if __name__ == "__main__":
    main()
