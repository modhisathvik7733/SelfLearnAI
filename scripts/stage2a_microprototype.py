"""Phase 2a sub-task 2a.0b — Cosmos-style micro-prototype (capacity test).

Smallest, fastest test of the paper-informed Approach B architecture
(see plan §19.13 + arXiv:2506.21170v1). ~15-20 min on GPU.

Question: can a tiny non-AR transformer decoder, conditioned on a
single E5-pooled ψ vector, MEMORIZE a 50-sentence training corpus
under parallel position-wise CE? Generalization is a follow-up
(2a.0c); this test only proves the architecture has the capacity
to recover sentences from a single 1024-dim conditioning vector.

If 2a.0b passes (median cos ≥ 0.95 on training + ≥80% grammatical),
we proceed to 2a.0c with the full paper recipe (CE + activation MSE
+ perturbation augmentation + feature dropout). If it fails on
capacity (cos low), single-ψ conditioning is too lossy and we either
(i) escalate to richer conditioning (Stage 1 chain ψ-states) or
(ii) fall back to §13.9 templated structural backbone. If it passes
cos but fails grammar, we add the paper's full augmentation recipe.

Architecture (single-ψ adaptation of the Cosmos recipe):

    ψ ∈ ℝ^1024  ──→  learned projection  ──→  N conditioning tokens (∈ ℝ^{N×h})
                                                       │
                                                       ▼
                              [cond_tokens, T output position embeddings]
                                                       │
                                              + global positional emb
                                                       │
                                                       ▼
                                  L bidirectional transformer encoder layers
                                                       │
                                                       ▼
                          take output positions  →  linear → vocab logits
                                                       │
                                                       ▼
                                  parallel position-wise cross-entropy

Differences from the paper (Cosmos):
  - Paper takes a SEQUENCE of encoder activations (h ∈ R^{L×768}) into
    a Perceiver Resampler. We have a SINGLE pooled ψ ∈ R^1024, so the
    Perceiver Resampler is replaced with a learned projection to N
    conditioning tokens.
  - Paper trains Gaussian diffusion in the compressed latent space.
    We skip diffusion for v1 — direct supervised training on the
    paired (ψ, text) data is enough to test capacity.
  - Paper uses CE + MSE on encoder activations + perturbation
    augmentation. v1 uses CE only. If v1 passes capacity but fails
    grammar, we add MSE + perturbations (planned but deferred).

Run on the GPU box (~15-20 min):
  python scripts/stage2a_microprototype.py
  python scripts/stage2a_microprototype.py --steps 10000 --hidden-dim 384
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

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn
from scripts.stage2a_quick_probe import grammar_grade, grammar_proxy


# ---------------------------------------------------------------------------
# Tiny templated training corpus (50 sentences)
# ---------------------------------------------------------------------------
# 5 concepts × 5 word pairs × 2 templates per concept = 50 sentences.
# Templates kept simple and grammatically correct so that "the architecture
# generated grammatical text" doesn't depend on the corpus being grammatical
# (i.e. the corpus is the ceiling, not the floor).

TEMPLATES = {
    "plural": [
        "the plural of {src} is {tgt}",
        "{src} becomes {tgt} in plural form",
    ],
    "past_tense": [
        "the past tense of {src} is {tgt}",
        "{src} becomes {tgt} in the past",
    ],
    "comparative": [
        "the comparative of {src} is {tgt}",
        "more than {src} is {tgt}",
    ],
    "opposite": [
        "the opposite of {src} is {tgt}",
        "{src} and {tgt} are opposites",
    ],
    "agentive": [
        "a person who {src} is a {tgt}",
        "someone who {src} is called a {tgt}",
    ],
}

WORD_PAIRS = {
    "plural":      [("cat", "cats"), ("dog", "dogs"), ("car", "cars"),
                    ("hat", "hats"), ("pen", "pens")],
    "past_tense":  [("walk", "walked"), ("jump", "jumped"), ("play", "played"),
                    ("look", "looked"), ("talk", "talked")],
    "comparative": [("big", "bigger"), ("tall", "taller"), ("small", "smaller"),
                    ("short", "shorter"), ("fast", "faster")],
    "opposite":    [("hot", "cold"), ("up", "down"), ("light", "dark"),
                    ("good", "bad"), ("loud", "quiet")],
    "agentive":    [("paints", "painter"), ("writes", "writer"), ("teaches", "teacher"),
                    ("builds", "builder"), ("dances", "dancer")],
}


def build_corpus() -> list[str]:
    sents: list[str] = []
    for concept, pairs in WORD_PAIRS.items():
        for src, tgt in pairs:
            for tpl in TEMPLATES[concept]:
                sents.append(tpl.format(src=src, tgt=tgt))
    return sents


# ---------------------------------------------------------------------------
# Decoder architecture
# ---------------------------------------------------------------------------

class TinyNonARDecoder(nn.Module):
    """Tiny non-autoregressive transformer decoder conditioned on a single ψ.

    Forward pass:
      1. Project ψ ∈ R^{B×D_enc} → R^{B×N×h} via a single Linear layer.
         These N conditioning tokens carry the meaning information.
      2. T learnable output-position embeddings, expanded to batch.
      3. Concat [conditioning tokens (N), output positions (T)] = N+T.
      4. Add global positional embeddings.
      5. L bidirectional transformer encoder layers (standard self-attention,
         no causal mask). Both groups of tokens attend to all others.
      6. Slice off the OUTPUT positions, project each through a linear head
         to vocab_size logits.
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        hidden_dim: int = 256,
        n_cond: int = 4,
        t_max: int = 16,
        vocab_size: int = 30522,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.n_cond = n_cond
        self.t_max = t_max
        self.vocab_size = vocab_size

        # ψ → N conditioning tokens (learned projection)
        self.psi_proj = nn.Linear(encoder_dim, n_cond * hidden_dim)
        # T learnable output position seeds
        self.output_pos_emb = nn.Parameter(torch.randn(t_max, hidden_dim) * 0.02)
        # Global positional embedding for the whole [cond; output] sequence
        self.pos_emb = nn.Parameter(
            torch.randn(n_cond + t_max, hidden_dim) * 0.02
        )
        # Bidirectional transformer (no causal mask — non-AR).
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            batch_first=True, activation="gelu",
            dropout=dropout, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        # Token output head
        self.token_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """psi: [B, encoder_dim] → logits: [B, T, vocab_size]"""
        B = psi.size(0)
        # 1. Conditioning tokens
        cond = self.psi_proj(psi).view(B, self.n_cond, self.hidden_dim)
        # 2. Output position seeds, expanded to batch
        out_seeds = self.output_pos_emb.unsqueeze(0).expand(B, -1, -1)
        # 3. Concat
        x = torch.cat([cond, out_seeds], dim=1)              # [B, N+T, h]
        # 4. Global positional embedding
        x = x + self.pos_emb.unsqueeze(0)
        # 5. Bidirectional transformer
        x = self.transformer(x)
        # 6. Slice output positions, project to vocab
        out = x[:, self.n_cond:, :]                          # [B, T, h]
        return self.token_head(out)                          # [B, T, V]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-cond", type=int, default=4)
    parser.add_argument("--t-max", type=int, default=16,
                        help="Max output sequence length (must be ≥ tokenized longest sentence).")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    # Acceptance thresholds
    parser.add_argument("--cos-min", type=float, default=0.95,
                        help="Median cos(encode(generated), psi_target) gate on training set.")
    parser.add_argument("--grammar-pass-rate", type=float, default=0.80,
                        help="Min fraction of training sentences passing grammar gate.")
    parser.add_argument("--out", default="results/stage2a/microprototype.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Phase 2a sub-task 2a.0b — Cosmos-style micro-prototype (capacity test)")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)
    DIM = enc_cfg["dim"]

    # ---- Corpus -------------------------------------------------------
    sents = build_corpus()
    n = len(sents)
    print(f"\nCorpus: {n} templated sentences (training-only capacity test)")
    # Show a few samples for transparency
    print("  samples:")
    for s in sents[:5]:
        print(f"    {s!r}")

    # Tokenize all sentences with the encoder's own tokenizer.
    # We use the same tokenizer so the decoder's vocab matches the encoder's.
    tokenized = tok(
        sents,
        padding="max_length", truncation=True,
        max_length=args.t_max, return_tensors="pt",
    ).to(args.device)
    target_ids = tokenized.input_ids                 # [N, T]
    target_mask = tokenized.attention_mask.float()   # [N, T]
    n_padded = (target_mask < 0.5).all(dim=-1).sum().item()
    if n_padded > 0:
        raise SystemExit(f"FATAL: {n_padded} sentences fully padded — "
                         f"increase --t-max above {args.t_max}.")
    longest = int(target_mask.sum(dim=-1).max().item())
    print(f"  longest tokenized length: {longest} (t_max={args.t_max})")

    # Encode all sentences (psi targets) — done once, cached.
    print("\nEncoding sentences (computing ψ targets) ...")
    psi_targets = encode(sents)                       # [N, DIM]
    print(f"  psi_targets shape: {tuple(psi_targets.shape)}")

    # ---- Decoder ------------------------------------------------------
    torch.manual_seed(args.seed)
    decoder = TinyNonARDecoder(
        encoder_dim=DIM,
        hidden_dim=args.hidden_dim,
        n_cond=args.n_cond,
        t_max=args.t_max,
        vocab_size=tok.vocab_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_mult=args.ffn_mult,
    ).to(args.device)
    n_params = sum(p.numel() for p in decoder.parameters())
    n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nDecoder: {n_params/1e6:.2f}M params total, "
          f"{n_trainable/1e6:.2f}M trainable")
    print(f"  layers={args.n_layers}  hidden={args.hidden_dim}  "
          f"heads={args.n_heads}  N_cond={args.n_cond}  T_max={args.t_max}")

    # ---- Train --------------------------------------------------------
    print(f"\nTraining {args.steps} steps  batch={args.batch_size}  lr={args.lr}")
    opt = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.01,
    )

    decoder.train()
    loss_history: list[float] = []
    log_every = max(1, args.steps // 20)
    for step in range(args.steps):
        opt.zero_grad()
        # Sample a random batch with replacement (the corpus is small).
        idx = torch.randint(0, n, (args.batch_size,), device=args.device)
        psi_batch = psi_targets[idx]                 # [B, DIM]
        ids_batch = target_ids[idx]                  # [B, T]

        logits = decoder(psi_batch)                  # [B, T, V]

        # Parallel position-wise cross-entropy (NOT next-token prediction —
        # all positions predicted in parallel, attention is bidirectional).
        # ALL positions including [PAD] participate in the loss: pad
        # positions have target=pad_token_id so the decoder learns to
        # output [PAD] there. Masking padding OUT of the loss left those
        # positions un-supervised, and at inference the decoder filled
        # them with random common tokens that dragged the re-encoded ψ
        # off-target (the v1 bug — see commit log of this script).
        # tokenizer.decode(skip_special_tokens=True) strips [PAD] cleanly.
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids_batch.reshape(-1),
        )

        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        opt.step()
        loss_history.append(float(ce_loss.item()))

        if step % log_every == 0 or step == args.steps - 1:
            print(f"  step {step:>5}/{args.steps}: ce_loss={ce_loss.item():.4f}")

    # ---- Eval on training set (capacity test) -------------------------
    print("\n" + "=" * 78)
    print("EVALUATION (on training corpus — capacity test)")
    print("=" * 78)
    decoder.eval()
    with torch.no_grad():
        all_logits = decoder(psi_targets)            # [N, T, V]
        gen_ids = all_logits.argmax(dim=-1)          # [N, T]

    results: list[dict] = []
    n_cos_pass = 0
    n_grammar_pass = 0
    for i in range(n):
        target_text = sents[i]
        # Decode generated tokens; skip special tokens / pads.
        gen_text = tok.decode(gen_ids[i].tolist(), skip_special_tokens=True)

        # Cos-recovery: re-encode generated text and compare to target ψ.
        psi_gen = encode([gen_text])[0]
        cos_recovered = float(F.cosine_similarity(
            psi_gen, psi_targets[i], dim=0,
        ).item())

        # Grammar (LanguageTool if available, else wordfreq proxy).
        n_errors, grammar_pass = grammar_grade(gen_text)
        proxy = grammar_proxy(gen_text)

        if cos_recovered >= args.cos_min:
            n_cos_pass += 1
        if grammar_pass:
            n_grammar_pass += 1

        results.append({
            "target": target_text,
            "generated": gen_text,
            "cos_recovered": cos_recovered,
            "lt_n_errors": n_errors,
            "grammar_pass": grammar_pass,
            "grammar_proxy": proxy,
            "exact_match": gen_text.strip() == target_text.strip(),
        })

    # Print first 10 sentences for qualitative inspection
    print(f"\nPer-sentence (first 10 of {n}):")
    for i, r in enumerate(results[:10]):
        cos_mark = "✓" if r["cos_recovered"] >= args.cos_min else "✗"
        gr_mark = "✓" if r["grammar_pass"] else "✗"
        em_mark = " (exact)" if r["exact_match"] else ""
        print(f"  [{i+1}] target:    {r['target']!r}")
        print(f"      generated: {r['generated']!r}{em_mark}")
        print(f"      {cos_mark} cos={r['cos_recovered']:.4f}  "
              f"{gr_mark} LT errors={r['lt_n_errors']}  "
              f"proxy={r['grammar_proxy']:.3f}")

    # ---- Verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    coses = sorted(r["cos_recovered"] for r in results)
    median_cos = coses[len(coses) // 2]
    n_exact = sum(1 for r in results if r["exact_match"])
    grammar_target = int(args.grammar_pass_rate * n)

    print(f"Median cos:                      {median_cos:.4f}  "
          f"(target ≥ {args.cos_min})")
    print(f"Sentences passing cos:           {n_cos_pass}/{n}")
    print(f"Sentences passing grammar gate:  {n_grammar_pass}/{n}  "
          f"(target ≥ {grammar_target})")
    print(f"Exact target matches:            {n_exact}/{n}")
    print(f"Final training CE loss:          {loss_history[-1]:.4f}")
    if loss_history[0] > 0:
        print(f"  (reduction: {loss_history[0]:.4f} → {loss_history[-1]:.4f})")

    cos_gate = median_cos >= args.cos_min
    grammar_gate = n_grammar_pass >= grammar_target

    if cos_gate and grammar_gate:
        verdict = "CAPACITY_PASS"
        message = (
            "The architecture has capacity to memorize the 50-sentence "
            "training corpus from a single ψ vector AND produce "
            "grammatical output. Single-ψ conditioning is viable. "
            "Proceed to sub-task 2a.0c (generalization on 1000 train + "
            "200 held-out, with full paper recipe: CE + activation MSE + "
            "perturbation augmentation + feature dropout)."
        )
    elif cos_gate and not grammar_gate:
        verdict = "GRAMMAR_FAIL"
        message = (
            "Capacity is fine (cos passes) but text isn't grammatical. "
            "The paper's ablation showed CE-only hits MAUVE 0.29 and "
            "needs MSE on activations (+58%) and perturbation (+36%). "
            "Add full augmentation recipe (Gaussian δ=0.7, 30% mask, "
            "feature dropout p=0.4, MSE on encoder activations) and retry."
        )
    elif not cos_gate and grammar_gate:
        verdict = "GRAMMAR_OK_BUT_COS_FAILS"
        message = (
            "Unusual: text is grammatical but doesn't recover ψ. The "
            "decoder is producing fluent text but not text faithful to "
            "the conditioning. Investigate: ψ may not carry enough "
            "information for the decoder to disambiguate sentences, or "
            "the conditioning bottleneck is too narrow."
        )
    else:
        verdict = "CAPACITY_FAIL"
        message = (
            "Architecture cannot memorize 50 sentences from a single "
            "1024-dim ψ. Single-pooled-vector conditioning appears too "
            "lossy for non-trivial sentence reconstruction. Two follow-"
            "ups: (i) try richer conditioning — Stage 1 PsiProgram "
            "exposes per-step ψ states (a SEQUENCE of ψ vectors) which "
            "matches the paper's setup more faithfully; (ii) escalate "
            "to plan §13.9 fallback (templated structural backbone for "
            "English explanations)."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ----------------------------------------------------
    payload = {
        "task": "2a.0b",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "n_corpus": n,
        "decoder": {
            "n_params_total": n_params,
            "n_params_trainable": n_trainable,
            "hidden_dim": args.hidden_dim,
            "n_cond": args.n_cond,
            "t_max": args.t_max,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "ffn_mult": args.ffn_mult,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "loss_first": loss_history[0] if loss_history else None,
            "loss_last": loss_history[-1] if loss_history else None,
            "loss_history_every": [
                {"step": s, "loss": loss_history[s]}
                for s in range(0, args.steps, log_every)
                if s < len(loss_history)
            ],
        },
        "thresholds": {
            "cos_min": args.cos_min,
            "grammar_pass_rate": args.grammar_pass_rate,
            "grammar_pass_target_count": grammar_target,
        },
        "summary": {
            "median_cos": median_cos,
            "n_cos_pass": n_cos_pass,
            "n_grammar_pass": n_grammar_pass,
            "n_exact_match": n_exact,
            "n_total": n,
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


if __name__ == "__main__":
    main()
