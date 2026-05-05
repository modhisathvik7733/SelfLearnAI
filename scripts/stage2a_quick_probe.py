"""Phase 2a quick probe — does the encoder admit continuous-vector inputs?

The smallest, fastest empirical resolution of Approach A vs Approach B
(plan §19.12). ~1 minute of GPU time on a single A100.

Question: can we optimize a free continuous matrix X ∈ R^(T×D) so that
encode(X) ≈ ψ_target, AND does snapping X to its nearest E5-vocab
embeddings produce text that re-encodes back to ψ_target?

The probe runs in TWO snap modes:

  RAW SNAP — independent per-position argmax over the full vocabulary.
    Original probe behavior. Often picks subword BPE fragments that
    don't form coherent text.

  REFINED SNAP — start from raw snap restricted to complete words
    (no ##-prefixed BPE pieces), then run coordinate-ascent
    refinement: cycle through positions, try the top-K alternatives
    at each, accept any swap that improves the full re-encoded
    cos(encode(text), ψ_target). Discrete optimization; cos only
    increases. Converges in 2–3 passes.

The refined-snap result is the honest test of Approach A's ceiling.
If even refined snap can't recover ψ AND produce grammatical text,
Approach A is dead and we commit to B (Gumbel-STE discrete diffusion).

Both modes are reported per-sentence and as medians.

This probe bypasses tokenization by passing inputs_embeds directly to
the BertModel-style forward (E5 is XLMRoberta-based, supports the same
inputs_embeds kwarg). The continuous matrix X plays the role of the
output of the word-embedding lookup; the rest of the encoder
processes it normally.

5 target sentences are tested independently. Verdict is based on
medians across the 5 under the BETTER (refined) snap.

Run:
  python scripts/stage2a_quick_probe.py
  python scripts/stage2a_quick_probe.py --device cuda --steps 2000
  python scripts/stage2a_quick_probe.py --no-refine    # raw-only legacy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


TARGET_SENTENCES = [
    "the plural of cat is cats",
    "the past tense of run is ran",
    "the comparative of big is bigger",
    "a person who paints is a painter",
    "the opposite of hot is cold",
]

# Verdict thresholds — same shape as the full plan's 2a.0 gate.
LOSS_OK = 1e-3
COS_RECOVERED_OK = 0.85


def build_complete_word_mask(tok) -> torch.Tensor:
    """Return a [vocab_size] bool tensor: True for tokens that look like
    complete words. Filters out ## BPE continuations + special tokens.

    Heuristic: keep tokens whose decoded form starts with a non-#, has
    length ≥ 2 (drops single-letter junk), and is alphanumeric or
    contains common punctuation. Dropping these is what fixes the
    'water catss covers thames pluraled' word-salad problem in the raw
    snap.
    """
    vocab_size = len(tok)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    special_ids = set(tok.all_special_ids)
    for i in range(vocab_size):
        if i in special_ids:
            continue
        s = tok.convert_ids_to_tokens(i)
        if not s or s.startswith("##"):
            continue
        if len(s) < 2:
            continue
        # Allow alphanumeric + apostrophe + dash. Excludes weird
        # symbols that snuck into the snap last run (⁺, etc.)
        if not all(c.isalnum() or c in "'-" for c in s):
            continue
        mask[i] = True
    return mask


def refine_tokens(
    initial_tokens: list[int],
    X_pos: torch.Tensor,            # [T, D] — the optimized per-position vectors
    word_emb: torch.Tensor,         # [V, D]
    *,
    encode_fn,                      # callable: list[str] -> tensor [B, D]
    psi_target: torch.Tensor,       # [D]
    tokenizer,
    vocab_mask: torch.Tensor,       # [V] bool — restrict candidates
    top_k: int = 20,
    max_iters: int = 5,
) -> tuple[list[int], float, int]:
    """Coordinate-ascent refinement over discrete tokens.

    Cycle through positions. At each, get the top-K candidate tokens
    (by cosine to that position's optimized vector, restricted by
    vocab_mask). Try each candidate; keep the swap iff it improves
    the full re-encoded cos(encode(text), psi_target).

    Returns (refined_tokens, final_cos, iterations_used).
    """
    tokens = list(initial_tokens)

    def cos_of_tokens(toks: list[int]) -> float:
        text = tokenizer.decode(toks, skip_special_tokens=True)
        if not text.strip():
            return -1.0
        psi = encode_fn([text])[0]
        return float(F.cosine_similarity(psi, psi_target, dim=0).item())

    # Restrict per-position similarity scores to the masked vocab.
    we_n = F.normalize(word_emb, dim=-1)                # [V, D]
    X_n = F.normalize(X_pos, dim=-1)                    # [T, D]
    sims = X_n @ we_n.T                                  # [T, V]
    masked_sims = sims.clone()
    masked_sims[:, ~vocab_mask] = -1e9
    top_k_per_pos = masked_sims.topk(top_k, dim=-1).indices.tolist()  # [T][K]

    current_cos = cos_of_tokens(tokens)
    iters_used = 0
    for it in range(max_iters):
        any_improvement = False
        for pos in range(len(tokens)):
            best_token = tokens[pos]
            best_cos = current_cos
            for cand in top_k_per_pos[pos]:
                if cand == best_token:
                    continue
                tokens[pos] = cand
                new_cos = cos_of_tokens(tokens)
                if new_cos > best_cos + 1e-6:
                    best_cos = new_cos
                    best_token = cand
                    any_improvement = True
            tokens[pos] = best_token
            current_cos = best_cos
        iters_used = it + 1
        if not any_improvement:
            break
    return tokens, current_cos, iters_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Number of continuous positions to optimize.")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Adam steps per target sentence.")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--refine", dest="refine", action="store_true", default=True,
        help="Run coordinate-ascent refinement after raw snap (default).",
    )
    parser.add_argument(
        "--no-refine", dest="refine", action="store_false",
        help="Skip refinement (legacy raw-snap-only mode).",
    )
    parser.add_argument(
        "--refine-top-k", type=int, default=20,
        help="Per-position candidate count during refinement.",
    )
    parser.add_argument(
        "--refine-max-iters", type=int, default=5,
        help="Max coordinate-ascent passes (early-stops on convergence).",
    )
    parser.add_argument("--out", default="results/stage2a/quick_probe.json")
    args = parser.parse_args()

    print("Phase 2a quick probe — encoder-input sanity")
    print("=" * 70)
    print(f"Encoder: {args.encoder}")
    print(f"Device:  {args.device}")
    print(f"T (positions): {args.seq_len}   steps/target: {args.steps}   lr: {args.lr}")

    print(f"\nLoading {args.encoder} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.encoder)
    mdl = AutoModel.from_pretrained(args.encoder).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = mdl.config.hidden_size
    print(f"  hidden_size: {DIM}")

    word_emb = mdl.get_input_embeddings().weight        # [V, D]
    we_std = float(word_emb.std().item())
    print(f"  vocab size: {word_emb.shape[0]}, word_emb std: {we_std:.4f}")

    # ---- Encoders -----------------------------------------------------
    def encode_text(texts: list[str]) -> torch.Tensor:
        """Standard mean-pooled encode (mirrors stage1's make_encode_fn).

        Wrapped in no_grad — we only need its OUTPUT as a target (no
        gradients flow back to the model from this path)."""
        with torch.no_grad():
            batch = tok(
                texts, padding=True, truncation=True, max_length=64,
                return_tensors="pt",
            ).to(args.device)
            out = mdl(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def encode_continuous(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode pre-computed embeddings. Bypasses tokenization but
        keeps the rest of the model's forward path identical
        (position embeddings, layernorm, attention, mean-pool).

        NOT wrapped in no_grad — gradients DO flow back to
        `inputs_embeds` so we can optimize it. Model parameters
        themselves remain frozen (set above)."""
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long, device=inputs_embeds.device,
            )
        out = mdl(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    # ---- [0] Consistency check: encode_text vs encode_continuous on real tokens
    print("\n[0] Consistency check (paths agree on real tokens?)")
    print("-" * 70)
    sentence = TARGET_SENTENCES[0]
    psi_text = encode_text([sentence])[0]
    with torch.no_grad():
        batch = tok(
            [sentence], padding=True, truncation=True, max_length=64,
            return_tensors="pt",
        ).to(args.device)
        real_embeds = mdl.get_input_embeddings()(batch["input_ids"])
        psi_cont = encode_continuous(real_embeds, batch["attention_mask"])[0]
    consistency_cos = float(F.cosine_similarity(psi_text, psi_cont, dim=0).item())
    print(f"  cos(encode_text, encode_continuous on real tokens) = {consistency_cos:.6f}")
    if consistency_cos < 0.99:
        print("  WARNING: paths diverge — investigate before trusting probe results")
    else:
        print("  OK: paths produce ~identical ψ for real tokens")

    # ---- [1] Per-sentence probe ---------------------------------------
    torch.manual_seed(args.seed)

    # Build the complete-word vocabulary mask once. Used during
    # refinement to filter out ## BPE pieces and weird symbols.
    print("\nBuilding complete-word vocabulary mask ...")
    complete_mask = build_complete_word_mask(tok).to(args.device)
    n_kept = int(complete_mask.sum().item())
    print(f"  vocab kept under complete-word filter: "
          f"{n_kept}/{complete_mask.numel()}")

    results: list[dict] = []
    print("\n[1] Per-sentence probe")
    print("-" * 70)
    for i, sent in enumerate(TARGET_SENTENCES):
        psi_target = encode_text([sent])[0]                    # [D]

        # Initialize X randomly, scaled to word-embedding distribution.
        X = torch.randn(1, args.seq_len, DIM, device=args.device) * we_std
        X.requires_grad_(True)
        opt = torch.optim.Adam([X], lr=args.lr)

        for step in range(args.steps):
            opt.zero_grad()
            psi_pred = encode_continuous(X)[0]
            loss = F.mse_loss(psi_pred, psi_target)
            loss.backward()
            opt.step()
        final_loss = float(loss.item())

        # ---- RAW SNAP: independent argmax over full vocabulary -------
        with torch.no_grad():
            X_n = F.normalize(X[0], dim=-1)                    # [T, D]
            we_n = F.normalize(word_emb, dim=-1)               # [V, D]
            sims = X_n @ we_n.T                                # [T, V]
            raw_tokens = sims.argmax(dim=-1).tolist()
        raw_text = tok.decode(raw_tokens, skip_special_tokens=True)
        psi_recovered_raw = encode_text([raw_text])[0]
        cos_raw = float(F.cosine_similarity(
            psi_recovered_raw, psi_target, dim=0,
        ).item())

        # Sanity: cos at the optimization endpoint (BEFORE snap).
        with torch.no_grad():
            psi_pred_final = encode_continuous(X)[0]
            cos_pre_snap = float(F.cosine_similarity(
                psi_pred_final, psi_target, dim=0,
            ).item())

        # ---- REFINED SNAP (default): start from complete-word argmax,
        # then coordinate-ascent over discrete tokens. Cos can ONLY
        # increase from the starting point.
        if args.refine:
            with torch.no_grad():
                masked_sims = sims.clone()
                masked_sims[:, ~complete_mask] = -1e9
                init_tokens_refined = masked_sims.argmax(dim=-1).tolist()
            ref_tokens, cos_refined, ref_iters = refine_tokens(
                init_tokens_refined,
                X[0].detach(),
                word_emb,
                encode_fn=encode_text,
                psi_target=psi_target,
                tokenizer=tok,
                vocab_mask=complete_mask,
                top_k=args.refine_top_k,
                max_iters=args.refine_max_iters,
            )
            ref_text = tok.decode(ref_tokens, skip_special_tokens=True)
        else:
            ref_tokens = raw_tokens
            ref_text = raw_text
            cos_refined = cos_raw
            ref_iters = 0

        # Headline cos for verdict purposes is the BETTER (refined)
        # score — we want to know Approach A's CEILING, not its floor.
        cos_recovered = cos_refined

        record = {
            "target": sent,
            "final_loss": final_loss,
            "cos_pre_snap": cos_pre_snap,
            "raw_snap": {
                "text": raw_text,
                "token_ids": raw_tokens,
                "cos_recovered": cos_raw,
            },
            "refined_snap": {
                "text": ref_text,
                "token_ids": ref_tokens,
                "cos_recovered": cos_refined,
                "iterations": ref_iters,
            },
            "cos_recovered_after_snap": cos_recovered,
        }
        results.append(record)

        loss_mark = "✓" if final_loss < LOSS_OK else "✗"
        cos_mark_raw = "✓" if cos_raw >= COS_RECOVERED_OK else "✗"
        cos_mark_ref = "✓" if cos_refined >= COS_RECOVERED_OK else "✗"
        print(f"\n  [{i+1}/{len(TARGET_SENTENCES)}] target: {sent!r}")
        print(f"    {loss_mark} final loss:        {final_loss:.6f} "
              f"(threshold < {LOSS_OK})")
        print(f"      cos(pre-snap, target):  {cos_pre_snap:.4f}")
        print(f"      RAW snap:               {raw_text!r}")
        print(f"    {cos_mark_raw} cos(re-encoded raw, target):     "
              f"{cos_raw:.4f} (threshold ≥ {COS_RECOVERED_OK})")
        if args.refine:
            print(f"      REFINED snap ({ref_iters} iters): {ref_text!r}")
            print(f"    {cos_mark_ref} cos(re-encoded refined, target): "
                  f"{cos_refined:.4f} (threshold ≥ {COS_RECOVERED_OK})  "
                  f"Δ over raw: {cos_refined - cos_raw:+.4f}")

    # ---- Verdict -----------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    losses = sorted(r["final_loss"] for r in results)
    coses = sorted(r["cos_recovered_after_snap"] for r in results)
    median_loss = losses[len(losses) // 2]
    median_cos = coses[len(coses) // 2]
    n_loss_ok = sum(1 for r in results if r["final_loss"] < LOSS_OK)
    n_cos_ok = sum(1 for r in results if r["cos_recovered_after_snap"] >= COS_RECOVERED_OK)

    print(f"Median final loss:               {median_loss:.6f}  "
          f"(target < {LOSS_OK})")
    print(f"Median cos(re-encoded, target):  {median_cos:.4f}  "
          f"(target ≥ {COS_RECOVERED_OK})")
    print(f"Sentences passing loss gate:     {n_loss_ok}/{len(results)}")
    print(f"Sentences passing recovery gate: {n_cos_ok}/{len(results)}")

    loss_gate = median_loss < LOSS_OK
    cos_gate = median_cos >= COS_RECOVERED_OK

    if loss_gate and cos_gate:
        verdict = "APPROACH_A_VIABLE"
        message = (
            "Both gates passed. The encoder accepts continuous inputs AND "
            "the token-snap step recovers meaning. Approach A is on the "
            "table. Plan revises: Phase 2a uses a continuous-output "
            "decoder + token snap at inference."
        )
    elif loss_gate and not cos_gate:
        verdict = "OFF_MANIFOLD_CONFIRMED"
        message = (
            "Optimization converges but snapped tokens don't recover ψ. "
            "E5 satisfies the loss with adversarial continuous solutions "
            "that don't correspond to coherent token sequences. Approach A "
            "is structurally dead. Commit to Approach B (Gumbel-STE "
            "discrete diffusion) per plan §19.12."
        )
    elif not loss_gate and cos_gate:
        verdict = "ANOMALY_LOSS_HIGH_BUT_RECOVERY_OK"
        message = (
            "Unexpected: loss didn't fully converge but the snap still "
            "recovered ψ. Investigate optimization (more steps? higher "
            "lr?) before committing to either approach."
        )
    else:
        verdict = "ENCODER_GRADIENTS_UNWORKABLE"
        message = (
            "Optimization didn't converge. The encoder's gradient surface "
            "is too rough for direct optimization, which would also hurt "
            "Approach B's Gumbel-STE training. Investigate before "
            "either approach."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ----------------------------------------------------
    payload = {
        "task": "2a.quick_probe",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "thresholds": {
            "loss_ok": LOSS_OK,
            "cos_recovered_ok": COS_RECOVERED_OK,
        },
        "consistency_cos_real_tokens": consistency_cos,
        "per_sentence": results,
        "summary": {
            "median_loss": median_loss,
            "median_cos_recovered": median_cos,
            "n_loss_ok": n_loss_ok,
            "n_cos_ok": n_cos_ok,
            "n_total": len(results),
        },
        "verdict": verdict,
        "message": message,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")


if __name__ == "__main__":
    main()
