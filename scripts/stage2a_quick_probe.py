"""Phase 2a quick probe — does the encoder admit continuous-vector inputs?

The smallest, fastest empirical resolution of Approach A vs Approach B
(plan §19.12). ~30 seconds of GPU time on a single A100.

Question: can we optimize a free continuous matrix X ∈ R^(T×D) so that
encode(X) ≈ ψ_target, AND does snapping X to its nearest E5-vocab
embeddings produce text that re-encodes back to ψ_target?

  - If YES (loss → 0 AND cos(re-encoded, target) ≥ 0.85):
        Approach A is viable. Continue Phase 2a with continuous-output
        decoder + token snap.
  - If NO (loss → 0 BUT cos low):
        E5 finds adversarial continuous solutions that don't correspond
        to any token sequence. Off-manifold confirmed. Commit to
        Approach B (Gumbel-STE discrete diffusion).
  - If loss DOESN'T converge:
        Encoder gradients are unworkable for direct optimization.
        Investigate before either approach.

This probe bypasses tokenization by passing inputs_embeds directly to
the BertModel-style forward (E5 is XLMRoberta-based, supports the same
inputs_embeds kwarg). The continuous matrix X plays the role of the
output of the word-embedding lookup; the rest of the encoder
processes it normally.

5 target sentences are tested independently. Verdict is based on
medians across the 5.

Run:
  python scripts/stage2a_quick_probe.py
  python scripts/stage2a_quick_probe.py --device cuda --steps 2000
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

        # Snap each row of X to nearest vocab token by cosine.
        with torch.no_grad():
            X_n = F.normalize(X[0], dim=-1)                    # [T, D]
            we_n = F.normalize(word_emb, dim=-1)               # [V, D]
            sims = X_n @ we_n.T                                # [T, V]
            tokens = sims.argmax(dim=-1).tolist()
        snapped_text = tok.decode(tokens, skip_special_tokens=True)

        # Re-encode the snapped text and compare to target.
        psi_recovered = encode_text([snapped_text])[0]
        cos_recovered = float(F.cosine_similarity(
            psi_recovered, psi_target, dim=0,
        ).item())

        # Sanity: cos at the optimization endpoint (BEFORE snap), should be very high.
        with torch.no_grad():
            psi_pred_final = encode_continuous(X)[0]
            cos_pre_snap = float(F.cosine_similarity(
                psi_pred_final, psi_target, dim=0,
            ).item())

        record = {
            "target": sent,
            "final_loss": final_loss,
            "cos_pre_snap": cos_pre_snap,
            "snapped_text": snapped_text,
            "snapped_token_ids": tokens,
            "cos_recovered_after_snap": cos_recovered,
        }
        results.append(record)

        loss_mark = "✓" if final_loss < LOSS_OK else "✗"
        cos_mark = "✓" if cos_recovered >= COS_RECOVERED_OK else "✗"
        print(f"\n  [{i+1}/{len(TARGET_SENTENCES)}] target: {sent!r}")
        print(f"    {loss_mark} final loss:        {final_loss:.6f} "
              f"(threshold < {LOSS_OK})")
        print(f"      cos(pre-snap, target):  {cos_pre_snap:.4f}")
        print(f"      snapped text:           {snapped_text!r}")
        print(f"    {cos_mark} cos(re-encoded, target):  {cos_recovered:.4f} "
              f"(threshold ≥ {COS_RECOVERED_OK})")

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
