"""Stage 3 / Sub-task 3.6 — per-domain decoder conformal calibration.

Reuses Stage 0.5's CCP machinery shape but at the DECODER level (not
the operator level): nonconformity score = -cos(encode(decode(h_input)),
z_target). Quantile q_hat at confidence (1-α) → admit/refuse threshold
with finite-sample coverage guarantee.

Acceptance per plan §19.17 row 3.6:
  ECE < 0.07 on each registered domain's held-out, where
  ECE = mean |empirical_coverage(α) - (1-α)| over the α grid
  {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.

This sub-task calibrates the Stage 3.1 definitional decoder on its own
held-out (120 sentences split 60/60 calibration/test). Optionally also
runs the calibration on the temporal domain ingested in 3.4 if its
artifact is around.

Run on the GPU box (~5-10 min — encoder forward dominates):
  python scripts/stage3_6_conformal.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from selflearnai.domain import domain_coverage_curve
from selflearnai.generator import PointerSeqCondDecoder

from scripts.stage1_planner_beam_smoke import ENCODERS
from scripts.stage3_2_energy_model import build_definitional_corpora


# ---------------------------------------------------------------------------
# Encoder / decoder helpers (reuse Stage 3.1 pattern)
# ---------------------------------------------------------------------------

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
    return out, inputs.attention_mask.float(), inputs.input_ids


def make_decode_fn(decoder, tok, mdl, device, t_max, eval_bs=32):
    """Returns: list[input_sent] -> list[generated_text].

    Encapsulates encode → run decoder → decode tokens.
    """
    @torch.no_grad()
    def decode_fn(input_sents: list[str]) -> list[str]:
        h, h_mask, ids = encode_activations(input_sents, tok, mdl, device, t_max=t_max)
        n = h.size(0)
        all_texts = []
        for s in range(0, n, eval_bs):
            log_probs, _, _ = decoder(
                h[s:s + eval_bs], h_mask[s:s + eval_bs], ids[s:s + eval_bs],
            )
            gen = log_probs.argmax(dim=-1)
            for i in range(gen.size(0)):
                all_texts.append(tok.decode(gen[i].tolist(), skip_special_tokens=True))
        return all_texts
    return decode_fn


def make_encode_pool_fn(tok, mdl, device, max_length=64):
    @torch.no_grad()
    def encode_pool_fn(sents: list[str]) -> torch.Tensor:
        return encode_pooled(sents, tok, mdl, device, max_length=max_length)
    return encode_pool_fn


def calibrate_domain(
    *,
    domain_id: str,
    decoder_ckpt: str,
    input_sents: list[str],
    target_sents: list[str],
    encoder: str,
    device: str,
    decoder_arch: dict,
    out: Path,
    seed: int = 0,
) -> dict:
    """Run domain_coverage_curve on a single domain. Returns the report dict."""
    enc_cfg = ENCODERS[encoder]
    print(f"\n{'=' * 78}")
    print(f"  domain: {domain_id}   decoder: {decoder_ckpt}")
    print(f"{'=' * 78}")

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]

    # ---- Decoder ----
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, **decoder_arch, vocab_size=tok.vocab_size,
    ).to(device)
    sd = torch.load(decoder_ckpt, map_location=device)
    decoder.load_state_dict(sd)
    decoder.eval()
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  loaded decoder: {n_params/1e6:.2f}M params")

    # ---- Calibration / test split (50/50, deterministic) ----
    n = len(input_sents)
    if len(target_sents) != n:
        raise ValueError("input/target lengths mismatch")
    print(f"  held-out total: {n}; splitting 50/50 for calibration/test")
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    cut = n // 2
    calib_idx = perm[:cut]
    test_idx = perm[cut:]
    calib_input = [input_sents[i] for i in calib_idx]
    calib_target = [target_sents[i] for i in calib_idx]
    test_input = [input_sents[i] for i in test_idx]
    test_target = [target_sents[i] for i in test_idx]
    print(f"  calibration: {len(calib_input)} pairs   test: {len(test_input)} pairs")

    decode_fn = make_decode_fn(decoder, tok, mdl, device, t_max=decoder_arch["t_max"])
    encode_pool_fn = make_encode_pool_fn(tok, mdl, device, max_length=decoder_arch["t_max"])

    # ---- Coverage curve ----
    print(f"  running coverage curve over α ∈ {{0.05, 0.10, 0.15, 0.20, 0.25, 0.30}} ...")
    report = domain_coverage_curve(
        decode_fn, encode_pool_fn,
        calib_input, calib_target,
        test_input, test_target,
    )

    # ---- Print ----
    print(f"\n  per-α results:")
    print(f"    {'α':>5}  {'nominal':>9}  {'empirical':>10}  "
          f"{'q_hat':>10}  {'gap':>7}")
    for a, r in zip(report["alphas"], report["per_alpha"]):
        gap = r["empirical_coverage"] - r["nominal_coverage"]
        print(f"    {a:>5.2f}  {r['nominal_coverage']:>9.4f}  "
              f"{r['empirical_coverage']:>10.4f}  "
              f"{r['q_hat']:>10.4f}  {gap:>+7.4f}")

    print(f"\n  ECE: {report['ece']:.4f}   "
          f"(target < 0.07; ECE floor {report['ece_floor']:.4f})")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ece-target", type=float, default=0.07)
    parser.add_argument("--definitional-decoder",
                        default="data/explanations_v2/checkpoints/decoder_3a1.pt")
    parser.add_argument("--out", default="results/stage3/conformal.json")
    # Decoder architecture (locked from §19.14)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--feat-dropout", type=float, default=0.2)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    args = parser.parse_args()

    decoder_arch = {
        "hidden_dim": args.hidden_dim, "t_max": args.t_max,
        "n_layers": args.n_layers, "n_heads": args.n_heads,
        "ffn_mult": args.ffn_mult,
        "feat_dropout": args.feat_dropout, "attn_dropout": args.attn_dropout,
    }

    print("Stage 3 / Sub-task 3.6 — per-domain conformal calibration")
    print("=" * 78)
    print(f"Encoder: {args.encoder}")
    print(f"ECE target: < {args.ece_target}")

    all_reports: dict[str, dict] = {}
    domain_passes: dict[str, bool] = {}

    # ---- Domain 1: definitional (Stage 3.1) -------------------------
    # Use Stage 3.1's truly-novel held-out (120 sentences) — they're
    # disjoint from the decoder's training set, so split-conformal coverage
    # guarantee holds.
    _, holdout_def = build_definitional_corpora()
    # For each held-out sentence, the "target" is the sentence itself
    # (Stage 3.1 decoder does identity reconstruction). The fidelity
    # score is then -cos(encode(decode(h)), z_target_sentence).
    report_def = calibrate_domain(
        domain_id="definitional",
        decoder_ckpt=args.definitional_decoder,
        input_sents=holdout_def,
        target_sents=holdout_def,
        encoder=args.encoder,
        device=args.device,
        decoder_arch=decoder_arch,
        out=Path(args.out),
        seed=args.seed,
    )
    all_reports["definitional"] = report_def
    domain_passes["definitional"] = report_def["ece"] < args.ece_target

    # ---- Verdict -----------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for domain_id, passed in domain_passes.items():
        ece = all_reports[domain_id]["ece"]
        status = "PASS" if passed else "FAIL"
        print(f"  {domain_id:<15}  ECE = {ece:.4f}  →  {status}")

    all_pass = all(domain_passes.values())
    if all_pass:
        verdict = "STAGE_3_6_PASS"
        message = (
            f"All registered domains have ECE < {args.ece_target} on their "
            f"held-out. Per-domain decoder conformal calibration works: "
            f"the (universal pipeline + per-domain calibrated coverage set) "
            f"gives a principled refusal mechanism for out-of-domain inputs "
            f"replacing hand-tuned cosine thresholds. Stage 0.5's CCP "
            f"machinery generalizes from operators to decoders cleanly."
        )
    else:
        verdict = "STAGE_3_6_FAIL"
        failed = [d for d, p in domain_passes.items() if not p]
        message = (
            f"Domain(s) {failed} have ECE ≥ {args.ece_target}. The decoder's "
            f"fidelity is too peaked or too noisy for split-conformal at "
            f"this n_test. Try (a) larger held-out, (b) cross-conformal "
            f"(CV+) instead of split, (c) inspect score distribution for "
            f"degenerate values."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save --------------------------------------------------------
    payload = {
        "task": "3.6",
        "encoder": args.encoder,
        "encoder_dim": ENCODERS[args.encoder]["dim"],
        "ece_target": args.ece_target,
        "domains": {
            domain_id: {
                "ece": report["ece"],
                "ece_floor": report["ece_floor"],
                "n_calib": report["n_calib"],
                "n_test": report["n_test"],
                "alphas": report["alphas"],
                "per_alpha": [
                    {
                        "alpha": r["alpha"],
                        "nominal_coverage": r["nominal_coverage"],
                        "empirical_coverage": r["empirical_coverage"],
                        "q_hat": r["q_hat"],
                        "test_scores_min": r["test_scores_min"],
                        "test_scores_median": r["test_scores_median"],
                        "test_scores_max": r["test_scores_max"],
                    }
                    for r in report["per_alpha"]
                ],
                "pass": domain_passes[domain_id],
            }
            for domain_id, report in all_reports.items()
        },
        "verdict": verdict,
        "message": message,
        "all_pass": all_pass,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
