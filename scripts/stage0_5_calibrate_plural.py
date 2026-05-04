"""Task 0.5.1 — per-operator conformal calibrator on the plural concept.

Builds the first calibrator on top of a fresh-trained plural operator
and reports empirical coverage + ECE. Single concept, single commit,
single acceptance gate.

Acceptance criteria (per plan §19.2 row 0.5.1):
  • Empirical coverage at α=0.1 within max(0.04, 1/n_test) of 0.9.
    With n_test=6 (plural held-out size) the floor is 1/6 ≈ 0.167,
    so the realistic gate is empirical ∈ [0.733, 1.0].
  • ECE across α ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} reported.
    Plural alone is granularity-limited (n_test=6 ⇒ ECE_floor ≈ 0.083),
    so the strict ECE < 0.05 gate fires only when we pool across
    concepts (Task 0.5.2). Here we report the number and check
    ECE < 0.10 as a per-concept sanity gate.

Pathway: text-only (no Stage 1 alignment), GTE-base — validated as
optimal for plural per RESULTS.md (Path 2 — text-only pathway).

Protocol (split-conformal):
  - Read 44 plural train pairs, 6 held-out pairs, 66 candidate pool words.
  - Random split (seed-controlled): 32 → operator training,
    12 → calibration. Held-out 6 → test.
  - Train fresh ConceptOperator on the 32. Frozen after training.
  - Calibrate with the 12. Disjoint from train and test ⇒ guarantee holds.
  - Evaluate on the 6 held-out at multiple α. Report.

Output:
  - Console: per-α coverage table, ECE, set-size stats, accept/fail line.
  - JSON: results/stage0_5/plural_conformal.json (machine-readable).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from selflearnai.concepts import ConceptOperator
from selflearnai.uncertainty import coverage_curve, ConformalOperatorCalibrator


ENCODERS = {
    "gte-base":    {"model": "thenlper/gte-base",    "dim":  768},
    "e5-large-v2": {"model": "intfloat/e5-large-v2", "dim": 1024},
}


# ---------------------------------------------------------------------------
# Data IO  (mirrors scripts/run_text_only_concepts.py — single-source style)
# ---------------------------------------------------------------------------

def _strip(s: str) -> str:
    """Strip inline TSV comments (everything after '#')."""
    idx = s.find("#")
    return (s[:idx] if idx >= 0 else s).strip()


def read_pairs(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                src, tgt = _strip(parts[0]), _strip(parts[1])
                if src and tgt:
                    out.append((src, tgt))
    return out


def read_pool(path: Path) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    with open(path) as f:
        for raw in f:
            w = _strip(raw.strip())
            if w and w not in seen:
                out.append(w)
                seen.add(w)
    return out


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_encode_fn(model, tokenizer, device: str, max_length: int = 64):
    """Return a `list[str] -> Tensor` closure with mean-pooled embeddings."""

    def encode(words: list[str]) -> torch.Tensor:
        inputs = tokenizer(
            words,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**inputs)
        last_hidden = out.last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.float()

    return encode


# ---------------------------------------------------------------------------
# Operator training
# ---------------------------------------------------------------------------

def train_operator(
    encode_fn, train_pairs: list[tuple[str, str]],
    *, dim: int, device: str, seed: int, epochs: int, lr: float,
) -> ConceptOperator:
    """Train a fresh ConceptOperator on (source -> target) pairs in
    encoder-native dim (no Stage 1 alignment)."""
    torch.manual_seed(seed)
    op = ConceptOperator(dim=dim).to(device)
    opt = torch.optim.AdamW(op.parameters(), lr=lr)
    z_src = encode_fn([p[0] for p in train_pairs])
    z_tgt = encode_fn([p[1] for p in train_pairs])
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(op(z_src), z_tgt)
        loss.backward()
        opt.step()
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op


# ---------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------

def acceptance_check(curve: dict, alpha_main: float = 0.10) -> dict:
    """Apply Task 0.5.1 gates and return PASS/FAIL with reasons."""
    n_test = curve["n_test"]
    floor = curve["ece_floor"]

    # Find the empirical coverage at the main alpha
    main = next((r for r in curve["per_alpha"] if abs(r["alpha"] - alpha_main) < 1e-9), None)
    if main is None:
        return {"pass": False, "reason": f"alpha={alpha_main} not in coverage curve"}

    nominal = main["nominal_coverage"]
    empirical = main["empirical_coverage"]

    # Coverage gate: within max(0.04, 1/n_test) of nominal — honest about
    # the granularity floor with small test sets.
    cov_tol = max(0.04, floor)
    cov_low = nominal - cov_tol
    cov_high = min(1.0, nominal + cov_tol)
    cov_pass = cov_low <= empirical <= cov_high

    # ECE gate: per-concept ECE realistically bounded near the discretization
    # floor; we accept ECE < 0.10 here. Strict ECE < 0.05 will be enforced
    # in Task 0.5.2 once we pool across all 7 concepts.
    ece_pass = curve["ece"] < 0.10

    return {
        "pass": cov_pass and ece_pass,
        "coverage_pass": cov_pass,
        "ece_pass": ece_pass,
        "alpha_main": alpha_main,
        "empirical_coverage": empirical,
        "nominal_coverage": nominal,
        "coverage_window": [cov_low, cov_high],
        "ece": curve["ece"],
        "ece_floor": floor,
        "ece_threshold": 0.10,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--data-dir", default="data/plurality")
    parser.add_argument("--encoder", default="gte-base", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=32,
                        help="How many train pairs to use for the operator; "
                             "remaining train pairs go to calibration.")
    parser.add_argument("--alpha-main", type=float, default=0.10,
                        help="Primary alpha for the coverage gate.")
    parser.add_argument("--out", default="results/stage0_5/plural_conformal.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print(f"Task 0.5.1 — plural conformal calibration")
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Pathway: text-only (no Stage 1 alignment)")

    # ---- Data ----
    data_dir = Path(args.data_dir)
    train_pairs_all = read_pairs(data_dir / "text_pairs_train.tsv")
    test_pairs = read_pairs(data_dir / "text_pairs_held_out.tsv")
    pool = read_pool(data_dir / "candidate_pool.txt")
    print(f"\nData: {len(train_pairs_all)} train, {len(test_pairs)} test, "
          f"{len(pool)} pool")

    if args.n_train >= len(train_pairs_all):
        raise SystemExit(
            f"--n-train ({args.n_train}) must leave >= 2 pairs for calibration; "
            f"only {len(train_pairs_all)} train pairs available."
        )

    # ---- Split: operator-train vs calibration ----
    rng = random.Random(args.seed)
    train_idx = list(range(len(train_pairs_all)))
    rng.shuffle(train_idx)
    op_train = [train_pairs_all[i] for i in train_idx[: args.n_train]]
    calib = [train_pairs_all[i] for i in train_idx[args.n_train:]]
    print(f"Split: {len(op_train)} operator-train, {len(calib)} calibration, "
          f"{len(test_pairs)} test  (seed={args.seed})")
    assert not (set(op_train) & set(calib)), "operator-train / calib leak"
    assert not (set(op_train) & set(test_pairs)), "operator-train / test leak"
    assert not (set(calib) & set(test_pairs)), "calib / test leak"
    print("Disjoint-set check: OK")

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Train operator ----
    print(f"\nTraining ConceptOperator (dim={enc_cfg['dim']}, "
          f"epochs={args.epochs}, lr={args.lr}) ...")
    op = train_operator(
        encode, op_train,
        dim=enc_cfg["dim"], device=args.device,
        seed=args.seed, epochs=args.epochs, lr=args.lr,
    )

    # Sanity: point-estimate accuracy on test (no calibration involved)
    with torch.no_grad():
        z_src = encode([p[0] for p in test_pairs])
        z_pool = encode(pool)
        z_pred = op(z_src)
        pred_n = F.normalize(z_pred, dim=-1)
        pool_n = F.normalize(z_pool, dim=-1)
        sims = pred_n @ pool_n.T
        argmax = sims.argmax(dim=-1).tolist()
        top1 = [pool[i] for i in argmax]
    correct = sum(p == t for p, t in zip(top1, [t for _, t in test_pairs]))
    print(f"Sanity (no calibration): top-1 accuracy on held-out = "
          f"{correct}/{len(test_pairs)}")

    # ---- Single-alpha calibrator (for printout) ----
    cal = ConformalOperatorCalibrator(alpha=args.alpha_main)
    cal.fit(op, encode, calib)
    main_eval = cal.evaluate(op, encode, test_pairs, pool)
    print(f"\nAt α={args.alpha_main}:  q_hat={cal.q_hat:.4f}  "
          f"empirical_coverage={main_eval['empirical_coverage']:.3f}  "
          f"(nominal {main_eval['nominal_coverage']:.3f})")
    print(f"Set sizes: mean={main_eval['mean_set_size']:.2f}  "
          f"median={main_eval['median_set_size']:.1f}  "
          f"max={main_eval['max_set_size']}")

    # ---- Coverage curve across multiple alphas ----
    curve = coverage_curve(op, encode, calib, test_pairs, pool,
                           alphas=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30))

    print("\nCoverage curve:")
    print(f"  {'alpha':>6}  {'nominal':>8}  {'empirical':>10}  {'mean|set|':>10}")
    for r in curve["per_alpha"]:
        print(f"  {r['alpha']:>6.2f}  {r['nominal_coverage']:>8.3f}  "
              f"{r['empirical_coverage']:>10.3f}  {r['mean_set_size']:>10.2f}")
    print(f"\nECE (mean |empirical − nominal| across α): {curve['ece']:.4f}")
    print(f"ECE granularity floor (1/n_test, n_test={curve['n_test']}): "
          f"{curve['ece_floor']:.4f}")

    # ---- Acceptance check ----
    accept = acceptance_check(curve, alpha_main=args.alpha_main)
    print("\n" + "=" * 60)
    print("ACCEPTANCE CHECK (Task 0.5.1)")
    print("=" * 60)
    print(f"  Coverage gate (α=0.10 ∈ {accept['coverage_window']}): "
          f"empirical={accept['empirical_coverage']:.3f} → "
          f"{'PASS' if accept['coverage_pass'] else 'FAIL'}")
    print(f"  ECE gate    (per-concept < {accept['ece_threshold']:.2f}): "
          f"ECE={accept['ece']:.4f} → "
          f"{'PASS' if accept['ece_pass'] else 'FAIL'}")
    overall = "PASS" if accept["pass"] else "FAIL"
    print(f"\n→ Task 0.5.1 ({args.encoder}): {overall}")

    # ---- Save JSON ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "0.5.1",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "seed": args.seed,
        "n_train_pairs": len(op_train),
        "n_calib_pairs": len(calib),
        "n_test_pairs": len(test_pairs),
        "sanity_top1": correct / len(test_pairs),
        "main_alpha_eval": {
            k: v for k, v in main_eval.items()
            if k not in ("in_set_flags", "set_sizes")  # numeric only in JSON
        },
        "main_alpha_in_set_flags": main_eval["in_set_flags"],
        "main_alpha_set_sizes": main_eval["set_sizes"],
        "coverage_curve": {
            "alphas": curve["alphas"],
            "ece": curve["ece"],
            "ece_floor": curve["ece_floor"],
            "n_test": curve["n_test"],
            "per_alpha": [
                {k: v for k, v in r.items() if k not in ("in_set_flags", "set_sizes")}
                for r in curve["per_alpha"]
            ],
        },
        "acceptance": accept,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")

    # Exit non-zero on FAIL so CI / scripts can react.
    raise SystemExit(0 if accept["pass"] else 1)


if __name__ == "__main__":
    main()
