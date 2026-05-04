"""Task 0.5.2 — extend conformal calibration to all 7 concepts; pool ECE.

Builds on Task 0.5.1's ConformalOperatorCalibrator. Each concept gets a
fresh operator + calibration; results are POOLED across concepts to get
a smoother empirical-coverage estimate (granularity 1/42 vs 1/6 per
concept), enabling the strict pooled-ECE < 0.05 gate.

Concepts:
  Full-data (split-conformal):  plural, past_tense, comparative, opposite_v2
  Few-shot  (leave-one-out CV):  agentive, superlative, young

Protocol:
  - Full-data: shuffle train (seeded). Use the first N_OP for operator
    training; the remaining (>= 2) as calibration. Test on the dataset's
    held-out file.
  - Few-shot (N_train = 3): train operator on all 3 train pairs. For each
    held-out pair i in 1..6: calibrate on the OTHER 5 held-outs, test on i.

Per α ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}:
  - Per-concept empirical coverage = mean(in_set across that concept's trials)
  - Pooled empirical coverage = mean(in_set across ALL concepts' trials)
  - Pooled ECE = mean over α of | pooled_empirical − (1 − α) |

Acceptance gate (Task 0.5.2):
  Pooled ECE < 0.05.
  (Per-concept ECE reported but not gated; granularity floor 1/6 per
  concept makes per-concept < 0.05 unreasonable.)
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
from selflearnai.uncertainty import ConformalOperatorCalibrator


ENCODERS = {
    "gte-base":    {"model": "thenlper/gte-base",    "dim":  768},
    "e5-large-v2": {"model": "intfloat/e5-large-v2", "dim": 1024},
}


CONCEPTS: list[dict] = [
    # Full-data path (split-conformal). N_op_train chosen so >= 2 calib.
    {"name": "plural",       "data_dir": "data/plurality",         "mode": "split", "n_op_train": 32},
    {"name": "past_tense",   "data_dir": "data/past_tense",        "mode": "split", "n_op_train": 32},
    {"name": "comparative",  "data_dir": "data/comparative",       "mode": "split", "n_op_train": 22},
    {"name": "opposite_v2",  "data_dir": "data/opposite_v2",       "mode": "split", "n_op_train": 22},
    # Few-shot path (leave-one-out cross-conformal on the 6-pair held-out)
    {"name": "agentive",     "data_dir": "data/few_shot/agentive",    "mode": "loo"},
    {"name": "superlative",  "data_dir": "data/few_shot/superlative", "mode": "loo"},
    {"name": "young",        "data_dir": "data/few_shot/young",       "mode": "loo"},
]


ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


# ---------------------------------------------------------------------------
# Data IO  (shared helpers — copied from 0.5.1 for self-contained script)
# ---------------------------------------------------------------------------

def _strip(s: str) -> str:
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
    def encode(words: list[str]) -> torch.Tensor:
        inputs = tokenizer(
            words, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**inputs)
        last_hidden = out.last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.float()
    return encode


def train_operator(
    encode_fn, train_pairs, *, dim, device, seed, epochs=2000, lr=1e-3,
) -> ConceptOperator:
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
# Per-concept evaluation
# ---------------------------------------------------------------------------

def evaluate_concept_split(
    *, encode_fn, dim: int, device: str, seed: int,
    train_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    pool: list[str],
    n_op_train: int,
    alphas: tuple[float, ...],
) -> dict:
    """Full-data path: split train into operator-train + calibration."""
    rng = random.Random(seed)
    idx = list(range(len(train_pairs)))
    rng.shuffle(idx)
    op_train = [train_pairs[i] for i in idx[:n_op_train]]
    calib    = [train_pairs[i] for i in idx[n_op_train:]]
    if len(calib) < 2:
        raise ValueError(
            f"Calib set too small ({len(calib)}). Lower n_op_train or use 'loo' mode."
        )

    op = train_operator(
        encode_fn, op_train, dim=dim, device=device,
        seed=seed,
    )

    per_alpha: dict[float, dict] = {}
    for a in alphas:
        cal = ConformalOperatorCalibrator(alpha=a)
        cal.fit(op, encode_fn, calib)
        ev = cal.evaluate(op, encode_fn, test_pairs, pool)
        per_alpha[a] = ev
    return {
        "mode": "split",
        "n_op_train": len(op_train),
        "n_calib":    len(calib),
        "n_test":     len(test_pairs),
        "per_alpha":  per_alpha,
    }


def evaluate_concept_loo(
    *, encode_fn, dim: int, device: str, seed: int,
    train_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    pool: list[str],
    alphas: tuple[float, ...],
) -> dict:
    """Few-shot path: train on full train; LOO on held-out for calibration+test."""
    op = train_operator(
        encode_fn, train_pairs, dim=dim, device=device,
        seed=seed,
    )

    # Per α, accumulate in_set flags across LOO folds.
    per_alpha: dict[float, dict] = {a: {"in_set_flags": [], "set_sizes": []} for a in alphas}
    for i in range(len(test_pairs)):
        calib_i = [test_pairs[j] for j in range(len(test_pairs)) if j != i]
        test_i  = [test_pairs[i]]
        for a in alphas:
            cal = ConformalOperatorCalibrator(alpha=a)
            cal.fit(op, encode_fn, calib_i)
            ev = cal.evaluate(op, encode_fn, test_i, pool)
            per_alpha[a]["in_set_flags"].extend(ev["in_set_flags"])
            per_alpha[a]["set_sizes"].extend(ev["set_sizes"])

    # Build evaluate-style dicts.
    out_per_alpha: dict[float, dict] = {}
    for a in alphas:
        flags = per_alpha[a]["in_set_flags"]
        sizes = per_alpha[a]["set_sizes"]
        out_per_alpha[a] = {
            "alpha": a,
            "nominal_coverage": 1.0 - a,
            "empirical_coverage": float(np.mean(flags)) if flags else float("nan"),
            "n_test": len(flags),
            "n_calib": len(test_pairs) - 1,    # always 5 in LOO with held-out=6
            "in_set_flags": flags,
            "set_sizes": sizes,
            "mean_set_size": float(np.mean(sizes)) if sizes else float("nan"),
            "median_set_size": float(np.median(sizes)) if sizes else float("nan"),
            "max_set_size": int(np.max(sizes)) if sizes else 0,
        }
    return {
        "mode": "loo",
        "n_op_train": len(train_pairs),
        "n_calib":    len(test_pairs) - 1,
        "n_test":     len(test_pairs),
        "per_alpha":  out_per_alpha,
    }


def per_concept_ece(per_alpha: dict[float, dict]) -> float:
    return float(np.mean([
        abs(per_alpha[a]["empirical_coverage"] - per_alpha[a]["nominal_coverage"])
        for a in per_alpha
    ]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="gte-base", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="results/stage0_5/all_concepts_conformal.json")
    parser.add_argument("--ece-gate", type=float, default=0.05,
                        help="Pooled-ECE acceptance threshold.")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print(f"Task 0.5.2 — calibrate all 7 concepts; pool ECE")
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    # ---- Load encoder once, reuse across concepts ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Per-concept evaluation ----
    per_concept: dict[str, dict] = {}
    for c in CONCEPTS:
        name = c["name"]
        data_dir = Path(c["data_dir"])
        train = read_pairs(data_dir / "text_pairs_train.tsv")
        test = read_pairs(data_dir / "text_pairs_held_out.tsv")
        pool = read_pool(data_dir / "candidate_pool.txt")
        print(f"\n--- {name} ({c['mode']}) — train={len(train)}, "
              f"holdout={len(test)}, pool={len(pool)} ---")

        if c["mode"] == "split":
            res = evaluate_concept_split(
                encode_fn=encode, dim=enc_cfg["dim"], device=args.device,
                seed=args.seed,
                train_pairs=train, test_pairs=test, pool=pool,
                n_op_train=c["n_op_train"], alphas=ALPHAS,
            )
        elif c["mode"] == "loo":
            res = evaluate_concept_loo(
                encode_fn=encode, dim=enc_cfg["dim"], device=args.device,
                seed=args.seed,
                train_pairs=train, test_pairs=test, pool=pool,
                alphas=ALPHAS,
            )
        else:
            raise ValueError(f"unknown mode {c['mode']}")

        ece = per_concept_ece(res["per_alpha"])
        res["per_concept_ece"] = ece
        per_concept[name] = res

        print(f"  n_op_train={res['n_op_train']}  n_calib={res['n_calib']}  "
              f"n_test={res['n_test']}  per-concept ECE={ece:.4f}")
        for a in ALPHAS:
            r = res["per_alpha"][a]
            print(f"    α={a:.2f}  emp={r['empirical_coverage']:.3f}  "
                  f"(nom {r['nominal_coverage']:.3f})  |set|≈{r['mean_set_size']:.2f}  "
                  f"n_trials={r['n_test']}")

    # ---- Pool across concepts ----
    pooled: dict[float, dict] = {}
    for a in ALPHAS:
        all_flags: list[int] = []
        all_sizes: list[int] = []
        for name in per_concept:
            all_flags.extend(per_concept[name]["per_alpha"][a]["in_set_flags"])
            all_sizes.extend(per_concept[name]["per_alpha"][a]["set_sizes"])
        pooled[a] = {
            "alpha": a,
            "nominal_coverage": 1.0 - a,
            "empirical_coverage": float(np.mean(all_flags)),
            "n_trials": len(all_flags),
            "mean_set_size": float(np.mean(all_sizes)),
            "median_set_size": float(np.median(all_sizes)),
            "max_set_size": int(np.max(all_sizes)),
        }
    pooled_ece = float(np.mean([
        abs(pooled[a]["empirical_coverage"] - pooled[a]["nominal_coverage"])
        for a in ALPHAS
    ]))

    print("\n" + "=" * 88)
    print("POOLED COVERAGE CURVE (across all 7 concepts)")
    print("=" * 88)
    print(f"  {'alpha':>6}  {'nominal':>8}  {'empirical':>10}  {'mean|set|':>10}  {'n_trials':>9}")
    for a in ALPHAS:
        r = pooled[a]
        print(f"  {a:>6.2f}  {r['nominal_coverage']:>8.3f}  "
              f"{r['empirical_coverage']:>10.3f}  {r['mean_set_size']:>10.2f}  "
              f"{r['n_trials']:>9}")

    print(f"\nPooled ECE: {pooled_ece:.4f}")
    print(f"Pooled-ECE granularity floor (1/n_pooled): "
          f"{1.0 / max(pooled[ALPHAS[0]]['n_trials'], 1):.4f}")

    # ---- Acceptance gate ----
    pooled_pass = pooled_ece < args.ece_gate
    print("\n" + "=" * 88)
    print("ACCEPTANCE CHECK (Task 0.5.2)")
    print("=" * 88)
    print(f"  Pooled-ECE gate (< {args.ece_gate}): "
          f"{pooled_ece:.4f} → {'PASS' if pooled_pass else 'FAIL'}")
    print(f"\n  Per-concept ECE (informational, not gated):")
    for name, res in per_concept.items():
        print(f"    {name:<14s}  {res['per_concept_ece']:.4f}")
    overall = "PASS" if pooled_pass else "FAIL"
    print(f"\n→ Task 0.5.2 ({args.encoder}): {overall}")

    # ---- Save JSON ----
    def _serialize_per_alpha(pa: dict) -> dict:
        return {
            str(a): {k: v for k, v in pa[a].items() if k not in ("in_set_flags", "set_sizes")}
            for a in pa
        }

    payload = {
        "task": "0.5.2",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "seed": args.seed,
        "alphas": list(ALPHAS),
        "per_concept": {
            name: {
                "mode": res["mode"],
                "n_op_train": res["n_op_train"],
                "n_calib":    res["n_calib"],
                "n_test":     res["n_test"],
                "per_alpha":  _serialize_per_alpha(res["per_alpha"]),
                "per_concept_ece": res["per_concept_ece"],
            }
            for name, res in per_concept.items()
        },
        "pooled": {str(a): pooled[a] for a in ALPHAS},
        "pooled_ece": pooled_ece,
        "ece_gate": args.ece_gate,
        "pass": pooled_pass,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if pooled_pass else 1)


if __name__ == "__main__":
    main()
