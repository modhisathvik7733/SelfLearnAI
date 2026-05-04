"""Sample-efficiency curve for concept operators.

For each working concept, train operators with progressively fewer training
pairs (N = 3, 5, 8, ...) and measure held_out_transfer at each N. Reveals
each concept's sample complexity at this architectural scale.

Why this matters: the project's central question is whether the architecture
is sample-efficient. This script gives a direct, plottable answer per
concept by sweeping training-set size.

Implementation note: foundations + Stage-1 adapters load ONCE; concept pairs
are pre-encoded once per concept. Each (N, seed) trial is just operator
re-init + ~1k epochs of MSE on cached embeddings → ~1 second per trial.
Total runtime: ~1 minute for 3 concepts × 8 N values × 3 seeds.

Usage:
    python3 scripts/sample_efficiency_curve.py \\
        --stage1-ckpt checkpoints/stage1/final.pt \\
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai import SHARED_DIM
from selflearnai.adapters.train_adapters import Stage1Config, Stage1Trainer
from selflearnai.concepts import ConceptOperator, InverseConceptOperator
from selflearnai.foundations import FrozenGTE


# Default sweep configuration. Override with CLI flags.
CONCEPTS = [
    ("plural",      "data/plurality"),
    ("past_tense",  "data/past_tense"),
    ("comparative", "data/comparative"),
]
N_VALUES = [3, 5, 8, 12, 16, 20, 25, 30]
N_SEEDS = 3


def _strip_comment(s: str) -> str:
    idx = s.find("#")
    return (s[:idx] if idx >= 0 else s).strip()


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    out = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                src = _strip_comment(parts[0])
                tgt = _strip_comment(parts[1])
                if src and tgt:
                    out.append((src, tgt))
    return out


def _read_pool(path: Path) -> list[str]:
    out = []
    with open(path) as f:
        for raw in f:
            line = _strip_comment(raw.strip())
            if line:
                out.append(line)
    return out


@torch.no_grad()
def encode_concept(bundle, gte, data_dir: Path, device: str) -> dict:
    """Pre-encode every pair / candidate ONCE per concept.
    All subsequent (N, seed) trials run on these cached tensors."""
    train_pairs = _read_pairs(data_dir / "text_pairs_train.tsv")
    held_out_pairs = _read_pairs(data_dir / "text_pairs_held_out.tsv")
    candidate_pool = _read_pool(data_dir / "candidate_pool.txt")

    train_src = bundle.adapter_t(gte.encode([p[0] for p in train_pairs]))
    train_tgt = bundle.adapter_t(gte.encode([p[1] for p in train_pairs]))
    held_src = bundle.adapter_t(gte.encode([p[0] for p in held_out_pairs]))
    pool_emb = bundle.adapter_t(gte.encode(candidate_pool))

    return {
        "train_pairs": train_pairs,
        "train_src": train_src,
        "train_tgt": train_tgt,
        "held_out_pairs": held_out_pairs,
        "held_src": held_src,
        "candidate_pool": candidate_pool,
        "pool_emb": pool_emb,
    }


def train_and_evaluate(
    enc: dict, N: int, seed: int, device: str,
    epochs: int = 1000, lr: float = 1e-3,
) -> float:
    """Subsample N training pairs, train a fresh operator, return held_out_transfer."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    n_train = enc["train_src"].shape[0]
    N = min(N, n_train)
    indices = torch.tensor(
        rng.sample(range(n_train), N), dtype=torch.long, device=device,
    )

    src = enc["train_src"].index_select(0, indices)
    tgt = enc["train_tgt"].index_select(0, indices)

    fwd = ConceptOperator(SHARED_DIM).to(device)
    inv = InverseConceptOperator(SHARED_DIM).to(device)
    opt = torch.optim.AdamW(
        list(fwd.parameters()) + list(inv.parameters()), lr=lr,
    )

    for _ in range(epochs):
        opt.zero_grad()
        l_fwd = F.mse_loss(fwd(src), tgt)
        l_inv = F.mse_loss(inv(tgt), src)
        (l_fwd + l_inv).backward()
        opt.step()

    # Evaluate held_out_transfer.
    with torch.no_grad():
        pred = fwd(enc["held_src"])
        pool_n = F.normalize(enc["pool_emb"], dim=-1)
        pred_n = F.normalize(pred, dim=-1)
        sims = pred_n @ pool_n.T                                       # (Nh, Np)
        best = sims.argmax(dim=-1).tolist()
        best_words = [enc["candidate_pool"][i] for i in best]
        targets = [p[1] for p in enc["held_out_pairs"]]
        correct = sum(b == t for b, t in zip(best_words, targets))
        return correct / len(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-ckpt", default="checkpoints/stage1/final.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/sample_efficiency.csv")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    print("Loading Stage-1 adapters + GTE ...")
    cfg = Stage1Config(device=args.device)
    bundle = Stage1Trainer.load_adapters(args.stage1_ckpt, cfg)
    bundle.eval()
    gte = FrozenGTE(device=args.device)

    print("Encoding all concept pairs once ...")
    encs = {}
    for name, data_dir in CONCEPTS:
        encs[name] = encode_concept(bundle, gte, Path(data_dir), args.device)
        print(f"  {name:14s}: {len(encs[name]['train_pairs'])} train pairs, "
              f"{len(encs[name]['held_out_pairs'])} held-out, "
              f"{len(encs[name]['candidate_pool'])} candidates")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSweeping {len(CONCEPTS)} concepts × {len(N_VALUES)} N × {args.seeds} seeds = "
          f"{len(CONCEPTS) * len(N_VALUES) * args.seeds} trials ...\n")

    rows = []
    for name, _ in CONCEPTS:
        enc = encs[name]
        for N in N_VALUES:
            transfers = []
            for seed in range(args.seeds):
                t = train_and_evaluate(
                    enc, N, seed, args.device, epochs=args.epochs,
                )
                transfers.append(t)
            mean = sum(transfers) / len(transfers)
            std = (sum((x - mean) ** 2 for x in transfers) / len(transfers)) ** 0.5
            rows.append({
                "concept": name, "N": N,
                "transfer_mean": mean, "transfer_std": std,
                "transfers": transfers,
            })
            t_str = "[" + ", ".join(f"{t:.2f}" for t in transfers) + "]"
            print(f"  {name:<14} N={N:3d}   transfer = {mean:.3f} ± {std:.3f}   {t_str}")

    # ---- Save CSV ----
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "N", "seed", "transfer"])
        for r in rows:
            for seed, t in enumerate(r["transfers"]):
                writer.writerow([r["concept"], r["N"], seed, t])
    print(f"\n→ saved CSV to {out_path}")

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("SAMPLE-EFFICIENCY CURVE — held_out_transfer (mean over seeds) by N")
    print("=" * 80)
    print(f"{'concept':<14}", end="")
    for N in N_VALUES:
        print(f"  N={N:<3d}", end="")
    print()
    print("-" * 80)
    for name, _ in CONCEPTS:
        print(f"{name:<14}", end="")
        for N in N_VALUES:
            r = next((r for r in rows if r["concept"] == name and r["N"] == N), None)
            cell = f"{r['transfer_mean']:.2f}" if r else "  --"
            print(f"  {cell:<5}", end="")
        print()

    # ---- ASCII plot ----
    print("\n" + "=" * 80)
    print("ASCII PLOT — bars are transfer rate, scale 0.0 to 1.0")
    print("=" * 80)
    bar_width = 16
    for name, _ in CONCEPTS:
        print(f"\n  {name}:")
        for N in N_VALUES:
            r = next((r for r in rows if r["concept"] == name and r["N"] == N), None)
            if r is None:
                continue
            filled = int(round(r["transfer_mean"] * bar_width))
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"    N={N:>3}  {bar}  {r['transfer_mean']:.2f}")

    # ---- Verdict ----
    print("\n" + "=" * 80)
    print("VERDICT — sample complexity per concept")
    print("=" * 80)
    threshold = 0.70
    print(f"  Smallest N where mean held_out_transfer ≥ {threshold}:")
    for name, _ in CONCEPTS:
        first_pass = None
        for N in N_VALUES:
            r = next((r for r in rows if r["concept"] == name and r["N"] == N), None)
            if r and r["transfer_mean"] >= threshold:
                first_pass = N
                break
        if first_pass is None:
            print(f"    {name:<14}  never reaches {threshold} within N≤{N_VALUES[-1]}")
        else:
            print(f"    {name:<14}  N = {first_pass}")

    # ---- Diagnostic for the meta-learning question ----
    print("\n  Reading the curves:")
    print("    • If concepts have similar curves → architecture is concept-agnostic.")
    print("    • If later concepts (comparative) need fewer pairs → meta-learning,")
    print("      i.e. the latent space's accumulated structure helps new concepts.")
    print("    • If earlier concepts (plural) are easiest → expected; semantic-rich")
    print("      content like plurality may have native GTE structure to lean on.")


if __name__ == "__main__":
    main()
