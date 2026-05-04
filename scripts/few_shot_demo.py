"""Few-shot concept-learning demo.

Demonstrates the project's central claim: define a brand-new concept with
just 3 training pairs, train the operator, evaluate on held-out items.

Three concepts are tested, ranked by how hard they should be:
  1. AGENTIVE     (write→writer, ...)  — clean morphological + semantic.
  2. SUPERLATIVE  (big→biggest, ...)   — clean morphological + semantic.
  3. YOUNG-ANIMAL (dog→puppy, ...)     — PURE SEMANTIC, no morphology.
                                          Hardest test: shares zero spelling
                                          between source and target.

If all three work with N=3, the architecture has demonstrated genuine
few-shot concept learning at this scale — the strongest single result of
the project.

Usage:
    python3 scripts/few_shot_demo.py \\
        --stage1-ckpt checkpoints/stage1/final.pt \\
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai import SHARED_DIM
from selflearnai.adapters.train_adapters import Stage1Config, Stage1Trainer
from selflearnai.concepts import ConceptOperator, InverseConceptOperator
from selflearnai.foundations import FrozenGTE


CONCEPTS = [
    {
        "name": "agentive",
        "description": "verb → person who does the action (-er / -or)",
        "data_dir": "data/few_shot/agentive",
    },
    {
        "name": "superlative",
        "description": "adjective → maximally-intensified form (-est)",
        "data_dir": "data/few_shot/superlative",
    },
    {
        "name": "young",
        "description": "adult animal → its baby (PURE SEMANTIC, no shared morphology)",
        "data_dir": "data/few_shot/young",
    },
]


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
    seen = set()
    with open(path) as f:
        for raw in f:
            line = _strip_comment(raw.strip())
            if line and line not in seen:
                out.append(line)
                seen.add(line)
    return out


def train_few_shot(
    train_pairs: list[tuple[str, str]],
    bundle, gte, device: str, epochs: int = 1500, lr: float = 1e-3, seed: int = 0,
):
    """Train one ConceptOperator + InverseConceptOperator on the given pairs.
    Returns (fwd, inv, src_emb, tgt_emb)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        z_src = bundle.adapter_t(gte.encode([p[0] for p in train_pairs]))
        z_tgt = bundle.adapter_t(gte.encode([p[1] for p in train_pairs]))
    fwd = ConceptOperator(SHARED_DIM).to(device)
    inv = InverseConceptOperator(SHARED_DIM).to(device)
    opt = torch.optim.AdamW(
        list(fwd.parameters()) + list(inv.parameters()), lr=lr,
    )
    for _ in range(epochs):
        opt.zero_grad()
        l_fwd = F.mse_loss(fwd(z_src), z_tgt)
        l_inv = F.mse_loss(inv(z_tgt), z_src)
        (l_fwd + l_inv).backward()
        opt.step()
    return fwd, inv, z_src, z_tgt


@torch.no_grad()
def evaluate(
    fwd, inv, bundle, gte,
    train_pairs: list[tuple[str, str]],
    held_out_pairs: list[tuple[str, str]],
    candidate_pool: list[str],
    top_k: int = 5,
) -> dict:
    """Apply the operator to held-out sources, retrieve nearest candidates."""
    src_texts = [p[0] for p in held_out_pairs]
    tgt_texts = [p[1] for p in held_out_pairs]
    z_src = bundle.adapter_t(gte.encode(src_texts))
    z_pred = fwd(z_src)
    z_pool = bundle.adapter_t(gte.encode(candidate_pool))
    z_pool_n = F.normalize(z_pool, dim=-1)
    z_pred_n = F.normalize(z_pred, dim=-1)
    sims = z_pred_n @ z_pool_n.T                                       # (Nh, Np)

    rows = []
    correct = 0
    for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts)):
        top_vals, top_idx = sims[i].topk(top_k)
        top_words = [candidate_pool[j] for j in top_idx.tolist()]
        is_correct = top_words[0] == tgt
        correct += int(is_correct)
        rows.append({
            "source": src, "target": tgt,
            "top1": top_words[0],
            "top_k": list(zip(top_words, top_vals.tolist())),
            "correct": is_correct,
        })

    # Sanity: also report transfer using v_const (mean shift, no MLP).
    train_z_src = bundle.adapter_t(gte.encode([p[0] for p in train_pairs]))
    train_z_tgt = bundle.adapter_t(gte.encode([p[1] for p in train_pairs]))
    v_const = (train_z_tgt - train_z_src).mean(dim=0)
    z_pred_lin = z_src + v_const
    z_pred_lin_n = F.normalize(z_pred_lin, dim=-1)
    sims_lin = z_pred_lin_n @ z_pool_n.T
    best_lin = sims_lin.argmax(dim=-1).tolist()
    correct_lin = sum(
        candidate_pool[best_lin[i]] == tgt_texts[i]
        for i in range(len(tgt_texts))
    )
    return {
        "rows": rows,
        "correct": correct,
        "total": len(held_out_pairs),
        "accuracy": correct / len(held_out_pairs),
        "linear_accuracy": correct_lin / len(held_out_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-ckpt", default="checkpoints/stage1/final.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("Loading Stage-1 adapters + GTE ...")
    cfg = Stage1Config(device=args.device)
    bundle = Stage1Trainer.load_adapters(args.stage1_ckpt, cfg)
    bundle.eval()
    gte = FrozenGTE(device=args.device)

    summary = []
    for concept in CONCEPTS:
        data_dir = Path(concept["data_dir"])
        train_pairs = _read_pairs(data_dir / "text_pairs_train.tsv")
        held_out_pairs = _read_pairs(data_dir / "text_pairs_held_out.tsv")
        candidate_pool = _read_pool(data_dir / "candidate_pool.txt")

        print()
        print("=" * 72)
        print(f"CONCEPT: {concept['name'].upper()}")
        print(f"  {concept['description']}")
        print("=" * 72)
        print(f"  Training pairs (N={len(train_pairs)}):")
        for s, t in train_pairs:
            print(f"    {s:>10s}  →  {t}")
        print(f"  Held-out pairs (N={len(held_out_pairs)}):")
        for s, t in held_out_pairs:
            print(f"    {s:>10s}  →  {t}")
        print(f"  Candidate pool size: {len(candidate_pool)}")

        # Train
        fwd, inv, _, _ = train_few_shot(
            train_pairs, bundle, gte, args.device, epochs=args.epochs,
        )

        # Evaluate
        result = evaluate(
            fwd, inv, bundle, gte,
            train_pairs, held_out_pairs, candidate_pool, top_k=args.top_k,
        )

        # Report
        print(f"\n  RESULTS (top-{args.top_k} predictions per held-out source):")
        for row in result["rows"]:
            mark = "✓" if row["correct"] else "✗"
            print(f"    {mark} {row['source']:>10s} → {row['target']:<10s}  "
                  f"top-1: {row['top1']}")
            for j, (word, score) in enumerate(row["top_k"]):
                tgt_marker = " ← target" if word == row["target"] else ""
                print(f"        rank-{j+1} ({score:+.3f})  {word}{tgt_marker}")

        print(f"\n  ACCURACY: {result['correct']}/{result['total']} "
              f"= {result['accuracy']:.3f}  "
              f"(linear baseline {result['linear_accuracy']:.3f})")

        summary.append({
            "concept": concept["name"],
            "n_train": len(train_pairs),
            "accuracy": result["accuracy"],
            "linear_accuracy": result["linear_accuracy"],
        })

    # Final summary
    print()
    print("=" * 72)
    print("FEW-SHOT CONCEPT-LEARNING SUMMARY")
    print("=" * 72)
    print(f"  {'concept':<14}  {'N_train':>7}  {'operator':>10}  {'linear':>8}")
    print(f"  {'-'*14}  {'-'*7}  {'-'*10}  {'-'*8}")
    for s in summary:
        print(f"  {s['concept']:<14}  {s['n_train']:>7}  "
              f"{s['accuracy']:>10.3f}  {s['linear_accuracy']:>8.3f}")
    print()

    n_passing = sum(1 for s in summary if s["accuracy"] >= 0.5)
    print(f"  Concepts at ≥ 0.5 held-out accuracy: {n_passing}/{len(summary)}")
    if n_passing == len(summary):
        print("\n  → Few-shot concept learning DEMONSTRATED at this scale.")
        print("    Three brand-new concepts learned from N=3 examples each,")
        print("    each generalizing to held-out items above chance.")


if __name__ == "__main__":
    main()
