"""Text-only pathway: concept × encoder accuracy matrix.

Implements the second pathway of the two-pathway architecture. For each
text-only concept (no image grounding required), train a fresh single-
head operator on raw encoder outputs (NO Stage 1 alignment), evaluate
held-out generalization. Repeat for each text encoder.

This validates the architectural finding from test_richer_encoder.py:
the 0.500 ceiling on cross-category-preserving concepts was a Stage 1
compression artifact + GTE-base capacity ceiling. Removing both
(skipping Stage 1, using a richer encoder) recovers the architecture's
real capability.

Output: a concept × encoder matrix of held-out accuracy + JSON dump
for downstream analysis.

Encoders sweep (~2.6GB download on first run):
  - thenlper/gte-base       (768-dim, current Stage 1 baseline)
  - intfloat/e5-large-v2    (1024-dim, ~335M params)

Concepts swept (any concept with text_pairs_train + text_pairs_held_out
+ candidate_pool files):
  - plural, past_tense, comparative   (full-data concepts)
  - agentive, superlative, young      (few-shot concepts, N=3)
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


ENCODERS = [
    {"name": "gte-base",     "model": "thenlper/gte-base",     "dim":  768},
    {"name": "e5-large-v2",  "model": "intfloat/e5-large-v2",  "dim": 1024},
]


CONCEPTS = [
    {"name": "plural",       "data_dir": "data/plurality"},
    {"name": "past_tense",   "data_dir": "data/past_tense"},
    {"name": "comparative",  "data_dir": "data/comparative"},
    {"name": "agentive",     "data_dir": "data/few_shot/agentive"},
    {"name": "superlative",  "data_dir": "data/few_shot/superlative"},
    {"name": "young",        "data_dir": "data/few_shot/young"},
]


def _strip(s):
    idx = s.find("#")
    return (s[:idx] if idx >= 0 else s).strip()


def read_pairs(path: Path):
    out = []
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


def read_pool(path: Path):
    out, seen = [], set()
    with open(path) as f:
        for raw in f:
            line = _strip(raw.strip())
            if line and line not in seen:
                out.append(line); seen.add(line)
    return out


@torch.no_grad()
def encode_words(model, tokenizer, words, device, max_length=64):
    inputs = tokenizer(
        words, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = model(**inputs)
    last_hidden = out.last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.float()


class Operator(nn.Module):
    """Same shape as ConceptOperator's forward direction. Native-dim;
    no Stage 1 adapter in the picture."""

    def __init__(self, dim: int, mlp_hidden: int = 192):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dim) * 0.02)
        self.alpha = nn.Parameter(torch.ones(1))
        self.residual = nn.Sequential(
            nn.Linear(2 * dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, z):
        v_b = self.v.expand_as(z)
        delta = self.alpha * self.v + self.residual(torch.cat([z, v_b], dim=-1))
        return z + delta


def train_eval(z_train_src, z_train_tgt, z_held_src, z_pool, pool_words,
               tgt_held, dim, device, epochs=2000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    op = Operator(dim).to(device)
    opt = torch.optim.AdamW(op.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(op(z_train_src), z_train_tgt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        z_pred = op(z_held_src)
        z_pred_n = F.normalize(z_pred, dim=-1)
        z_pool_n = F.normalize(z_pool, dim=-1)
        sims = z_pred_n @ z_pool_n.T
        best = sims.argmax(dim=-1).tolist()
        preds = [pool_words[i] for i in best]
    correct = sum(p == t for p, t in zip(preds, tgt_held))
    return correct / len(tgt_held), preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--out", default="results/text_only_concepts.json")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    # Load all concepts.
    print("Loading concept data:")
    concept_data = {}
    for c in CONCEPTS:
        d = Path(c["data_dir"])
        if not d.exists():
            print(f"  ⚠ skipping {c['name']}: {d} not found")
            continue
        train_pairs = read_pairs(d / "text_pairs_train.tsv")
        held_pairs  = read_pairs(d / "text_pairs_held_out.tsv")
        pool        = read_pool(d / "candidate_pool.txt")
        concept_data[c["name"]] = {
            "train": train_pairs, "held": held_pairs, "pool": pool,
        }
        print(f"  {c['name']:14s}: {len(train_pairs)} train, "
              f"{len(held_pairs)} held-out, {len(pool)} pool")

    # Run encoder × concept matrix.
    results = {}
    for enc in ENCODERS:
        print(f"\nLoading {enc['name']} ({enc['model']}) ...")
        tok = AutoTokenizer.from_pretrained(enc["model"])
        mdl = AutoModel.from_pretrained(enc["model"]).to(args.device).eval()
        for p in mdl.parameters():
            p.requires_grad_(False)

        results[enc["name"]] = {}
        for c_name, data in concept_data.items():
            train_pairs = data["train"]
            held_pairs = data["held"]
            pool = data["pool"]

            z_train_src = encode_words(
                mdl, tok, [p[0] for p in train_pairs], args.device,
            )
            z_train_tgt = encode_words(
                mdl, tok, [p[1] for p in train_pairs], args.device,
            )
            z_held_src = encode_words(
                mdl, tok, [p[0] for p in held_pairs], args.device,
            )
            z_pool = encode_words(mdl, tok, pool, args.device)

            accs, last_preds = [], None
            for seed in range(args.seeds):
                acc, preds = train_eval(
                    z_train_src, z_train_tgt, z_held_src, z_pool, pool,
                    [p[1] for p in held_pairs], dim=enc["dim"],
                    device=args.device, seed=seed, epochs=args.epochs,
                )
                accs.append(acc); last_preds = preds
            mean = sum(accs) / len(accs)
            results[enc["name"]][c_name] = {
                "mean": mean, "accs": accs, "last_preds": last_preds,
            }
            accs_str = "[" + ", ".join(f"{a:.2f}" for a in accs) + "]"
            print(f"  {enc['name']:14s} × {c_name:14s}  acc={mean:.3f}  {accs_str}")

        del mdl, tok
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- Combined matrix ----
    print()
    print("=" * 80)
    print("CONCEPT × ENCODER ACCURACY MATRIX (text-only pathway, no Stage 1)")
    print("=" * 80)
    header = f"{'concept':<14}"
    for enc in ENCODERS:
        header += f"  {enc['name']:>14}"
    print(header)
    print("-" * len(header))
    for c_name in concept_data:
        row = f"{c_name:<14}"
        for enc in ENCODERS:
            r = results[enc["name"]][c_name]
            row += f"  {r['mean']:>14.3f}"
        print(row)

    # ---- Best encoder per concept ----
    print()
    print("=" * 80)
    print("BEST ENCODER PER CONCEPT (text-only pathway)")
    print("=" * 80)
    for c_name in concept_data:
        best_enc = max(
            ENCODERS, key=lambda e: results[e["name"]][c_name]["mean"],
        )
        best_acc = results[best_enc["name"]][c_name]["mean"]
        gte_acc = results.get("gte-base", {}).get(c_name, {}).get("mean", 0.0)
        delta = best_acc - gte_acc
        print(f"  {c_name:<14}  →  {best_enc['name']:<14}  "
              f"acc={best_acc:.3f}  (Δ vs gte={delta:+.3f})")

    # ---- Save JSON ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")


if __name__ == "__main__":
    main()
