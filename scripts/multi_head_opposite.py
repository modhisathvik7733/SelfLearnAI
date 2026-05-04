"""Multi-head sweep on the OPPOSITE / antonym concept.

The single-head ConceptOperator scores ~0% on antonyms — multi-axial concepts
where the same operator must move {big→small, hot→cold, true→false, ...}
across DISJOINT semantic axes. A single direction-vector cannot span axes
that point in unrelated directions in encoder space.

Hypothesis (Option 3): K direction-vectors + a router lets each head
specialize on one axis (size, temperature, truth, emotion, …). The router
sends each input to the right head.

Two sweep dimensions:

  1. K ∈ {1, 2, 3, 4}          — number of heads.
  2. residual ∈ {shared, per-head}
       shared    — one residual MLP across all K heads
                  (`MultiHeadConceptOperator`).
       per-head  — K independent residual MLPs
                  (`MultiHeadConceptOperatorPerHead`). Lets each head's
                  non-linearity specialize too, not just its direction.

We run on the RAW encoder (no Stage 1) — the text-only pathway. Stage 1
compression hurts text-only concepts (validated in
run_text_only_concepts.py), so we keep the latent space as-is.

Default encoder: GTE-base (768-dim). E5-large-v2 (1024-dim) is selectable
to test whether richer encoder + multi-head together break the floor.

Reads:
  data/opposite/text_pairs_train.tsv     (30 antonym pairs, 5+ axes)
  data/opposite/text_pairs_held_out.tsv  (6 held-out antonym pairs)
  data/opposite/candidate_pool.txt       (targets + distractors)

Outputs printed table:
  arch                  params  acc(seed1, seed2, seed3)  mean
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F


ENCODERS = {
    "gte-base":    {"model": "thenlper/gte-base",    "dim":  768},
    "e5-large-v2": {"model": "intfloat/e5-large-v2", "dim": 1024},
}


def _strip(s: str) -> str:
    """Strip inline TSV comments (everything after '#')."""
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
def encode_words(model, tokenizer, words, device, max_length: int = 64):
    inputs = tokenizer(
        words, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = model(**inputs)
    last_hidden = out.last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.float()


# ---------------------------------------------------------------------------
# Operator architectures (native-dim — no Stage-1 adapter).
# Mirror selflearnai.concepts.operator but parameterized on dim so we can run
# on raw 768-dim or 1024-dim encoder outputs.
# ---------------------------------------------------------------------------

class SingleHead(nn.Module):
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


class MultiHeadShared(nn.Module):
    """K directions + router + ONE shared residual MLP."""

    def __init__(self, dim: int, K: int, mlp_hidden: int = 192,
                 router_hidden: int = 64):
        super().__init__()
        self.K = K
        self.v = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.alpha = nn.Parameter(torch.ones(K))
        self.router = nn.Sequential(
            nn.Linear(dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, K),
        )
        self.residual = nn.Sequential(
            nn.Linear(2 * dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, z):
        weights = torch.softmax(self.router(z), dim=-1)        # (..., K)
        head_dirs = self.v * self.alpha.unsqueeze(-1)           # (K, D)
        weighted_v = weights @ head_dirs                        # (..., D)
        delta = weighted_v + self.residual(
            torch.cat([z, weighted_v], dim=-1)
        )
        return z + delta


class MultiHeadPerHead(nn.Module):
    """K directions + router + K INDEPENDENT residual MLPs."""

    def __init__(self, dim: int, K: int, mlp_hidden: int = 192,
                 router_hidden: int = 64):
        super().__init__()
        self.K = K
        self.v = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.alpha = nn.Parameter(torch.ones(K))
        self.router = nn.Sequential(
            nn.Linear(dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, K),
        )
        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, dim),
            )
            for _ in range(K)
        ])

    def forward(self, z):
        weights = torch.softmax(self.router(z), dim=-1)        # (..., K)
        deltas = []
        for k in range(self.K):
            v_k = self.v[k].expand_as(z)
            d_k = self.alpha[k] * self.v[k] + self.residuals[k](
                torch.cat([z, v_k], dim=-1)
            )
            deltas.append(d_k)
        deltas = torch.stack(deltas, dim=-2)                    # (..., K, D)
        delta = (weights.unsqueeze(-1) * deltas).sum(dim=-2)
        return z + delta


def build_op(arch: str, dim: int, mlp_hidden: int = 192) -> nn.Module:
    if arch == "single_head":
        return SingleHead(dim, mlp_hidden=mlp_hidden)
    if arch.startswith("shared_K"):
        K = int(arch.split("K")[1])
        return MultiHeadShared(dim, K=K, mlp_hidden=mlp_hidden)
    if arch.startswith("per_head_K"):
        K = int(arch.split("K")[1])
        return MultiHeadPerHead(dim, K=K, mlp_hidden=mlp_hidden)
    raise ValueError(f"Unknown arch {arch}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------

def train_eval(z_train_src, z_train_tgt, z_held_src, z_pool, pool_words,
               tgt_held, *, arch: str, dim: int, device, seed: int,
               epochs: int, lr: float, mlp_hidden: int):
    torch.manual_seed(seed)
    op = build_op(arch, dim, mlp_hidden=mlp_hidden).to(device)
    opt = torch.optim.AdamW(op.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(op(z_train_src), z_train_tgt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        z_pred = op(z_held_src)
        pred_n = F.normalize(z_pred, dim=-1)
        pool_n = F.normalize(z_pool, dim=-1)
        sims = pred_n @ pool_n.T
        best = sims.argmax(dim=-1).tolist()
        preds = [pool_words[i] for i in best]
    correct = sum(p == t for p, t in zip(preds, tgt_held))
    return correct / len(tgt_held), preds, op


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/opposite")
    parser.add_argument("--encoder", default="gte-base",
                        choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlp-hidden", type=int, default=192)
    parser.add_argument("--ks", type=str, default="2,3,4",
                        help="Comma-separated K values to sweep")
    parser.add_argument("--show-routing", action="store_true",
                        help="Print router head assignments per held-out source")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    enc = ENCODERS[args.encoder]
    print(f"Encoder: {args.encoder} ({enc['model']}, dim={enc['dim']})")

    data_dir = Path(args.data_dir)
    train_pairs = read_pairs(data_dir / "text_pairs_train.tsv")
    held_pairs  = read_pairs(data_dir / "text_pairs_held_out.tsv")
    pool        = read_pool(data_dir / "candidate_pool.txt")
    print(f"Data: {len(train_pairs)} train, {len(held_pairs)} held-out, "
          f"{len(pool)} pool")

    print(f"\nLoading {enc['model']} ...")
    tok = AutoTokenizer.from_pretrained(enc["model"])
    mdl = AutoModel.from_pretrained(enc["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)

    print("Encoding train / held-out / pool ...")
    z_train_src = encode_words(mdl, tok, [p[0] for p in train_pairs], args.device)
    z_train_tgt = encode_words(mdl, tok, [p[1] for p in train_pairs], args.device)
    z_held_src  = encode_words(mdl, tok, [p[0] for p in held_pairs],  args.device)
    z_pool      = encode_words(mdl, tok, pool, args.device)
    tgt_held    = [p[1] for p in held_pairs]

    # Build arch list.
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    archs = ["single_head"]
    for K in ks:
        archs.append(f"shared_K{K}")
        archs.append(f"per_head_K{K}")

    print(f"\nSweeping {len(archs)} architectures × {args.seeds} seeds:\n")
    print("=" * 88)
    print(f"{'arch':<18}  {'params':>9}  {'mean acc':>9}  {'per-seed':<22}  {'preds':<20}")
    print("=" * 88)

    rows = []
    last_ops: dict[str, nn.Module] = {}
    for arch in archs:
        accs, last_preds, last_op = [], None, None
        for seed in range(args.seeds):
            acc, preds, op = train_eval(
                z_train_src, z_train_tgt, z_held_src, z_pool, pool, tgt_held,
                arch=arch, dim=enc["dim"], device=args.device,
                seed=seed, epochs=args.epochs, lr=args.lr,
                mlp_hidden=args.mlp_hidden,
            )
            accs.append(acc); last_preds = preds; last_op = op
        mean = sum(accs) / len(accs)
        rows.append({"arch": arch, "params": n_params(last_op),
                     "mean": mean, "accs": accs, "last_preds": last_preds})
        last_ops[arch] = last_op
        accs_str = "[" + ",".join(f"{a:.2f}" for a in accs) + "]"
        preds_str = ",".join(last_preds[:3]) + "..."
        print(f"{arch:<18}  {n_params(last_op):>9,}  {mean:>9.3f}  "
              f"{accs_str:<22}  {preds_str:<20}")

    # ---- per-item predictions for best arch ----
    best = max(rows, key=lambda r: r["mean"])
    print("\n" + "=" * 88)
    print(f"DETAIL — best arch: {best['arch']} (mean acc {best['mean']:.3f})")
    print("=" * 88)
    for (src, tgt), pred in zip(held_pairs, best["last_preds"]):
        mark = "✓" if pred == tgt else "✗"
        print(f"  {mark} {src:>8s} → {tgt:<8s}  predicted: {pred}")

    # ---- routing diagnostic ----
    if args.show_routing:
        # Best multi-head arch (shared or per-head).
        mh = max(
            (r for r in rows if not r["arch"].startswith("single")),
            key=lambda r: r["mean"],
        )
        op = last_ops[mh["arch"]]
        print("\n" + "=" * 88)
        print(f"ROUTING — head assignments for held-out sources ({mh['arch']})")
        print("=" * 88)
        with torch.no_grad():
            weights = torch.softmax(op.router(z_held_src), dim=-1)
        for (src, tgt), w in zip(held_pairs, weights):
            ws = "  ".join(f"h{k}={w[k].item():.2f}" for k in range(w.shape[0]))
            print(f"  {src:>8s} → {tgt:<8s}   {ws}")

    # ---- comparison ----
    base   = next(r for r in rows if r["arch"] == "single_head")
    sh_best = max((r for r in rows if r["arch"].startswith("shared")),
                  key=lambda r: r["mean"], default=None)
    ph_best = max((r for r in rows if r["arch"].startswith("per_head")),
                  key=lambda r: r["mean"], default=None)

    print("\n" + "=" * 88)
    print("INTERPRETATION")
    print("=" * 88)
    print(f"  single_head (baseline):          acc = {base['mean']:.3f}  "
          f"({base['params']:,} params)")
    if sh_best is not None:
        print(f"  shared-residual best ({sh_best['arch']}):   "
              f"acc = {sh_best['mean']:.3f}  ({sh_best['params']:,} params)")
    if ph_best is not None:
        print(f"  per-head-residual best ({ph_best['arch']}): "
              f"acc = {ph_best['mean']:.3f}  ({ph_best['params']:,} params)")
    print()
    if sh_best is not None and ph_best is not None:
        d_routing = sh_best["mean"] - base["mean"]
        d_specialize = ph_best["mean"] - sh_best["mean"]
        print(f"  Δ from adding routing  (single → shared):  {d_routing:+.3f}")
        print(f"  Δ from per-head MLPs  (shared → per-head): {d_specialize:+.3f}")
        print()
        if ph_best["mean"] - base["mean"] > 0.20:
            print("  → Multi-head BREAKS the multi-axial limit. Routing + specialization")
            print("    let the operator span disjoint antonym axes.")
        elif sh_best["mean"] - base["mean"] > 0.20:
            print("  → Routing alone is enough. Per-head residuals add no extra signal —")
            print("    the K direction-vectors do all the work.")
        else:
            print("  → Multi-head does NOT clear the floor. The bottleneck is in the")
            print(f"    encoder geometry itself ({args.encoder}). Try richer encoder.")


if __name__ == "__main__":
    main()
