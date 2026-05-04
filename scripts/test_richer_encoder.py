"""Does a richer text encoder break the young-animal 0.50 ceiling?

The 5-architecture sweep showed: single-head, single-head-bigger-MLP,
and multi-head K=2/3/4 all plateau at exactly 0.500 on cross-category-
preserving concepts. Strongly suggests the limit is in the text encoder
(GTE-base), not the operator.

This script tests that hypothesis directly. Skips Stage-1 alignment —
trains a fresh single-head operator on each encoder's RAW outputs.
Same training pairs, same held-out, same no-lures pool, 3 seeds each.

If a richer encoder (E5-large or BGE-large) breaks 0.500, the limit
was GTE-base specifically. If they all stall at 0.500, the limit is
structural across text encoders for species-specific cross-category
mappings — a real architectural finding.

First run downloads ~2.6GB of encoder weights (E5 + BGE). After that
it's ~1 minute per run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F


ENCODERS = [
    {"name": "gte-base",       "model": "thenlper/gte-base",        "dim":  768},
    {"name": "e5-large-v2",    "model": "intfloat/e5-large-v2",     "dim": 1024},
    {"name": "bge-large-v1.5", "model": "BAAI/bge-large-en-v1.5",   "dim": 1024},
]

TRAIN_POOL = [
    ("dog", "puppy"),
    ("cat", "kitten"),
    ("cow", "calf"),
    ("chicken", "chick"),
    ("duck", "duckling"),
    ("goose", "gosling"),
    ("swan", "cygnet"),
    ("frog", "tadpole"),
    ("butterfly", "caterpillar"),
]

HELD_OUT = [
    ("horse", "foal"),
    ("deer", "fawn"),
    ("bear", "cub"),
    ("sheep", "lamb"),
    ("goat", "kid"),
    ("pig", "piglet"),
]

POOL_NO_LURES = [
    "puppy", "kitten", "calf", "foal", "fawn", "cub", "lamb", "kid",
    "piglet", "chick", "duckling", "gosling", "joey", "tadpole",
    "caterpillar", "fledgling", "cygnet", "hatchling", "fry",
    "pup", "nestling", "larva", "embryo", "infant",
]


@torch.no_grad()
def encode_words(model, tokenizer, words, device, max_length=64):
    """Mean-pool encoder outputs over tokens, masked by attention. fp32."""
    inputs = tokenizer(
        words, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = model(**inputs)
    last_hidden = out.last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.float()


class SimpleOperator(nn.Module):
    """Same shape as ConceptOperator (forward direction only). Takes any
    input dim — used to fairly compare encoders without re-running Stage 1."""

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


def train_one_seed(
    z_train_src, z_train_tgt,
    z_held_src, z_pool, pool_words, tgt_held,
    dim: int, device: str, seed: int, epochs: int, lr: float = 1e-3,
):
    torch.manual_seed(seed)
    op = SimpleOperator(dim).to(device)
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
    parser.add_argument(
        "--encoders", nargs="+", default=None,
        help="subset of encoder names to test (e.g. --encoders gte-base e5-large-v2)",
    )
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    selected = ENCODERS if not args.encoders else \
        [e for e in ENCODERS if e["name"] in args.encoders]

    print(f"\nTesting {len(selected)} encoders on young-animal task (no Stage 1)")
    print(f"  Training: {len(TRAIN_POOL)} pairs")
    print(f"  Held-out: {len(HELD_OUT)} pairs")
    print(f"  Pool:     {len(POOL_NO_LURES)} candidates (no adult lures)")

    src_train = [p[0] for p in TRAIN_POOL]
    tgt_train = [p[1] for p in TRAIN_POOL]
    src_held = [p[0] for p in HELD_OUT]
    tgt_held = [p[1] for p in HELD_OUT]

    rows = []
    last_preds: dict[str, list[str]] = {}

    for enc in selected:
        print(f"\nLoading {enc['name']} ({enc['model']}) ...")
        tok = AutoTokenizer.from_pretrained(enc["model"])
        mdl = AutoModel.from_pretrained(enc["model"]).to(args.device).eval()
        for p in mdl.parameters():
            p.requires_grad_(False)

        z_train_src = encode_words(mdl, tok, src_train, args.device)
        z_train_tgt = encode_words(mdl, tok, tgt_train, args.device)
        z_held_src = encode_words(mdl, tok, src_held, args.device)
        z_pool = encode_words(mdl, tok, POOL_NO_LURES, args.device)

        # Diagnostic: how separated are source/target in this encoder's space?
        cos_pairs = F.cosine_similarity(z_train_src, z_train_tgt, dim=-1)
        cos_mean = cos_pairs.mean().item()
        cos_min = cos_pairs.min().item()
        cos_max = cos_pairs.max().item()

        # Sweep seeds.
        accs, last = [], None
        for seed in range(args.seeds):
            acc, preds = train_one_seed(
                z_train_src, z_train_tgt,
                z_held_src, z_pool, POOL_NO_LURES, tgt_held,
                dim=enc["dim"], device=args.device,
                seed=seed, epochs=args.epochs,
            )
            accs.append(acc); last = preds
        mean = sum(accs) / len(accs)
        rows.append({
            "name": enc["name"], "dim": enc["dim"],
            "cos_mean": cos_mean, "cos_min": cos_min, "cos_max": cos_max,
            "mean_acc": mean, "accs": accs,
        })
        last_preds[enc["name"]] = last

        del mdl, tok
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- Summary ----
    print()
    print("=" * 92)
    print(f"{'encoder':<18}  {'dim':>5}  {'cos(src,tgt)':<22}  "
          f"{'mean acc':>10}  {'per-seed':<22}")
    print("=" * 92)
    for r in rows:
        cos_str = f"mean={r['cos_mean']:.2f} [{r['cos_min']:.2f}-{r['cos_max']:.2f}]"
        accs_str = "[" + ", ".join(f"{a:.2f}" for a in r["accs"]) + "]"
        print(f"{r['name']:<18}  {r['dim']:>5}  {cos_str:<22}  "
              f"{r['mean_acc']:>10.3f}  {accs_str:<22}")

    best = max(rows, key=lambda r: r["mean_acc"])
    print(f"\nDETAIL — best encoder: {best['name']} (mean acc {best['mean_acc']:.3f})")
    for (src, tgt), pred in zip(HELD_OUT, last_preds[best["name"]]):
        mark = "✓" if pred == tgt else "✗"
        print(f"  {mark} {src:>10s} → {tgt:<10s}  predicted: {pred}")

    # ---- Interpretation ----
    print()
    print("=" * 92)
    print("INTERPRETATION")
    print("=" * 92)
    gte = next((r for r in rows if r["name"] == "gte-base"), None)
    others = [r for r in rows if r["name"] != "gte-base"]
    if gte and others:
        best_other = max(others, key=lambda r: r["mean_acc"])
        delta = best_other["mean_acc"] - gte["mean_acc"]
        print(f"  GTE-base (raw, no Stage 1):  {gte['mean_acc']:.3f}")
        print(f"  Best richer encoder:         {best_other['mean_acc']:.3f}  "
              f"({best_other['name']})")
        print(f"  Δ from richer encoder:       {delta:+.3f}")
        print()
        if delta > 0.20:
            print("  → Richer encoder BREAKS the 0.50 ceiling. The limit was in")
            print("    GTE-base specifically; cross-category-preserving concepts")
            print("    work with a richer encoder.")
        elif delta > 0.05:
            print("  → Richer encoder helps somewhat but ceiling largely persists.")
            print("    Capacity matters but isn't the full story.")
        else:
            print("  → Richer encoder doesn't break the ceiling. The limit is")
            print("    STRUCTURAL across text encoders: species-specific cross-")
            print("    category mappings need either knowledge-base augmentation")
            print("    or multimodal grounding (visual species identity).")
    elif rows:
        print(f"  Only ran {len(rows)} encoder(s). Need both GTE and a richer")
        print("  encoder to compute the comparison.")


if __name__ == "__main__":
    main()
