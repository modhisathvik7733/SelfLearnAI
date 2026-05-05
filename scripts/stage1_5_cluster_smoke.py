"""Task 1.5.2 — Ψ-shift clustering + per-cluster operator-consistency.

Synthetic 3-masked-concept benchmark for the discovery clustering
substrate. Three concepts (plural, past_tense, comparative — all with
40+ training pairs in repo) are treated as MASKED: their (src, tgt)
pairs are pooled into a single unlabeled stream of ΔΨ residuals, and
the discovery pipeline must:

  1. Cluster the ΔΨ stream and recover the 3 concepts as separable
     groups (silhouette + sweep over k ∈ [2, 6] picks the best K).
  2. For each predicted cluster, train a fresh ConceptOperator on its
     members and verify operator-consistency: cos(op(src), tgt) has
     mean ≥ 0.80 AND var < 0.10 across cluster members. This is
     safeguard #1 (plan §19.9, memory): centroid similarity alone
     produces fake clusters.

Cluster purity is computed against the held-out true labels (which
the discovery code never sees during clustering or operator training).
Per the §19.9 acceptance: ≥ 2/3 true concepts recovered with purity
≥ 0.7 AND ≥ 2/3 clusters pass the operator-consistency check.

Acceptance gates (Task 1.5.2):

  HARD GATES:
    - Best K (silhouette sweep) in {2, 3, 4} (close to true 3).
    - Best silhouette ≥ 0.4.
    - ≥ 2/3 true concepts recovered (purity ≥ 0.7 by majority vote).
    - ≥ 2/3 clusters pass operator-consistency
      (mean_cos ≥ 0.80 AND var_cos < 0.10).

  INFORMATIONAL:
    - 3/3 recovery + all clusters consistent (the strong outcome).

Run on the GPU box:
  python scripts/stage1_5_cluster_smoke.py
  python scripts/stage1_5_cluster_smoke.py --encoder gte-base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from selflearnai.discovery import (
    cluster_with_silhouette_sweep,
    compute_psi_shifts,
    operator_consistency,
    train_quick_operator,
)

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn, read_pairs


# Three concepts with enough train pairs (40+ each) for honest clustering.
# AGENTIVE / SUPERLATIVE / YOUNG only have ~6 pairs total — too sparse.
MASKED_CONCEPTS: list[tuple[str, str, int]] = [
    # (concept_name, data_dir, n_pairs_to_use)
    ("plural",      "data/plurality",   12),
    ("past_tense",  "data/past_tense",  12),
    ("comparative", "data/comparative", 12),
]


def majority_concept(concepts: list[str]) -> tuple[str, float]:
    """Return (most-frequent concept, purity) for a list of concept labels."""
    if not concepts:
        return ("(empty)", 0.0)
    counts: dict[str, int] = {}
    for c in concepts:
        counts[c] = counts.get(c, 0) + 1
    top = max(counts.items(), key=lambda kv: kv[1])
    return (top[0], top[1] / len(concepts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operator-epochs", type=int, default=300)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--n-init", type=int, default=8)
    parser.add_argument("--silhouette-min", type=float, default=0.40)
    parser.add_argument("--purity-min", type=float, default=0.70)
    parser.add_argument("--consistency-mean-min", type=float, default=0.80)
    parser.add_argument("--consistency-var-max", type=float, default=0.10)
    parser.add_argument("--out", default="results/stage1_5/cluster_smoke.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Task 1.5.2 — Ψ-shift clustering + operator-consistency smoke")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Masked concepts (treated as unlabeled): "
          f"{[c for c, _, _ in MASKED_CONCEPTS]}")

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Build the unlabeled ΔΨ stream from masked concepts ----
    # Held-aside: per-pair true labels, used ONLY for purity computation.
    all_pairs: list[tuple[str, str]] = []
    true_labels: list[str] = []
    per_concept_used: dict[str, int] = {}
    for concept, ddir, n_use in MASKED_CONCEPTS:
        pairs = read_pairs(Path(ddir) / "text_pairs_train.tsv")
        if len(pairs) < n_use:
            raise SystemExit(
                f"{concept}: only {len(pairs)} pairs in {ddir}, need {n_use}"
            )
        for p in pairs[:n_use]:
            all_pairs.append(p)
            true_labels.append(concept)
        per_concept_used[concept] = n_use
    n = len(all_pairs)
    print(f"\nUnlabeled stream: {n} pairs ({per_concept_used})")

    print("\nEncoding pairs and computing ΔΨ ...")
    z_src, z_tgt, shifts = compute_psi_shifts(encode, all_pairs)
    print(f"  z_src: {tuple(z_src.shape)}, z_tgt: {tuple(z_tgt.shape)}, "
          f"shifts: {tuple(shifts.shape)}")

    # ---- Cluster ----
    print(f"\nKMeans sweep (k ∈ [{args.k_min}, {args.k_max}], "
          f"n_init={args.n_init}) ...")
    sweep = cluster_with_silhouette_sweep(
        shifts.cpu(),
        k_min=args.k_min,
        k_max=args.k_max,
        n_init=args.n_init,
        seed=args.seed,
    )
    print(f"  per-K silhouette / inertia:")
    for k, m in sweep.per_k.items():
        mark = " *" if k == sweep.best_k else "  "
        print(f"   {mark} k={k}  silhouette={m['silhouette']:+.4f}  "
              f"inertia={m['inertia']:.4f}")
    print(f"  best K = {sweep.best_k} (silhouette={sweep.best_silhouette:+.4f})")
    labels = sweep.best_result.labels.tolist()

    # ---- Cluster purity (against true labels) ----
    print("\n" + "=" * 78)
    print("CLUSTER → CONCEPT MAPPING (majority vote, audit only)")
    print("=" * 78)
    cluster_records: list[dict] = []
    recovered_concepts: set[str] = set()
    n_purity_pass = 0
    for cid in sorted(set(labels)):
        members = [i for i, lab in enumerate(labels) if lab == cid]
        member_concepts = [true_labels[i] for i in members]
        majority, purity = majority_concept(member_concepts)
        purity_ok = purity >= args.purity_min
        if purity_ok:
            n_purity_pass += 1
            recovered_concepts.add(majority)
        # Per-cluster operator-consistency.
        z_src_c = z_src[members]
        z_tgt_c = z_tgt[members]
        op = train_quick_operator(
            z_src_c, z_tgt_c,
            dim=enc_cfg["dim"], device=args.device,
            seed=args.seed, epochs=args.operator_epochs,
        )
        cons = operator_consistency(
            op, z_src_c.to(args.device), z_tgt_c.to(args.device),
            mean_threshold=args.consistency_mean_min,
            var_threshold=args.consistency_var_max,
        )
        purity_mark = "✓" if purity_ok else "✗"
        cons_mark = "✓" if cons.passes else "✗"
        print(
            f"  cluster {cid}: n={len(members)}  "
            f"{purity_mark} majority={majority} purity={purity:.2f}  "
            f"{cons_mark} consistency mean={cons.mean_cos:.3f} "
            f"var={cons.var_cos:.4f} "
            f"(min={cons.min_cos:.3f}, max={cons.max_cos:.3f})"
        )
        cluster_records.append({
            "cluster_id": cid,
            "n_members": len(members),
            "majority_concept": majority,
            "purity": purity,
            "purity_pass": purity_ok,
            "consistency": {
                "mean_cos": cons.mean_cos,
                "var_cos": cons.var_cos,
                "min_cos": cons.min_cos,
                "max_cos": cons.max_cos,
                "passes_mean": cons.passes_mean,
                "passes_var": cons.passes_var,
                "passes": cons.passes,
            },
        })

    n_consistent = sum(1 for r in cluster_records if r["consistency"]["passes"])
    n_recovered = len(recovered_concepts)
    n_true_concepts = len(MASKED_CONCEPTS)
    n_clusters = len(cluster_records)

    # ---- Acceptance ----
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.5.2)")
    print("=" * 78)
    k_in_range = args.k_min <= sweep.best_k <= 4
    sil_pass = sweep.best_silhouette >= args.silhouette_min
    recovery_pass = n_recovered >= 2
    consistency_pass = n_consistent >= 2

    print(
        f"  Best K in [{args.k_min}, 4]: K={sweep.best_k}  "
        f"→ {'PASS' if k_in_range else 'FAIL'}"
    )
    print(
        f"  Silhouette ≥ {args.silhouette_min:.2f}: "
        f"{sweep.best_silhouette:+.4f}  "
        f"→ {'PASS' if sil_pass else 'FAIL'}"
    )
    print(
        f"  Concepts recovered (purity ≥ {args.purity_min:.2f}): "
        f"{n_recovered}/{n_true_concepts} "
        f"({sorted(recovered_concepts)})  "
        f"→ {'PASS' if recovery_pass else 'FAIL'}"
    )
    print(
        f"  Clusters operator-consistent (mean ≥ {args.consistency_mean_min:.2f} "
        f"AND var < {args.consistency_var_max:.2f}): "
        f"{n_consistent}/{n_clusters}  "
        f"→ {'PASS' if consistency_pass else 'FAIL'}"
    )
    overall = k_in_range and sil_pass and recovery_pass and consistency_pass
    print(f"\n→ Task 1.5.2: {'PASS' if overall else 'FAIL'}")

    # ---- Save JSON ----
    payload = {
        "task": "1.5.2",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "masked_concepts": [c for c, _, _ in MASKED_CONCEPTS],
        "n_pairs": n,
        "per_concept_used": per_concept_used,
        "sweep": {
            "best_k": sweep.best_k,
            "best_silhouette": sweep.best_silhouette,
            "per_k": sweep.per_k,
        },
        "thresholds": {
            "silhouette_min": args.silhouette_min,
            "purity_min": args.purity_min,
            "consistency_mean_min": args.consistency_mean_min,
            "consistency_var_max": args.consistency_var_max,
        },
        "clusters": cluster_records,
        "n_recovered_concepts": n_recovered,
        "n_true_concepts": n_true_concepts,
        "n_consistent_clusters": n_consistent,
        "n_clusters": n_clusters,
        "pass": overall,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
