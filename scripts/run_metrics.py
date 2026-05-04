"""Run the full metric battery against a Stage-1 + Stage-2 checkpoint.

Usage:
    # After both stages have trained:
    python3 scripts/run_metrics.py \\
        --stage1-ckpt checkpoints/stage1/final.pt \\
        --stage2-ckpt checkpoints/stage2_plurality/final.pt \\
        --concept plural \\
        --data-dir data/plurality \\
        --eval-pairs eval/coco_eval_subset.tsv

`eval-pairs` is a TSV: caption<TAB>image_path, ~500 lines from the held-out
COCO split. Used for grounding metrics.

Output: prints all metrics + green/red verdict per plan thresholds.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from selflearnai import SHARED_DIM
from selflearnai.adapters import AdapterBundle
from selflearnai.adapters.train_adapters import Stage1Config, Stage1Trainer
from selflearnai.concepts import ConceptOperator, InverseConceptOperator
from selflearnai.foundations import FrozenCLIP, FrozenGTE, FrozenVJEPA2
from selflearnai.metrics import (
    report_concept,
    report_grounding,
    report_failure_modes,
)


# Plan-locked thresholds, all in one place.
THRESHOLDS = {
    "cross_modal_cosine":   0.5,    # plan 6a
    "min_per_dim_std":      0.3,    # plan 6c
    "anchor_cosine":        0.85,   # plan 6c
    "perturbation_min":     0.05,   # plan 6c (vision-must-respond)
    "coherence":            0.7,    # plan 6b
    "held_out_transfer":    0.7,    # plan 6b
    "inversibility":        0.7,    # plan 6b
    "xmodal_direction":     0.5,    # plan 6a
    "category_gap_max":     0.3,    # plan 6c (memorization detector)
}


def _verdict(name: str, value: float, threshold: float, direction: str = ">") -> str:
    if direction == ">":
        ok = value >= threshold
    else:
        ok = value <= threshold
    flag = "✓" if ok else "✗"
    return f"  {flag} {name:35s} {value:+.3f}  (threshold {direction} {threshold})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--concept", default="plural")
    parser.add_argument("--data-dir", default="data/plurality")
    parser.add_argument("--eval-pairs", required=True,
                        help="TSV of (caption \\t image_path) for grounding tests")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = Stage1Config(device=args.device)
    bundle: AdapterBundle = Stage1Trainer.load_adapters(args.stage1_ckpt, cfg)
    bundle.eval()

    gte = FrozenGTE(device=args.device)
    clip = FrozenCLIP(device=args.device)
    vjepa = FrozenVJEPA2(device=args.device)

    # Load Stage-2 operators
    fwd = ConceptOperator(SHARED_DIM).to(args.device)
    inv = InverseConceptOperator(SHARED_DIM).to(args.device)
    ckpt = torch.load(args.stage2_ckpt, map_location=args.device, weights_only=False)
    fwd.load_state_dict(ckpt["fwd"])
    inv.load_state_dict(ckpt["inv"])
    fwd.eval(); inv.eval()

    # Read eval grounding pairs
    eval_pairs = []
    with open(args.eval_pairs) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cap, path = line.split("\t")[:2]
            eval_pairs.append((cap, Path(path)))
    if not eval_pairs:
        sys.exit("eval-pairs is empty.")

    # Read concept data
    data_dir = Path(args.data_dir)
    def _read(path):
        return [tuple(l.strip().split("\t")[:2]) for l in open(path) if l.strip()]
    train_pairs = _read(data_dir / "text_pairs_train.tsv")
    held_out_pairs = _read(data_dir / "text_pairs_held_out.tsv")
    candidate_pool = [l.strip() for l in open(data_dir / "candidate_pool.txt") if l.strip()]
    image_pairs_tsv = data_dir / "image_pairs.tsv"
    image_pairs = []
    if image_pairs_tsv.exists():
        for l in open(image_pairs_tsv):
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            parts = l.split("\t")
            if len(parts) >= 2:
                image_pairs.append((Path(parts[0]), Path(parts[1])))

    # ---- Grounding metrics ----
    print("\n=== Grounding metrics (alignment quality) ===")
    g = report_grounding(bundle, (gte, clip, vjepa), eval_pairs, args.device)
    print(_verdict("cross_modal_cosine", g["cross_modal_cosine"],
                   THRESHOLDS["cross_modal_cosine"]))
    print(_verdict("retrieval_recall@5", g["retrieval_recall@5"],
                   5 / len(eval_pairs)))
    print(_verdict("visual_perturbation",
                   g["visual_perturbation"],
                   THRESHOLDS["perturbation_min"]))
    for name, stats in g["per_dim_stddev"].items():
        print(_verdict(f"min_std[{name}]", stats["min_std"],
                       THRESHOLDS["min_per_dim_std"]))

    # ---- Concept metrics ----
    print(f"\n=== Concept metrics — {args.concept} ===")
    cr = report_concept(
        name=args.concept,
        fwd=fwd, inv=inv,
        bundle=bundle, gte=gte, vjepa=vjepa,
        train_pairs=train_pairs,
        held_out_pairs=held_out_pairs,
        candidate_pool=candidate_pool,
        image_pairs=image_pairs if image_pairs else None,
    )
    print(_verdict("intra_direction_coherence", cr.coherence,
                   THRESHOLDS["coherence"]))
    print(_verdict("held_out_transfer", cr.held_out_transfer,
                   THRESHOLDS["held_out_transfer"]))
    print(_verdict("inversibility", cr.inversibility,
                   THRESHOLDS["inversibility"]))
    print(_verdict("pure_translation", cr.pure_translation,
                   THRESHOLDS["held_out_transfer"]))   # same threshold as transfer
    if not (cr.cross_modal_cosine != cr.cross_modal_cosine):  # NaN check
        print(_verdict("cross_modal_direction", cr.cross_modal_cosine,
                       THRESHOLDS["xmodal_direction"]))


if __name__ == "__main__":
    main()
