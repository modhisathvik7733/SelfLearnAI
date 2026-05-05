"""Task 1.5 — conformal calibration on the Tier-2 intent classifier.

Wraps the trained classifier (Task 1.4) in a calibrated coverage-set
predictor with finite-sample guarantee. Replaces softmax-argmax + a
fixed confidence threshold with calibrated prediction sets at chosen
α, where:
  - a non-empty set with one element  → confident prediction;
  - a set with multiple elements      → ambiguous, ask for help / show
                                         alternatives;
  - an empty set                      → honest "I don't know" (refuse).

Protocol (final form after the synthesized-holdout attempt produced
under-coverage from distribution shift between synthesized scores and
real-eval scores):

  1. Synthesize the training corpus (Task 1.4 generator) and train the
     classifier on the FULL corpus.
  2. Encode the canonical + paraphrase eval rows (66 total). These are
     not in training data.
  3. Run LEAVE-ONE-OUT CROSS-CONFORMAL on the eval rows: for each row
     i, calibrate on the other 65 and evaluate on row i alone.
     Aggregate over all 66 folds.
  4. Reports per-alpha empirical coverage, set sizes, empty-set count,
     median q_hat across folds, and ECE.

Why LOO cross-conformal: calibration data MUST come from the same
distribution as test for the coverage guarantee to hold. The eval
rows are the only items we have at the eval distribution. LOO uses
each row as a test once with the other (n-1) as calibration, which
is the standard small-sample-but-exchangeable recipe.

Acceptance gate (Task 1.5):
  - Empirical coverage at α=0.10 ≥ (1−α) − 0.04 = 0.86. Over-coverage
    (>0.94) is FINE — conformal-conservative is by-design; only
    UNDER-coverage breaks the guarantee.
  - ECE across the α grid below 0.05.

This calibrator is what the router (Task 1.6) uses to gate Tier-2's
output instead of a hard softmax threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from selflearnai.intent import (
    IntentClassifier,
    IntentClassMapping,
)
from selflearnai.uncertainty import loo_classification_coverage_curve

# Reuse training corpus + encoder helpers from Task 1.4.
from scripts.stage1_intent_classifier_train import (
    ENCODERS,
    UNKNOWN,
    ALL_CLASSES,
    build_training_corpus,
    make_encode_fn,
    train_classifier,
)
from scripts.stage1_intent_grammar_smoke import read_eval_rows


# Note: an earlier draft of Task 1.5 used a stratified hold-out from the
# SYNTHESIZED training corpus as the calibration set. That approach
# produced under-coverage (the synthesized scores cluster much closer
# to zero than the real-eval scores; q_hat ended up too tight for test).
# The fix is the LOO cross-conformal protocol used below: keep
# calibration distribution = test distribution by drawing both from the
# same set of eval rows, with leave-one-out folds.


# ---------------------------------------------------------------------------
# Encode rows into (probs, labels)
# ---------------------------------------------------------------------------

def rows_to_arrays(
    rows: list[tuple[str, str, str, str]],
    classifier: IntentClassifier,
    encode_fn,
    mapping: IntentClassMapping,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Encode questions, run classifier, return (probs, labels, questions)."""
    questions = [r[0] for r in rows]
    z = encode_fn(questions)
    with torch.no_grad():
        names, probs = classifier.predict_batch(z)
    probs = probs.cpu().numpy()
    labels = []
    for question, expected_concept, expected_source, status in rows:
        if status == "parseable":
            labels.append(mapping.index(expected_concept))
        else:
            labels.append(mapping.unknown_index)
    return probs, np.array(labels, dtype=int), questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="gte-base", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--canonical", default="data/intent_eval/canonical.tsv")
    parser.add_argument("--paraphrase", default="data/intent_eval/paraphrase.tsv")
    parser.add_argument("--out", default="results/stage1/intent_classifier_conformal.json")
    parser.add_argument("--alpha-main", type=float, default=0.10)
    parser.add_argument("--ece-gate", type=float, default=0.05)
    args = parser.parse_args()

    print("Task 1.5 — Tier-2 conformal calibration")
    print("=" * 70)
    enc_cfg = ENCODERS[args.encoder]
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    mapping = IntentClassMapping(classes=ALL_CLASSES, unknown_label=UNKNOWN)

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Train classifier on the FULL synthesized training corpus ----
    print("Synthesizing training corpus + training classifier ...")
    train_q, train_y = build_training_corpus()
    classifier = train_classifier(
        train_questions=train_q, train_labels=train_y,
        encode_fn=encode, mapping=mapping,
        encoder_dim=enc_cfg["dim"], hidden_dim=args.hidden_dim,
        device=args.device,
        epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
    )
    print(f"  classifier params: {classifier.num_parameters():,}  "
          f"(trained on {len(train_q)} synthesized questions)")

    # ---- Eval rows: canonical + paraphrase combined ----
    canonical_rows = read_eval_rows(Path(args.canonical))
    paraphrase_rows = read_eval_rows(Path(args.paraphrase))
    eval_rows = list(canonical_rows) + list(paraphrase_rows)
    print(f"\nEval rows: canonical={len(canonical_rows)}, "
          f"paraphrase={len(paraphrase_rows)}, total={len(eval_rows)}")

    def _per_class_eval(rows):
        out = defaultdict(int)
        for r in rows:
            cls = r[1] if r[3] == "parseable" else mapping.unknown_label
            out[cls] += 1
        return dict(out)

    print(f"  by class: {_per_class_eval(eval_rows)}")

    # ---- Encode eval rows + score with classifier ----
    eval_probs, eval_labels, _ = rows_to_arrays(
        eval_rows, classifier, encode, mapping,
    )

    # ---- LOO cross-conformal coverage curve ----
    # Each fold uses 65 rows for calibration, 1 row as test. Aggregating
    # 66 single-row trials gives a smooth empirical-coverage estimate
    # while keeping calibration distribution = test distribution.
    print("\nRunning LOO cross-conformal across α ∈ {0.05, ..., 0.30} "
          f"({len(eval_rows)} folds × 6 α values = {len(eval_rows) * 6} fits) ...")
    curve = loo_classification_coverage_curve(
        probs=eval_probs, labels=eval_labels,
        alphas=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    )

    print("\n" + "=" * 78)
    print(f"COVERAGE CURVE  (LOO cross-conformal, n_calib={curve['n_calib']}, n_test={curve['n_test']})")
    print("=" * 78)
    print(f"  {'α':>5}  {'nominal':>8}  {'empirical':>10}  {'mean|set|':>10}  {'empty':>6}  {'med q_hat':>10}")
    for r in curve["per_alpha"]:
        print(
            f"  {r['alpha']:>5.2f}  {r['nominal_coverage']:>8.3f}  "
            f"{r['empirical_coverage']:>10.3f}  {r['mean_set_size']:>10.2f}  "
            f"{r['n_empty_sets']:>6d}  {r['median_q_hat']:>10.4f}"
        )
    print(f"\nECE: {curve['ece']:.4f}  (granularity floor 1/n_test = {curve['ece_floor']:.4f})")

    # ---- Pick the row at the main α for the acceptance check ----
    main_row = next(
        (r for r in curve["per_alpha"] if abs(r["alpha"] - args.alpha_main) < 1e-9),
        None,
    )
    if main_row is None:
        raise SystemExit(f"alpha_main={args.alpha_main} not in curve")

    print(f"\nAt α={args.alpha_main}: empirical={main_row['empirical_coverage']:.3f}  "
          f"|set|={main_row['mean_set_size']:.2f}  "
          f"empty={main_row['n_empty_sets']}/{main_row['n_test']}  "
          f"median q_hat={main_row['median_q_hat']:.4f}")

    # ---- Acceptance gate (lower-bound coverage; over-coverage is fine) ----
    n_test = main_row["n_test"]
    floor = 1.0 / n_test if n_test else float("inf")
    cov_low = main_row["nominal_coverage"] - max(0.04, floor)
    cov_pass = main_row["empirical_coverage"] >= cov_low
    ece_pass = curve["ece"] < args.ece_gate

    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.5)")
    print("=" * 78)
    print(f"  Coverage at α={args.alpha_main}: empirical={main_row['empirical_coverage']:.3f} "
          f"≥ {cov_low:.3f}  (lower-bound; over-coverage is fine)  → "
          f"{'PASS' if cov_pass else 'FAIL'}")
    print(f"  ECE across α grid: {curve['ece']:.4f} < {args.ece_gate:.2f}  → "
          f"{'PASS' if ece_pass else 'FAIL'}")
    overall = cov_pass and ece_pass
    print(f"\n→ Task 1.5: {'PASS' if overall else 'FAIL'}")

    # ---- Save JSON ----
    payload = {
        "task": "1.5",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "seed": args.seed,
        "alpha_main": args.alpha_main,
        "ece_gate": args.ece_gate,
        "n_calib_per_fold": curve["n_calib"],
        "n_test": curve["n_test"],
        "main_alpha_row": {
            k: v for k, v in main_row.items()
            if k not in ("set_sizes",)
        },
        "coverage_curve": {
            "alphas": curve["alphas"],
            "ece": curve["ece"],
            "ece_floor": curve["ece_floor"],
            "per_alpha": [
                {k: v for k, v in r.items() if k not in ("set_sizes",)}
                for r in curve["per_alpha"]
            ],
        },
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
