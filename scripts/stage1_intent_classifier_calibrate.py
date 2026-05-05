"""Task 1.5 — conformal calibration on the Tier-2 intent classifier.

Wraps the trained classifier (Task 1.4) in a calibrated coverage-set
predictor with finite-sample guarantee. Replaces softmax-argmax + a
fixed confidence threshold with calibrated prediction sets at chosen
α, where:
  - a non-empty set with one element  → confident prediction;
  - a set with multiple elements      → ambiguous, ask for help / show
                                         alternatives;
  - an empty set                      → honest "I don't know" (refuse).

Protocol:
  1. Re-train the classifier from scratch (same configuration as
     Task 1.4) so we have a fresh model to calibrate.
  2. Encode the canonical + paraphrase eval rows (66 total) using the
     same encoder. These are NOT in the classifier's training data.
  3. Stratified 50/50 split: half of the eval rows become the
     calibration set, the other half becomes the test set. Every
     concept (and 'unknown') is represented in BOTH halves.
  4. Fit ClassificationConformalCalibrator on the calibration half.
  5. Evaluate on the test half across α ∈ {0.05, .., 0.30}; report
     coverage curve + ECE.

Acceptance gate (Task 1.5):
  - Empirical coverage at α=0.10 within max(0.04, 1/n_test) of 0.90.
    With n_test ≈ 33 the gate widens to [0.86, 0.94] effectively.
  - ECE across the α grid below 0.05 (granularity floor 1/33 ≈ 0.030).

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
from selflearnai.uncertainty import (
    ClassificationConformalCalibrator,
    classification_coverage_curve,
)

# Reuse training corpus + encoder helpers from Task 1.4.
from scripts.stage1_intent_classifier_train import (
    ENCODERS,
    CONCEPTS,
    UNKNOWN,
    ALL_CLASSES,
    build_training_corpus,
    make_encode_fn,
    train_classifier,
)
from scripts.stage1_intent_grammar_smoke import read_eval_rows


# ---------------------------------------------------------------------------
# Stratified split of the eval rows into calibration + test halves.
# Every concept (and 'unknown') gets ~50/50; deterministic via seed.
# ---------------------------------------------------------------------------

def stratified_eval_split(
    rows: list[tuple[str, str, str, str]],
    seed: int,
    mapping: IntentClassMapping,
) -> tuple[list[tuple], list[tuple]]:
    """Returns (calibration_rows, test_rows)."""
    by_class: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        question, expected_concept, expected_source, status = row
        if status == "parseable":
            by_class[expected_concept].append(row)
        elif status == "refuse":
            by_class[mapping.unknown_label].append(row)
        else:
            raise ValueError(f"unexpected status: {status}")

    rng = np.random.default_rng(seed)
    calib: list[tuple] = []
    test: list[tuple] = []
    for cls, members in by_class.items():
        idx = list(range(len(members)))
        rng.shuffle(idx)
        half = len(members) // 2
        calib.extend(members[i] for i in idx[:half])
        test.extend(members[i] for i in idx[half:])
    return calib, test


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

    # ---- Encoder + classifier ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

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
    print(f"  classifier params: {classifier.num_parameters():,}")

    # ---- Eval rows + stratified split ----
    canonical_rows = read_eval_rows(Path(args.canonical))
    paraphrase_rows = read_eval_rows(Path(args.paraphrase))
    all_rows = list(canonical_rows) + list(paraphrase_rows)
    print(f"\nEval rows: canonical={len(canonical_rows)}, "
          f"paraphrase={len(paraphrase_rows)}, total={len(all_rows)}")

    calib_rows, test_rows = stratified_eval_split(
        all_rows, seed=args.seed, mapping=mapping,
    )
    print(f"Stratified 50/50 split (seed={args.seed}): "
          f"calibration={len(calib_rows)}, test={len(test_rows)}")

    # Per-class balance
    def _per_class(rows):
        out = defaultdict(int)
        for r in rows:
            cls = r[1] if r[3] == "parseable" else mapping.unknown_label
            out[cls] += 1
        return dict(out)

    print(f"  calib by class: {_per_class(calib_rows)}")
    print(f"  test  by class: {_per_class(test_rows)}")

    # ---- Probabilities ----
    calib_probs, calib_labels, _ = rows_to_arrays(
        calib_rows, classifier, encode, mapping,
    )
    test_probs, test_labels, test_questions = rows_to_arrays(
        test_rows, classifier, encode, mapping,
    )

    # ---- Calibration curve ----
    curve = classification_coverage_curve(
        train_probs=calib_probs, train_labels=calib_labels,
        test_probs=test_probs, test_labels=test_labels,
        alphas=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    )

    print("\n" + "=" * 70)
    print("COVERAGE CURVE  (calibrated on disjoint half of eval rows)")
    print("=" * 70)
    print(f"  {'α':>5}  {'nominal':>8}  {'empirical':>10}  {'mean|set|':>10}  {'empty':>6}  {'q_hat':>7}")
    for r in curve["per_alpha"]:
        print(
            f"  {r['alpha']:>5.2f}  {r['nominal_coverage']:>8.3f}  "
            f"{r['empirical_coverage']:>10.3f}  {r['mean_set_size']:>10.2f}  "
            f"{r['n_empty_sets']:>6d}  {r['q_hat']:>7.4f}"
        )
    print(f"\nECE: {curve['ece']:.4f}  (granularity floor 1/n_test = {curve['ece_floor']:.4f})")

    # ---- Per-row predictions at the main α (for diagnostics) ----
    main_cal = ClassificationConformalCalibrator(alpha=args.alpha_main)
    main_cal.fit(calib_probs, calib_labels)
    main_eval = main_cal.evaluate(test_probs, test_labels)
    sets = main_cal.predict_set_indices(test_probs)
    print(f"\nAt α={args.alpha_main}:  q_hat={main_cal.q_hat:.4f}  "
          f"empirical={main_eval['empirical_coverage']:.3f}  "
          f"|set|={main_eval['mean_set_size']:.2f}  "
          f"empty={main_eval['n_empty_sets']}/{main_eval['n_test']}")

    # ---- Acceptance gate ----
    n_test = main_eval["n_test"]
    floor = 1.0 / n_test if n_test else float("inf")
    cov_low = main_eval["nominal_coverage"] - max(0.04, floor)
    cov_high = min(1.0, main_eval["nominal_coverage"] + max(0.04, floor))
    cov_pass = cov_low <= main_eval["empirical_coverage"] <= cov_high
    ece_pass = curve["ece"] < args.ece_gate

    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK (Task 1.5)")
    print("=" * 70)
    print(f"  Coverage at α={args.alpha_main}: empirical={main_eval['empirical_coverage']:.3f} "
          f"∈ [{cov_low:.3f}, {cov_high:.3f}]  → "
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
        "n_calib": len(calib_rows),
        "n_test": len(test_rows),
        "main_alpha": {
            k: v for k, v in main_eval.items()
            if k not in ("in_set_flags", "set_sizes")
        },
        "coverage_curve": {
            "alphas": curve["alphas"],
            "ece": curve["ece"],
            "ece_floor": curve["ece_floor"],
            "per_alpha": [
                {k: v for k, v in r.items() if k not in ("in_set_flags", "set_sizes")}
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
