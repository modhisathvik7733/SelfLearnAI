"""Task 1.5 — conformal calibration on the Tier-2 intent classifier.

Wraps the trained classifier (Task 1.4) in a calibrated coverage-set
predictor with finite-sample guarantee. Replaces softmax-argmax + a
fixed confidence threshold with calibrated prediction sets at chosen
α, where:
  - a non-empty set with one element  → confident prediction;
  - a set with multiple elements      → ambiguous, ask for help / show
                                         alternatives;
  - an empty set                      → honest "I don't know" (refuse).

Protocol (revised after first run showed coarse q_hat from a too-small
calibration set):

  1. Synthesize the training corpus (Task 1.4 generator).
  2. Stratified hold-out: reserve ~100 of the 1041 synthesized
     questions for CALIBRATION. The classifier is trained on the
     remaining ~941; the held-out 100 are never seen during training,
     preserving exchangeability with the eval set as long as the
     training distribution covers the eval phrasings (it does — see
     Task 1.4 paraphrase recall).
  3. Encode the canonical + paraphrase eval rows (66 total) — these
     are also held out from training. Used as the TEST set.
  4. Fit ClassificationConformalCalibrator on the synthesized
     calibration half (n=100).
  5. Evaluate on the full 66-row eval (n=66) across α ∈ {0.05, ..,
     0.30}; report coverage curve + ECE.

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
from selflearnai.uncertainty import (
    ClassificationConformalCalibrator,
    classification_coverage_curve,
)

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


# ---------------------------------------------------------------------------
# Stratified hold-out of the SYNTHESIZED training corpus for calibration.
# This gives us a calibration set that is (a) larger than splitting the
# eval rows (~100 vs ~33), (b) drawn from the same distribution the
# classifier was NOT trained on, (c) deterministic via seed.
# ---------------------------------------------------------------------------

def stratified_train_holdout(
    questions: list[str],
    labels: list[str],
    n_per_class: int,
    seed: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Hold out `n_per_class` items per class from the training corpus.

    Returns (train_q, train_y, holdout_q, holdout_y).
    Held-out items are NEVER passed to the classifier trainer; they
    become the calibration set for the conformal predictor.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)

    holdout_idx: set[int] = set()
    for lab, idxs in by_class.items():
        if len(idxs) <= n_per_class:
            raise ValueError(
                f"class {lab!r} has only {len(idxs)} items; can't hold out {n_per_class}"
            )
        rng.shuffle(idxs)
        holdout_idx.update(idxs[:n_per_class])

    train_q: list[str] = []
    train_y: list[str] = []
    holdout_q: list[str] = []
    holdout_y: list[str] = []
    for i, (q, lab) in enumerate(zip(questions, labels)):
        if i in holdout_idx:
            holdout_q.append(q)
            holdout_y.append(lab)
        else:
            train_q.append(q)
            train_y.append(lab)
    return train_q, train_y, holdout_q, holdout_y


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
    parser.add_argument(
        "--n-calib-per-class", type=int, default=12,
        help="How many synthesized training questions per class to hold out "
             "for calibration. Default 12; with 8 classes ⇒ ~96 calibration "
             "items, well above the 31 we had under the eval-split design.",
    )
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

    # ---- Build corpus + hold out for calibration ----
    print("Synthesizing training corpus ...")
    full_q, full_y = build_training_corpus()
    train_q, train_y, holdout_q, holdout_y = stratified_train_holdout(
        full_q, full_y,
        n_per_class=args.n_calib_per_class,
        seed=args.seed,
    )
    print(
        f"  full corpus: {len(full_q)}; "
        f"trainer: {len(train_q)} ({len(train_q)/len(full_q):.1%}); "
        f"calibration hold-out: {len(holdout_q)}"
    )

    # ---- Train classifier on the non-held-out portion ----
    print("Training classifier on non-held-out training portion ...")
    classifier = train_classifier(
        train_questions=train_q, train_labels=train_y,
        encode_fn=encode, mapping=mapping,
        encoder_dim=enc_cfg["dim"], hidden_dim=args.hidden_dim,
        device=args.device,
        epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
    )
    print(f"  classifier params: {classifier.num_parameters():,}")

    # ---- Eval rows are the FULL canonical + paraphrase as test ----
    canonical_rows = read_eval_rows(Path(args.canonical))
    paraphrase_rows = read_eval_rows(Path(args.paraphrase))
    test_rows = list(canonical_rows) + list(paraphrase_rows)
    print(f"\nEval (TEST set): canonical={len(canonical_rows)}, "
          f"paraphrase={len(paraphrase_rows)}, total={len(test_rows)}")

    # Per-class balance for visibility
    def _per_class_holdout(ys):
        return {c: ys.count(c) for c in mapping.classes}

    def _per_class_eval(rows):
        out = defaultdict(int)
        for r in rows:
            cls = r[1] if r[3] == "parseable" else mapping.unknown_label
            out[cls] += 1
        return dict(out)

    print(f"  calib hold-out by class: {_per_class_holdout(holdout_y)}")
    print(f"  test by class:           {_per_class_eval(test_rows)}")

    # ---- Encode + score calibration hold-out ----
    z_calib = encode(holdout_q)
    with torch.no_grad():
        _, calib_probs_t = classifier.predict_batch(z_calib)
    calib_probs = calib_probs_t.cpu().numpy()
    calib_labels = np.array(
        [mapping.index(l) for l in holdout_y], dtype=int,
    )

    # ---- Encode + score test eval rows ----
    test_probs, test_labels, _ = rows_to_arrays(
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
    print(f"\nAt α={args.alpha_main}:  q_hat={main_cal.q_hat:.4f}  "
          f"empirical={main_eval['empirical_coverage']:.3f}  "
          f"|set|={main_eval['mean_set_size']:.2f}  "
          f"empty={main_eval['n_empty_sets']}/{main_eval['n_test']}")

    # ---- Acceptance gate ----
    # Lower-bound coverage gate: in conformal prediction, OVER-coverage
    # (empirical > 1 - α) is fine; only UNDER-coverage breaks the
    # validity guarantee. Allow a small slack below nominal to absorb
    # discretization noise from the finite test set.
    n_test = main_eval["n_test"]
    floor = 1.0 / n_test if n_test else float("inf")
    cov_low = main_eval["nominal_coverage"] - max(0.04, floor)
    cov_pass = main_eval["empirical_coverage"] >= cov_low
    ece_pass = curve["ece"] < args.ece_gate

    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK (Task 1.5)")
    print("=" * 70)
    print(f"  Coverage at α={args.alpha_main}: empirical={main_eval['empirical_coverage']:.3f} "
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
        "n_calib": len(holdout_q),
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
