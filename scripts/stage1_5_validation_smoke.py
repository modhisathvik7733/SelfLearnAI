"""Task 1.5.3 — three-criterion candidate validation gate smoke.

Validates the validation gate works in BOTH directions: admits a real
candidate and rejects two known-bad candidates with the right failure
modes. Without exercising both directions, the gate could trivially
pass everything (false-PASS bug) or reject everything (false-FAIL bug).

Setup:

  Concept under test: PLURAL (44 train pairs in repo, plenty for a
  proper held-out split). Treated as MASKED for this smoke — we
  pretend the system doesn't yet have a plural operator and is asking
  whether a candidate it just trained should be admitted to the
  registry.

  Baseline operator library: the 6 known concepts EXCEPT plural.
  Trained at the standard 2000 epochs each.

  Held-out task slice: 8 plural pairs the candidate never saw, used
  for both criterion 1 (cos→truth) and criterion 3 (planner-utility).

Three test candidates:

  CANDIDATE A — REAL plural operator
    Trained on the first 30 plural train pairs at full strength
    (2000 epochs). Should pass all three criteria.

  CANDIDATE B — IDENTITY operator
    Constructed via make_identity_candidate(): alpha=0, residual MLP
    zero'd. forward(z) = z. Should fail criterion 2 (non-triviality).
    Will also fail cos-truth on plural pairs (output = source ≠
    target), but the GATE is "any criterion failure rejects" — and
    we want non-triviality to be the *first* failure surfaced.

  CANDIDATE C — BAD operator
    Trained on plural sources paired with SHUFFLED targets (so it
    learns nothing useful). Should fail criterion 1 (cos→truth).
    Non-trivial because the residual MLP did learn something — just
    not the right thing.

Acceptance gates (Task 1.5.3, all HARD):

  - CANDIDATE A passes all three criteria (passes_cos_truth=True,
    passes_non_triviality=True, passes_planner_utility=True,
    passes=True).
  - CANDIDATE B is rejected with passes_non_triviality=False.
  - CANDIDATE C is rejected with passes_cos_truth=False.

Run on the GPU box (~5–8 min: encoder + train 6 baseline ops + train
2 candidate ops + 3 validation runs):
  python scripts/stage1_5_validation_smoke.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from selflearnai.discovery import (
    make_identity_candidate,
    validate_candidate,
)

from scripts.stage1_planner_beam_smoke import (
    ENCODERS, make_encode_fn, read_pairs, train_operator,
)


# Baseline operator library: all 7 known concepts EXCEPT plural.
BASELINE_CONCEPTS_DATA: list[tuple[str, str]] = [
    ("past_tense",   "data/past_tense"),
    ("comparative",  "data/comparative"),
    ("superlative",  "data/few_shot/superlative"),
    ("opposite",     "data/opposite_v2"),
    ("agentive",     "data/few_shot/agentive"),
    ("young",        "data/few_shot/young"),
]

PLURAL_DATA_DIR = "data/plurality"
N_TRAIN_FOR_CANDIDATE = 30  # remaining pairs become held-out
N_HOLDOUT_PAIRS = 8         # used for cos→truth + planner tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-epochs", type=int, default=2000)
    parser.add_argument("--candidate-epochs", type=int, default=2000)
    parser.add_argument("--cos-truth-min", type=float, default=0.85)
    parser.add_argument("--non-triviality-max-cos", type=float, default=0.99)
    parser.add_argument("--planner-cos-threshold", type=float, default=0.80)
    parser.add_argument("--planner-beam-width", type=int, default=4)
    parser.add_argument("--planner-max-depth", type=int, default=3)
    parser.add_argument("--out", default="results/stage1_5/validation_smoke.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Task 1.5.3 — three-criterion validation gate smoke")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Baseline library (no plural): "
          f"{[c for c, _ in BASELINE_CONCEPTS_DATA]}")
    print(f"Concept under validation: plural (treated as new candidate)")

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Build held-out slice from plural data ----
    plural_pairs = read_pairs(Path(PLURAL_DATA_DIR) / "text_pairs_train.tsv")
    if len(plural_pairs) < N_TRAIN_FOR_CANDIDATE + N_HOLDOUT_PAIRS:
        raise SystemExit(
            f"plural data: {len(plural_pairs)} pairs < required "
            f"{N_TRAIN_FOR_CANDIDATE + N_HOLDOUT_PAIRS}"
        )
    candidate_train_pairs = plural_pairs[:N_TRAIN_FOR_CANDIDATE]
    holdout_pairs = plural_pairs[N_TRAIN_FOR_CANDIDATE:N_TRAIN_FOR_CANDIDATE + N_HOLDOUT_PAIRS]
    print(f"\nPlural pairs: {len(candidate_train_pairs)} train + "
          f"{len(holdout_pairs)} held-out")

    print("Encoding train + held-out plural pairs ...")
    z_src_train = encode([p[0] for p in candidate_train_pairs])
    z_tgt_train = encode([p[1] for p in candidate_train_pairs])
    z_src_holdout = encode([p[0] for p in holdout_pairs])
    z_tgt_holdout = encode([p[1] for p in holdout_pairs])

    # ---- Train baseline operator library (the 6 non-plural concepts) ----
    print(f"\nTraining {len(BASELINE_CONCEPTS_DATA)} baseline operators "
          f"@ {args.baseline_epochs} epochs each ...")
    baseline_operators: dict[str, object] = {}
    for concept, ddir in BASELINE_CONCEPTS_DATA:
        train_pairs = read_pairs(Path(ddir) / "text_pairs_train.tsv")
        op = train_operator(
            encode, train_pairs,
            dim=enc_cfg["dim"], device=args.device,
            seed=args.seed, epochs=args.baseline_epochs,
        )
        baseline_operators[concept] = op
        print(f"  trained {concept}  ({len(train_pairs)} pairs)")

    # ---- Build planner-utility task slice ----
    # Each task = (psi_start = z_src_holdout[i], psi_goal = z_tgt_holdout[i]).
    # Baseline planner shouldn't reach these goals (no plural op);
    # augmented planner (with valid plural candidate) should.
    planner_holdout_tasks = [
        (z_src_holdout[i], z_tgt_holdout[i]) for i in range(N_HOLDOUT_PAIRS)
    ]
    print(f"Planner-utility task slice: {len(planner_holdout_tasks)} held-out plural pairs")

    # ---- Train CANDIDATE A: real plural operator ----
    print(f"\n[A] Training REAL plural candidate @ {args.candidate_epochs} epochs ...")
    cand_real = train_operator(
        encode, candidate_train_pairs,
        dim=enc_cfg["dim"], device=args.device,
        seed=args.seed, epochs=args.candidate_epochs,
    )

    # ---- Train CANDIDATE C: bad operator (shuffled targets) ----
    # Match seed-controlled determinism: shuffle targets independently.
    print(f"[C] Training BAD plural candidate (shuffled targets) ...")
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 999)
    perm = torch.randperm(len(candidate_train_pairs), generator=rng).tolist()
    shuffled_pairs = [
        (candidate_train_pairs[i][0], candidate_train_pairs[perm[i]][1])
        for i in range(len(candidate_train_pairs))
    ]
    # Sanity: shuffled meaningfully differs from original on >50% of indices.
    n_unchanged = sum(
        1 for i in range(len(candidate_train_pairs))
        if shuffled_pairs[i][1] == candidate_train_pairs[i][1]
    )
    print(f"  shuffled-target pairs: {len(candidate_train_pairs) - n_unchanged}/"
          f"{len(candidate_train_pairs)} altered")
    cand_bad = train_operator(
        encode, shuffled_pairs,
        dim=enc_cfg["dim"], device=args.device,
        seed=args.seed, epochs=args.candidate_epochs,
    )

    # ---- Build CANDIDATE B: identity ----
    print("[B] Constructing IDENTITY candidate (alpha=0, residual MLP zero'd) ...")
    cand_identity = make_identity_candidate(dim=enc_cfg["dim"], device=args.device)

    # ---- Validate each candidate ----
    common_kwargs = dict(
        z_src_holdout=z_src_holdout,
        z_tgt_holdout=z_tgt_holdout,
        z_src_for_triviality=z_src_train,
        baseline_operators=baseline_operators,
        planner_holdout_tasks=planner_holdout_tasks,
        cos_truth_min=args.cos_truth_min,
        non_triviality_max_cos=args.non_triviality_max_cos,
        planner_cos_threshold=args.planner_cos_threshold,
        planner_beam_width=args.planner_beam_width,
        planner_max_depth=args.planner_max_depth,
    )

    print("\n" + "=" * 78)
    print("PER-CANDIDATE VALIDATION")
    print("=" * 78)

    test_cases = [
        ("real_plural",      cand_real,     "expected: PASS all 3"),
        ("identity",         cand_identity, "expected: FAIL non-triviality"),
        ("bad_shuffled_tgt", cand_bad,      "expected: FAIL cos→truth"),
    ]

    results: dict[str, dict] = {}
    for name, op, expectation in test_cases:
        print(f"\n— Candidate: {name}  ({expectation})")
        r = validate_candidate(op, name, **common_kwargs)
        c1 = "✓" if r.passes_cos_truth else "✗"
        c2 = "✓" if r.passes_non_triviality else "✗"
        c3 = "✓" if r.passes_planner_utility else "✗"
        overall = "PASS" if r.passes else "FAIL"
        print(
            f"   {c1} generalization     mean={r.cos_truth_mean:.3f}  "
            f"min={r.cos_truth_min:.3f}  threshold≥{r.cos_truth_threshold:.2f}"
        )
        print(
            f"   {c2} non-triviality     max(cos→input)={r.cos_to_input_max:.3f}  "
            f"threshold<{r.non_triviality_threshold:.2f}"
        )
        print(
            f"   {c3} planner-utility    baseline={r.n_baseline_solved}  "
            f"augmented={r.n_augmented_solved}  Δ={r.utility_improvement}  "
            f"(of {r.n_planner_tasks} tasks, planner cos≥{r.planner_cos_threshold:.2f})"
        )
        print(f"   → overall: {overall}")
        if r.failing_criteria:
            for fc in r.failing_criteria:
                print(f"     · failing: {fc}")
        results[name] = {
            "passes_cos_truth": r.passes_cos_truth,
            "passes_non_triviality": r.passes_non_triviality,
            "passes_planner_utility": r.passes_planner_utility,
            "passes": r.passes,
            "cos_truth_mean": r.cos_truth_mean,
            "cos_truth_min": r.cos_truth_min,
            "cos_to_input_max": r.cos_to_input_max,
            "n_baseline_solved": r.n_baseline_solved,
            "n_augmented_solved": r.n_augmented_solved,
            "utility_improvement": r.utility_improvement,
            "failing_criteria": r.failing_criteria,
        }

    # ---- Acceptance ----
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.5.3)")
    print("=" * 78)
    real_pass = results["real_plural"]["passes"]
    identity_correct_reject = (
        not results["identity"]["passes"]
        and not results["identity"]["passes_non_triviality"]
    )
    bad_correct_reject = (
        not results["bad_shuffled_tgt"]["passes"]
        and not results["bad_shuffled_tgt"]["passes_cos_truth"]
    )
    print(
        f"  Real plural candidate passes 3/3:   "
        f"{'PASS' if real_pass else 'FAIL'}"
    )
    print(
        f"  Identity candidate rejected (non-triviality): "
        f"{'PASS' if identity_correct_reject else 'FAIL'}"
    )
    print(
        f"  Bad candidate rejected (cos→truth):  "
        f"{'PASS' if bad_correct_reject else 'FAIL'}"
    )
    overall = real_pass and identity_correct_reject and bad_correct_reject
    print(f"\n→ Task 1.5.3: {'PASS' if overall else 'FAIL'}")

    # ---- Save JSON ----
    payload = {
        "task": "1.5.3",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "baseline_concepts": [c for c, _ in BASELINE_CONCEPTS_DATA],
        "n_candidate_train": N_TRAIN_FOR_CANDIDATE,
        "n_holdout": N_HOLDOUT_PAIRS,
        "thresholds": {
            "cos_truth_min": args.cos_truth_min,
            "non_triviality_max_cos": args.non_triviality_max_cos,
            "planner_cos_threshold": args.planner_cos_threshold,
        },
        "candidates": results,
        "acceptance": {
            "real_pass": real_pass,
            "identity_correct_reject": identity_correct_reject,
            "bad_correct_reject": bad_correct_reject,
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
