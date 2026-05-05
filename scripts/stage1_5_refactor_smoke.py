"""Task 1.5.4 — e-graph subchain refactor + macro promotion smoke.

Validates the two-gate macro promotion pipeline end-to-end. Two parts:

  PART A — FREQUENCY GATE (synthetic wake buffer)
    Constructs a synthetic success channel with planted recurrences
    and singletons. Verifies that mine_subchain_candidates returns
    exactly the sub-chains crossing frequency_min and rejects
    everything else.

  PART B — UTILITY GATE (real operators + held-out tasks)
    Trains 4 real operators (agentive, plural, past_tense, opposite),
    runs the utility evaluator on held-out (verb → plural_agent)
    tasks (drawn from CHAIN_TRIPLES) at depth=1 against TWO macros:

      - USEFUL macro = (agentive, plural)
        Baseline library = {past_tense, opposite} (does NOT contain
        the macro's component ops). Macro adds genuine new capability —
        baseline cannot reach plural_agent without it. Expected PASS.

      - USELESS macro = (past_tense, opposite)
        Same baseline = {past_tense, opposite}. Both components are
        already in the baseline; the macro chains them in a direction
        irrelevant to verb→plural_agent. Augmented planner gains no
        useful path. Expected FAIL utility.

    Note on the original draft (failed first run): putting `agentive`
    and `plural` BOTH in the baseline let the planner reach cos≈0.896
    at depth=1 with `agentive` alone (painter ≈ painters in E5), so
    the macro's marginal contribution was only +0.003. The fix is
    structural: baseline must NOT contain the macro's components for
    the gate to measure the macro's real value. Same pattern as
    Task 1.5.3 (where "baseline excludes plural" was correct).

Acceptance gates (Task 1.5.4, all HARD):

  Part A:
    - mine returns ('agentive', 'plural') with n_occurrences = 6
      (5 exact + 1 from a planted (agentive, plural, opposite) entry).
    - mine does NOT return ('opposite',), ('past_tense',) — those are
      length-1 (below min_len) AND below frequency_min anyway.
    - mine does NOT return ('plural', 'opposite') — appears once
      (sub-prefix of one chain), below frequency_min.

  Part B (two sub-tests):
    - Useful (agentive, plural) macro at depth=1: mean improvement
      ≥ 0.01 AND ≥ 1 task individually improves by ≥ 0.01.  PASS.
    - Useless (past_tense, opposite) macro at depth=1: rejected
      (improvement < 0.01 or no individual tasks improve).
      Validates the gate fires in BOTH directions.

Run on the GPU box (~3–5 min: encoder + 2 ops trained):
  python scripts/stage1_5_refactor_smoke.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selflearnai.discovery import (
    WakeBuffer,
    evaluate_macro_utility,
    macro_name_from_chain,
    make_macro_op,
    mine_subchain_candidates,
)
from selflearnai.planner import PsiProgram, PsiProgramStep

from scripts.stage1_planner_beam_smoke import (
    CHAIN_TRIPLES, ENCODERS, make_encode_fn, read_pairs, train_operator,
)


# ---------------------------------------------------------------------------
# Part A: synthetic success buffer
# ---------------------------------------------------------------------------

def make_synthetic_program(
    *,
    source: str,
    target: str,
    chain: list[str],
) -> PsiProgram:
    """Tiny PsiProgram fixture — the mining counts only chain content,
    so embeddings are dummy."""
    dim = 4
    psi_initial = [0.0] * dim
    psi_goal = [1.0] * dim
    steps = [
        PsiProgramStep(
            step_index=i,
            op_name=op_name,
            psi_after=[float(i)] * dim,
            cos_to_goal_after=0.5,
            verification={"all_passed": True, "gates": {}},
        )
        for i, op_name in enumerate(chain)
    ]
    return PsiProgram(
        source=source,
        target=target,
        encoder_name="synthetic",
        encoder_dim=dim,
        psi_initial=psi_initial,
        psi_goal=psi_goal,
        chain=chain,
        steps=steps,
        final_top1_word=target,
        final_cos_to_goal=0.95,
        timestamp=PsiProgram.now_timestamp(),
        metadata={"task": "1.5.4", "synthetic": True},
    )


# Planted chains — total 5 + 1 + 1 + 2 = 9 success entries.
SYNTHETIC_SUCCESS_PLAN = [
    # Five exact (agentive, plural) chains — the recurring pattern.
    ("paint", "painters", ["agentive", "plural"]),
    ("drive", "drivers",  ["agentive", "plural"]),
    ("sing",  "singers",  ["agentive", "plural"]),
    ("dance", "dancers",  ["agentive", "plural"]),
    ("run",   "runners",  ["agentive", "plural"]),
    # One length-3 chain that adds 1 to (agentive, plural) and 1 each to
    # (plural, opposite) and (agentive, plural, opposite).
    ("help",  "non-helpers", ["agentive", "plural", "opposite"]),
    # One length-1 chain — should be ignored entirely (below min_len=2).
    ("cold",  "hot",         ["opposite"]),
    # Two more length-1 chains — same reason.
    ("walk",  "walked",      ["past_tense"]),
    ("run",   "ran",         ["past_tense"]),
]


def part_a_frequency_gate(args) -> tuple[bool, dict]:
    print("\n" + "=" * 78)
    print("PART A — FREQUENCY GATE (synthetic wake buffer)")
    print("=" * 78)

    tmp = Path(tempfile.mkdtemp(prefix="refactor_smoke_"))
    print(f"  buffer root: {tmp}")
    buf = WakeBuffer(tmp)
    for source, target, chain in SYNTHETIC_SUCCESS_PLAN:
        buf.append(
            make_synthetic_program(source=source, target=target, chain=chain),
            channel="success",
            routing_reason="synthetic plant for refactor smoke",
            routing_metadata={"chain_len": len(chain)},
        )
    success = buf.read(channel="success")
    print(f"  planted {len(success)} success entries")

    candidates = mine_subchain_candidates(
        success,
        min_chain_length=args.min_chain_length,
        max_chain_length=args.max_chain_length,
        frequency_min=args.frequency_min,
    )
    print(f"  mining (min_len={args.min_chain_length}, "
          f"max_len={args.max_chain_length}, "
          f"frequency_min={args.frequency_min}):")
    for c in candidates:
        print(f"    {c.chain}  n_occurrences={c.n_occurrences}  "
              f"sources={c.sources_seen}")

    # Expected: exactly one candidate, ('agentive', 'plural') with n=6.
    expected_chain = ("agentive", "plural")
    expected_n = 6
    only_one = len(candidates) == 1
    correct_chain = bool(candidates) and candidates[0].chain == expected_chain
    correct_count = bool(candidates) and candidates[0].n_occurrences == expected_n

    # Negative checks: ensure none of the singletons / sub-prefixes leaked.
    leaked = [
        c.chain for c in candidates
        if c.chain in {("opposite",), ("past_tense",), ("plural", "opposite"),
                       ("agentive", "plural", "opposite")}
    ]
    no_leaks = not leaked

    part_a_pass = only_one and correct_chain and correct_count and no_leaks
    print(f"\n  ✓ exactly one candidate           "
          f"{'PASS' if only_one else 'FAIL (got %d)' % len(candidates)}")
    print(f"  ✓ candidate is {expected_chain}    "
          f"{'PASS' if correct_chain else 'FAIL'}")
    print(f"  ✓ n_occurrences = {expected_n}             "
          f"{'PASS' if correct_count else 'FAIL'}")
    print(f"  ✓ no singletons / sub-chains leaked  "
          f"{'PASS' if no_leaks else 'FAIL: %s' % leaked}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    return part_a_pass, {
        "n_planted": len(success),
        "n_candidates": len(candidates),
        "candidates": [
            {"chain": list(c.chain), "n_occurrences": c.n_occurrences,
             "sources_seen": c.sources_seen}
            for c in candidates
        ],
        "leaked_subchains": [list(s) for s in leaked],
        "pass": part_a_pass,
    }


# ---------------------------------------------------------------------------
# Part B: utility gate on real (agentive, plural) macro
# ---------------------------------------------------------------------------

def part_b_utility_gate(args, encode, enc_cfg) -> tuple[bool, dict]:
    print("\n" + "=" * 78)
    print("PART B — UTILITY GATE (useful vs useless macro)")
    print("=" * 78)

    # Train all 4 operators we'll need. The baseline library will use
    # only past_tense + opposite; the macros are built from the full set.
    print(f"  Training 4 base operators @ {args.candidate_epochs} epochs each ...")
    train_specs = [
        ("agentive",   "data/few_shot/agentive"),
        ("plural",     "data/plurality"),
        ("past_tense", "data/past_tense"),
        ("opposite",   "data/opposite_v2"),
    ]
    all_ops: dict[str, object] = {}
    for name, ddir in train_specs:
        pairs = read_pairs(Path(ddir) / "text_pairs_train.tsv")
        op = train_operator(
            encode, pairs,
            dim=enc_cfg["dim"], device=args.device,
            seed=args.seed, epochs=args.candidate_epochs,
        )
        all_ops[name] = op
        print(f"    trained {name} ({len(pairs)} pairs)")

    # Baseline library: ops NOT in the useful macro's chain.
    baseline_operators = {
        "past_tense": all_ops["past_tense"],
        "opposite":   all_ops["opposite"],
    }
    print(f"  Baseline library: {sorted(baseline_operators.keys())} "
          f"(deliberately EXCLUDES agentive + plural so the useful "
          f"macro provides new capability)")

    # Held-out tasks: from CHAIN_TRIPLES, encode (verb → plural_agent).
    # Each is a depth-2 problem (agentive then plural) under the FULL
    # operator set; the baseline (past_tense + opposite) cannot reach
    # plural_agent at any depth using only those ops.
    print(f"  Held-out tasks: {len(CHAIN_TRIPLES)} (verb → plural_agent) pairs")
    holdout_tasks: list[tuple] = []
    for verb, _, plural_agent in CHAIN_TRIPLES:
        z_start = encode([verb]).squeeze(0)
        z_goal = encode([plural_agent]).squeeze(0)
        holdout_tasks.append((z_start, z_goal))

    from selflearnai.discovery import MacroCandidate

    test_cases = [
        {
            "name":       "useful_macro",
            "chain":      ("agentive", "plural"),
            "expected":   "PASS",
            "rationale":  "components NOT in baseline → adds capability",
        },
        {
            "name":       "useless_macro",
            "chain":      ("past_tense", "opposite"),
            "expected":   "FAIL",
            "rationale":  "components ARE in baseline; chain irrelevant to task",
        },
    ]

    sub_records: dict[str, dict] = {}
    for case in test_cases:
        macro_chain = case["chain"]
        # Use ALL trained ops (including agentive/plural even when they're
        # not in baseline) to BUILD the macro's callable. The macro is a
        # single composed op — it doesn't care that the planner's library
        # excludes its components.
        macro_op = make_macro_op(macro_chain, all_ops)
        macro_name = macro_name_from_chain(macro_chain)
        candidate = MacroCandidate(
            chain=macro_chain, n_occurrences=6,
            sources_seen=[v for v, _, _ in CHAIN_TRIPLES],
        )
        print(f"\n— Sub-case: {case['name']}  "
              f"(macro={macro_chain}, expected: {case['expected']})")
        print(f"   {case['rationale']}")
        result = evaluate_macro_utility(
            candidate, macro_op,
            operators=baseline_operators,
            holdout_tasks=holdout_tasks,
            macro_depth=args.macro_depth,
            cos_improvement_min=args.cos_improvement_min,
            per_task_improvement_min=args.per_task_improvement_min,
            n_tasks_improved_min=args.n_tasks_improved_min,
            beam_width=args.beam_width,
        )
        verdict = "PASS" if result.passes_utility else "FAIL"
        mark = "✓" if (
            (case["expected"] == "PASS" and result.passes_utility)
            or (case["expected"] == "FAIL" and not result.passes_utility)
        ) else "✗"
        print(
            f"   {mark} baseline_cos={result.planner_baseline_cos_mean:.3f}  "
            f"aug_cos={result.planner_augmented_cos_mean:.3f}  "
            f"Δ={result.planner_cos_improvement_mean:+.4f}  "
            f"tasks_improved={result.n_tasks_improved}/{result.n_planner_tasks}  "
            f"→ utility {verdict}"
        )
        if result.failing_reasons:
            for f in result.failing_reasons:
                print(f"     · {f}")
        sub_records[case["name"]] = {
            "macro_name": macro_name,
            "chain": list(macro_chain),
            "expected": case["expected"],
            "verdict": verdict,
            "planner_baseline_cos_mean": result.planner_baseline_cos_mean,
            "planner_augmented_cos_mean": result.planner_augmented_cos_mean,
            "planner_cos_improvement_mean": result.planner_cos_improvement_mean,
            "n_tasks_improved": result.n_tasks_improved,
            "n_planner_tasks": result.n_planner_tasks,
            "passes_utility": result.passes_utility,
            "failing_reasons": result.failing_reasons,
        }

    # Acceptance: useful_macro must PASS, useless_macro must FAIL.
    useful_pass = sub_records["useful_macro"]["passes_utility"]
    useless_fail = not sub_records["useless_macro"]["passes_utility"]
    part_b_pass = useful_pass and useless_fail

    print()
    print(f"  useful macro passes utility:    "
          f"{'PASS' if useful_pass else 'FAIL'}")
    print(f"  useless macro rejected:         "
          f"{'PASS' if useless_fail else 'FAIL'}")
    print(f"  → Part B overall: {'PASS' if part_b_pass else 'FAIL'}")

    return part_b_pass, {
        "baseline_library": sorted(baseline_operators.keys()),
        "sub_cases": sub_records,
        "useful_pass": useful_pass,
        "useless_fail": useless_fail,
        "pass": part_b_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-epochs", type=int, default=2000)
    # Mining
    parser.add_argument("--min-chain-length", type=int, default=2)
    parser.add_argument("--max-chain-length", type=int, default=3)
    parser.add_argument("--frequency-min", type=int, default=5)
    # Utility
    parser.add_argument("--macro-depth", type=int, default=1)
    parser.add_argument("--cos-improvement-min", type=float, default=0.01)
    parser.add_argument("--per-task-improvement-min", type=float, default=0.01)
    parser.add_argument("--n-tasks-improved-min", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--out", default="results/stage1_5/refactor_smoke.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Task 1.5.4 — e-graph subchain refactor + macro promotion smoke")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    # Encoder load happens once and is reused for Part B.
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    part_a_pass, part_a_record = part_a_frequency_gate(args)
    part_b_pass, part_b_record = part_b_utility_gate(args, encode, enc_cfg)

    # ---- Acceptance ----
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.5.4)")
    print("=" * 78)
    print(f"  Part A (frequency gate, synthetic buffer):  "
          f"{'PASS' if part_a_pass else 'FAIL'}")
    print(f"  Part B (utility gate, real macro):           "
          f"{'PASS' if part_b_pass else 'FAIL'}")
    overall = part_a_pass and part_b_pass
    print(f"\n→ Task 1.5.4: {'PASS' if overall else 'FAIL'}")

    payload = {
        "task": "1.5.4",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "thresholds": {
            "min_chain_length": args.min_chain_length,
            "max_chain_length": args.max_chain_length,
            "frequency_min": args.frequency_min,
            "macro_depth": args.macro_depth,
            "cos_improvement_min": args.cos_improvement_min,
            "per_task_improvement_min": args.per_task_improvement_min,
            "n_tasks_improved_min": args.n_tasks_improved_min,
        },
        "part_a": part_a_record,
        "part_b": part_b_record,
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
