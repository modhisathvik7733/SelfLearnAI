"""Task 1.3 — Tier-1 coverage report across canonical + paraphrase eval sets.

Runs the deterministic Tier-1 typed-grammar parser against every
configured eval file and produces a coverage table. Used to:

  1. Confirm Tier-1 still hits 100% on canonical phrasings (regression).
  2. Measure Tier-1's coverage on paraphrased phrasings — the baseline
     Tier 2 (Ψ-space classifier, Task 1.4) needs to lift.
  3. Per-concept breakdown so we can see which concepts have
     paraphrase-recall blind spots.

Acceptance gate (Task 1.3):
  - Canonical eval: 100% parseable + 100% refusal (no regression
    from Task 1.1).
  - Paraphrase eval: ≥ 30% parseable accuracy (the Tier-1 limit
    documented in plan §19.2; informational target — much lower
    than the canonical 100% by design).

Output: console summary + JSON to results/stage1/intent_coverage.json.
Pure CPU, no encoder / no model loading.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selflearnai.intent import TypedGrammarParser
from scripts.stage1_intent_grammar_smoke import read_eval_rows, classify


# Default eval set list. Each entry is (label, path, gate_dict).
# gate_dict carries the per-eval acceptance thresholds:
#   parseable_min: minimum fraction parseable_correct / parseable_total.
#   refuse_min:    minimum fraction refuse_correct / refuse_total. None
#                  if there are no refuse rows in this eval set.
#   hard:          if True, falling below the threshold fails Task 1.3.
#                  if False, the threshold is a SOFT TARGET — printed as
#                  a warning when missed but does not gate. Used for
#                  paraphrase coverage which Tier 1 deliberately can't
#                  fully solve (Tier 2 closes the gap in Task 1.4).
DEFAULT_EVALS: list[dict] = [
    {
        "label":   "canonical",
        "path":    "data/intent_eval/canonical.tsv",
        "gate":    {"parseable_min": 1.00, "refuse_min": 1.00, "hard": True},
    },
    {
        "label":   "paraphrase",
        "path":    "data/intent_eval/paraphrase.tsv",
        "gate":    {"parseable_min": 0.20, "refuse_min": None, "hard": False},
    },
]


def coverage_per_concept(rows, parser: TypedGrammarParser) -> dict[str, dict]:
    """Per-concept coverage, restricted to parseable rows (refuse rows
    don't have a target concept)."""
    by_concept: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "correct": 0, "wrong": 0, "missed": 0}
    )
    for question, concept, source, status in rows:
        if status != "parseable":
            continue
        by_concept[concept]["total"] += 1
        intent = parser(question)
        if intent is None:
            by_concept[concept]["missed"] += 1
        elif intent.concept == concept and intent.source.lower() == source.lower():
            by_concept[concept]["correct"] += 1
        else:
            by_concept[concept]["wrong"] += 1
    return dict(by_concept)


def evaluate_eval_set(parser, label: str, path: Path, gate: dict) -> dict:
    rows = read_eval_rows(path)
    report = classify(parser, rows)
    counters = report["counters"]
    per_concept = coverage_per_concept(rows, parser)

    parseable_acc = (
        counters["parseable_correct"] / counters["parseable_total"]
        if counters["parseable_total"] else float("nan")
    )
    refuse_acc = (
        counters["refuse_correct"] / counters["refuse_total"]
        if counters["refuse_total"] else None
    )

    parseable_meets = (
        counters["parseable_total"] == 0
        or parseable_acc >= gate["parseable_min"]
    )
    if gate["refuse_min"] is None or counters["refuse_total"] == 0:
        refuse_meets = True
    else:
        refuse_meets = refuse_acc >= gate["refuse_min"]

    # Soft gates report PASS (informational) regardless; hard gates
    # propagate FAIL up to the overall acceptance check.
    is_hard = gate.get("hard", True)
    parseable_pass = parseable_meets if is_hard else True
    refuse_pass = refuse_meets if is_hard else True

    return {
        "label": label,
        "path": str(path),
        "n_rows": len(rows),
        "counters": counters,
        "per_concept": per_concept,
        "parseable_acc": parseable_acc,
        "refuse_acc": refuse_acc,
        "gate": gate,
        "is_hard_gate": is_hard,
        "parseable_meets": parseable_meets,
        "refuse_meets": refuse_meets,
        "parseable_pass": parseable_pass,
        "refuse_pass": refuse_pass,
        "pass": parseable_pass and refuse_pass,
        "failures": report["failures"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        "--out", default="results/stage1/intent_coverage.json",
    )
    parser.add_argument(
        "--show-failures", action="store_true", default=False,
        help="Print individual missed paraphrases (verbose).",
    )
    args = parser.parse_args()

    print("Task 1.3 — Tier-1 typed-grammar coverage report")
    print("=" * 78)

    parser_obj = TypedGrammarParser()
    print(f"Patterns:  {parser_obj.num_patterns}")
    print(f"Concepts:  {parser_obj.supported_concepts}\n")

    results: list[dict] = []
    for spec in DEFAULT_EVALS:
        path = Path(spec["path"])
        if not path.exists():
            print(f"  ⚠ skipping {spec['label']}: file {path} not found")
            continue
        r = evaluate_eval_set(parser_obj, spec["label"], path, spec["gate"])
        results.append(r)

    # ---- Per-eval summary ----
    print("=" * 78)
    print("EVAL SUMMARY")
    print("=" * 78)
    header = f"  {'eval':<14}  {'rows':>5}  {'parseable':>9}  {'refuse':>7}  {'acc-gate':>9}  status"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        c = r["counters"]
        cell_paraphrase = (
            f"{c['parseable_correct']}/{c['parseable_total']}"
        ) if c["parseable_total"] else "-"
        cell_refuse = (
            f"{c['refuse_correct']}/{c['refuse_total']}"
        ) if c["refuse_total"] else "-"
        gate_str = f">= {r['gate']['parseable_min']:.2f}"
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"  {r['label']:<14}  {r['n_rows']:>5}  {cell_paraphrase:>9}  "
            f"{cell_refuse:>7}  {gate_str:>9}  {status}"
        )

    # ---- Per-concept paraphrase coverage ----
    para_result = next((r for r in results if r["label"] == "paraphrase"), None)
    if para_result:
        print("\nPer-concept paraphrase coverage (Tier-1 baseline; Tier-2 lifts):")
        rows_pc = para_result["per_concept"]
        print(f"  {'concept':<14}  {'correct':>7}  {'missed':>6}  {'wrong':>5}  {'total':>5}  {'acc':>5}")
        for c in sorted(rows_pc.keys()):
            d = rows_pc[c]
            acc = d["correct"] / d["total"] if d["total"] else 0
            print(f"  {c:<14}  {d['correct']:>7}  {d['missed']:>6}  {d['wrong']:>5}  {d['total']:>5}  {acc:>5.2f}")

    # ---- Failures (paraphrase only; canonical failures would already be a regression) ----
    if args.show_failures and para_result and para_result["failures"]:
        print("\nParaphrase failures (informational, not gated):")
        for f in para_result["failures"]:
            print(f"  [{f['type']}] {f['question']!r}")

    # ---- Acceptance gate ----
    overall = all(r["pass"] for r in results) and bool(results)
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.3)")
    print("=" * 78)
    for r in results:
        cov = r["parseable_acc"] if r["counters"]["parseable_total"] else 0.0
        gate_kind = "hard" if r["is_hard_gate"] else "soft target"
        meets_str = "OK" if r["parseable_meets"] else "below"
        if r["is_hard_gate"]:
            outcome = "PASS" if r["parseable_pass"] else "FAIL"
        else:
            outcome = "PASS (informational)"
        print(
            f"  {r['label']:<14}  parseable={cov:.3f}  "
            f"({gate_kind} >= {r['gate']['parseable_min']:.2f}: {meets_str})  "
            f"→ {outcome}"
        )
    print(f"\n→ Task 1.3: {'PASS' if overall else 'FAIL'}")

    # ---- Save JSON ----
    payload = {
        "task": "1.3",
        "n_patterns": parser_obj.num_patterns,
        "concepts": parser_obj.supported_concepts,
        "results": [
            {k: v for k, v in r.items() if k != "failures"}
            for r in results
        ],
        # Failures separately to keep the main payload scannable.
        "failures": {
            r["label"]: r["failures"] for r in results
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
