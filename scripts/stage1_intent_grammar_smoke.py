"""Task 1.1 — Tier-1 typed-grammar intent parser smoke test.

Reads `data/intent_eval/canonical.tsv` and runs the parser against each
row, classifying every prediction into one of four bins:

  - parseable_correct : status=parseable, parser matched the right
                        (concept, source).
  - parseable_wrong   : status=parseable, parser matched but with wrong
                        concept or source.
  - parseable_missed  : status=parseable, parser returned None.
  - refuse_correct    : status=refuse, parser returned None.
  - refuse_violated   : status=refuse, parser matched anyway.

Acceptance gate (Task 1.1):
  - 100% accuracy on parseable rows (every canonical phrasing matches
    the right concept and source).
  - 100% refusal on refuse rows (no out-of-library pattern is hijacked
    by an existing regex).

Pure CPU, no encoder / no model loading. Runs in milliseconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selflearnai.intent import Intent, TypedGrammarParser


def _strip(s: str) -> str:
    return s.strip()


def read_eval_rows(path: Path) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                raise ValueError(
                    f"{path}: row has < 4 tab columns: {line!r}"
                )
            question = _strip(parts[0])
            concept = _strip(parts[1])
            source = _strip(parts[2])
            status = _strip(parts[3])
            if status not in ("parseable", "refuse"):
                raise ValueError(
                    f"{path}: status must be parseable|refuse, got {status!r}"
                )
            rows.append((question, concept, source, status))
    return rows


def classify(parser: TypedGrammarParser, rows: list[tuple[str, str, str, str]]) -> dict:
    counters = {
        "parseable_total": 0,
        "parseable_correct": 0,
        "parseable_wrong": 0,
        "parseable_missed": 0,
        "refuse_total": 0,
        "refuse_correct": 0,
        "refuse_violated": 0,
    }
    failures: list[dict] = []

    for question, concept, source, status in rows:
        intent = parser(question)
        if status == "parseable":
            counters["parseable_total"] += 1
            if intent is None:
                counters["parseable_missed"] += 1
                failures.append({
                    "type": "parseable_missed",
                    "question": question,
                    "expected_concept": concept,
                    "expected_source": source,
                    "got": None,
                })
            elif intent.concept == concept and intent.source.lower() == source.lower():
                counters["parseable_correct"] += 1
            else:
                counters["parseable_wrong"] += 1
                failures.append({
                    "type": "parseable_wrong",
                    "question": question,
                    "expected_concept": concept,
                    "expected_source": source,
                    "got_concept": intent.concept,
                    "got_source": intent.source,
                    "got_pattern_id": intent.pattern_id,
                })
        elif status == "refuse":
            counters["refuse_total"] += 1
            if intent is None:
                counters["refuse_correct"] += 1
            else:
                counters["refuse_violated"] += 1
                failures.append({
                    "type": "refuse_violated",
                    "question": question,
                    "got_concept": intent.concept,
                    "got_source": intent.source,
                    "got_pattern_id": intent.pattern_id,
                })

    return {"counters": counters, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        "--eval", default="data/intent_eval/canonical.tsv",
        help="Path to canonical eval TSV.",
    )
    parser.add_argument(
        "--out", default="results/stage1/intent_grammar_smoke.json",
    )
    parser.add_argument(
        "--show-failures", action="store_true", default=True,
        help="Print each failure (default true).",
    )
    args = parser.parse_args()

    print("Task 1.1 — Tier-1 typed-grammar intent parser smoke")
    print("=" * 70)

    parser_obj = TypedGrammarParser()
    print(f"Patterns:  {parser_obj.num_patterns}")
    print(f"Concepts:  {parser_obj.supported_concepts}")

    eval_path = Path(args.eval)
    rows = read_eval_rows(eval_path)
    print(f"Eval set:  {eval_path}  ({len(rows)} rows)\n")

    report = classify(parser_obj, rows)
    c = report["counters"]

    print(f"  parseable: {c['parseable_correct']}/{c['parseable_total']} correct, "
          f"{c['parseable_wrong']} wrong, {c['parseable_missed']} missed")
    print(f"  refuse:    {c['refuse_correct']}/{c['refuse_total']} correct, "
          f"{c['refuse_violated']} violated")

    if report["failures"] and args.show_failures:
        print("\nFailures:")
        for f in report["failures"]:
            if f["type"] == "parseable_missed":
                print(f"  [missed] {f['question']!r} → expected ({f['expected_concept']}, {f['expected_source']})")
            elif f["type"] == "parseable_wrong":
                print(
                    f"  [wrong]  {f['question']!r} → expected "
                    f"({f['expected_concept']}, {f['expected_source']}); "
                    f"got ({f['got_concept']}, {f['got_source']}) via {f['got_pattern_id']}"
                )
            elif f["type"] == "refuse_violated":
                print(
                    f"  [hijacked refuse] {f['question']!r} → "
                    f"matched ({f['got_concept']}, {f['got_source']}) via {f['got_pattern_id']}"
                )

    parseable_pass = (
        c["parseable_total"] > 0
        and c["parseable_correct"] == c["parseable_total"]
    )
    refuse_pass = (
        c["refuse_total"] > 0
        and c["refuse_correct"] == c["refuse_total"]
    )
    overall = parseable_pass and refuse_pass

    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK (Task 1.1)")
    print("=" * 70)
    print(f"  Parseable rows: {'PASS' if parseable_pass else 'FAIL'} "
          f"({c['parseable_correct']}/{c['parseable_total']})")
    print(f"  Refuse rows:    {'PASS' if refuse_pass else 'FAIL'} "
          f"({c['refuse_correct']}/{c['refuse_total']})")
    print(f"\n→ Task 1.1: {'PASS' if overall else 'FAIL'}")

    payload = {
        "task": "1.1",
        "eval_path": str(eval_path),
        "n_rows": len(rows),
        "n_patterns": parser_obj.num_patterns,
        "concepts": parser_obj.supported_concepts,
        "counters": c,
        "failures": report["failures"],
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
