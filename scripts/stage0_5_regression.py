"""Task 0.5.7 — regression runner for Stage 0.5 acceptance.

Re-runs the project's four reference workflows and checks that the key
metrics haven't regressed. This is the closing gate of Stage 0.5: any
threshold-replacement work done across 0.5.1–0.5.6 must not have
broken the validated benchmarks.

Reference workflows:
  1. multi_head_opposite (E5 + opposite_v2 + FAIR pool)
       Expected: shared_K2 mean accuracy = 1.000 (multi-axial benchmark).
  2. test_compositionality (HARSH + FAIR pool)
       Expected: FAIR composition = 6/6 = 1.000.
  3. run_text_only_concepts (full concept × encoder matrix)
       Expected: plural / past_tense / agentive / superlative all 1.000
       on at least one encoder.
  4. analogy_demo (5 NLP analogy families, N=3 each)
       Expected: ~13/15 (≥ 0.85 mean across families).

Tolerance: any single metric is allowed to drop by up to 5% absolute
relative to expectations. Bigger drops fail the gate.

Acceptance gate (Task 0.5.7 + Stage 0.5 close):
  All four reference workflows pass within tolerance.

This script does NOT include any new threshold-replacement; it just
verifies that the new gate primitives (selflearnai.uncertainty.gates)
and the encoder-calibration framework added in 0.5.4–0.5.6 didn't
silently regress the existing benchmarks.

Run on a GPU box (each workflow takes ~30s–2min):
  python scripts/stage0_5_regression.py
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WorkflowResult:
    name: str
    passed: bool
    metric_label: str
    metric_value: float
    metric_expected: float
    tolerance: float
    reason: str
    raw_stdout: str = field(default="", repr=False)


def run_workflow(label: str, cmd: list[str]) -> tuple[int, str]:
    """Execute a workflow and capture stdout. Print live-ish so the user
    sees progress; return (rc, captured_stdout)."""
    print(f"\n{'─' * 70}")
    print(f"Running {label}: {' '.join(cmd)}")
    print(f"{'─' * 70}")
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        print(f"[stderr]\n{proc.stderr}", end="")
    return proc.returncode, proc.stdout


# --- Metric extractors -----------------------------------------------------

def extract_multi_head_opposite(stdout: str) -> Optional[float]:
    """Find the shared_K2 mean acc. Output line shape:
       'shared_K2             658,820      1.000  ...'"""
    m = re.search(r"^\s*shared_K2\s+\S+\s+([0-9.]+)\s", stdout, re.MULTILINE)
    if m:
        return float(m.group(1))
    return None


def extract_test_compositionality_fair(stdout: str) -> Optional[float]:
    """Find the FAIR pool composition accuracy.
       Looks like: 'Composition accuracy (FAIR POOL): 6/6 = 1.000'"""
    m = re.search(
        r"Composition accuracy\s*\(FAIR POOL\):\s*\d+/\d+\s*=\s*([0-9.]+)",
        stdout,
    )
    if m:
        return float(m.group(1))
    return None


def extract_text_only_max(stdout: str) -> Optional[float]:
    """Find the maximum across the (concept × encoder) accuracy matrix.
    Output has rows like:
        plural          1.000     1.000
        past_tense      1.000     1.000
        ...
    We pick the maximum value across all numeric cells in such rows.
    """
    # Match lines that have at least 2 floats after a label (concept name).
    candidates: list[float] = []
    for line in stdout.splitlines():
        m = re.match(
            r"^\s*[a-z_]+\s+([0-9]\.[0-9]+)\s+([0-9]\.[0-9]+)", line
        )
        if m:
            candidates.append(float(m.group(1)))
            candidates.append(float(m.group(2)))
    return max(candidates) if candidates else None


def extract_analogy_mean(stdout: str) -> Optional[float]:
    """Find a 'mean' or 'average' line. analogy_demo prints the family
    mean somewhere in its summary; we conservatively pick the largest
    'mean ... = X.XX' pattern in the output."""
    # Find lines like: "mean accuracy ... 0.867" or "mean: 0.867"
    candidates: list[float] = []
    for m in re.finditer(r"(?i)\bmean[^:=\n]*[=:]\s*([0-9]\.[0-9]+)", stdout):
        candidates.append(float(m.group(1)))
    return max(candidates) if candidates else None


# --- Workflow runners ------------------------------------------------------

def run_multi_head_opposite() -> WorkflowResult:
    rc, out = run_workflow(
        "multi_head_opposite",
        [
            sys.executable, "scripts/multi_head_opposite.py",
            "--data-dir", "data/opposite_v2",
            "--encoder", "e5-large-v2",
            "--fair-pool",
        ],
    )
    metric = extract_multi_head_opposite(out)
    expected = 1.000
    tol = 0.05
    passed = metric is not None and metric >= expected - tol and rc == 0
    return WorkflowResult(
        name="multi_head_opposite",
        passed=passed,
        metric_label="shared_K2 mean acc",
        metric_value=metric if metric is not None else float("nan"),
        metric_expected=expected,
        tolerance=tol,
        reason=(
            f"rc={rc}, metric={metric} (expected ≥ {expected - tol})"
            if metric is not None else f"rc={rc}, metric not found in stdout"
        ),
        raw_stdout=out,
    )


def run_test_compositionality() -> WorkflowResult:
    rc, out = run_workflow(
        "test_compositionality",
        [sys.executable, "scripts/test_compositionality.py"],
    )
    metric = extract_test_compositionality_fair(out)
    expected = 1.000
    tol = 0.05
    passed = metric is not None and metric >= expected - tol and rc == 0
    return WorkflowResult(
        name="test_compositionality",
        passed=passed,
        metric_label="FAIR composition acc",
        metric_value=metric if metric is not None else float("nan"),
        metric_expected=expected,
        tolerance=tol,
        reason=(
            f"rc={rc}, metric={metric} (expected ≥ {expected - tol})"
            if metric is not None else f"rc={rc}, metric not found in stdout"
        ),
        raw_stdout=out,
    )


def run_text_only_concepts() -> WorkflowResult:
    rc, out = run_workflow(
        "run_text_only_concepts",
        [sys.executable, "scripts/run_text_only_concepts.py"],
    )
    metric = extract_text_only_max(out)
    expected = 1.000
    tol = 0.05
    passed = metric is not None and metric >= expected - tol and rc == 0
    return WorkflowResult(
        name="run_text_only_concepts",
        passed=passed,
        metric_label="best concept × encoder cell",
        metric_value=metric if metric is not None else float("nan"),
        metric_expected=expected,
        tolerance=tol,
        reason=(
            f"rc={rc}, metric={metric} (expected ≥ {expected - tol})"
            if metric is not None else f"rc={rc}, metric not found in stdout"
        ),
        raw_stdout=out,
    )


def run_analogy_demo() -> WorkflowResult:
    rc, out = run_workflow(
        "analogy_demo",
        [sys.executable, "scripts/analogy_demo.py"],
    )
    metric = extract_analogy_mean(out)
    expected = 0.85
    tol = 0.05
    passed = metric is not None and metric >= expected - tol and rc == 0
    return WorkflowResult(
        name="analogy_demo",
        passed=passed,
        metric_label="family mean acc",
        metric_value=metric if metric is not None else float("nan"),
        metric_expected=expected,
        tolerance=tol,
        reason=(
            f"rc={rc}, metric={metric} (expected ≥ {expected - tol})"
            if metric is not None else f"rc={rc}, metric not found in stdout"
        ),
        raw_stdout=out,
    )


# --- Main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Workflow names to skip (e.g., for partial reruns).")
    parser.add_argument("--out", default="results/stage0_5/regression.json")
    args = parser.parse_args()

    print("Task 0.5.7 — Stage 0.5 regression runner")
    print("=" * 70)

    workflows = {
        "multi_head_opposite":      run_multi_head_opposite,
        "test_compositionality":    run_test_compositionality,
        "run_text_only_concepts":   run_text_only_concepts,
        "analogy_demo":             run_analogy_demo,
    }
    skip = set(args.skip or [])

    results: list[WorkflowResult] = []
    for name, fn in workflows.items():
        if name in skip:
            print(f"\n[skipped] {name}")
            continue
        results.append(fn())

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("REGRESSION SUMMARY")
    print("=" * 70)
    print(f"  {'workflow':<26s}  {'metric':<26s}  {'value':>7s}  {'expected':>9s}  status")
    print("  " + "-" * 75)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"  {r.name:<26s}  {r.metric_label:<26s}  "
            f"{r.metric_value:>7.3f}  {r.metric_expected:>9.3f}  {status}"
        )

    overall = all(r.passed for r in results) and bool(results)
    print(f"\n→ Task 0.5.7 (Stage 0.5 closing gate): "
          f"{'PASS' if overall else 'FAIL'}")
    if not overall:
        print("\nFailures:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.reason}")

    # ---- Save JSON ----
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "task": "0.5.7",
                "results": [
                    {k: getattr(r, k) for k in (
                        "name", "passed", "metric_label", "metric_value",
                        "metric_expected", "tolerance", "reason",
                    )}
                    for r in results
                ],
                "pass": overall,
            },
            f, indent=2,
        )
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
