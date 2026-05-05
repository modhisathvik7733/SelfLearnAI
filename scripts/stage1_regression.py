"""Task 1.13 — Stage 1 closing regression runner.

Re-runs every Stage 1 acceptance script in order and reports a single
PASS/FAIL. Mirror of `stage0_5_regression.py` but for Stage 1.

Each Stage 1 acceptance script self-reports via:
  - exit code 0 (PASS) / 1 (FAIL)
  - a final `→ Task X.Y: PASS|FAIL` verdict line on stdout

so this runner doesn't re-extract metrics — it trusts each script's
own gate and surfaces the verdict line in the summary table. The
script-level gates were chosen and audited at the time the task
shipped; second-guessing them here would be hiding signal.

Workflows (in execution order):

    Task 1.1   intent grammar smoke
    Task 1.3   intent coverage report (Tier-1)
    Task 1.4   intent Tier-2 classifier training
    Task 1.5   intent Tier-2 conformal calibration
    Task 1.6   beam-search planner (chain + end-state)
    Task 1.7   cos-progress + step-bonus heuristic
    Task 1.8   operator prior + chain_hint integration
    Task 1.10  per-step verification gates (Type/Axiom/Drift/Conformal)
    Task 1.11  Ψ-program serialization round-trip + replay
    Task 1.12  depth-3-to-5 planner validation

Tasks 1.2 (no standalone test) and 1.9 (deferred value function) are
intentionally absent.

Encoder policy: encoder-bearing scripts default to e5-large-v2 because
that's where most Stage 1 acceptance evidence was collected and where
chain recovery sits at its strongest baseline. Pass `--encoder
gte-base` to re-run the same gauntlet on the alternate encoder, or
`--encoders e5-large-v2 gte-base` to run both back-to-back (each
encoder produces an independent PASS/FAIL).

Acceptance gate (Task 1.13 + Stage 1 close):
  All workflows PASS (exit 0) on the chosen encoder(s).

Run:
  python scripts/stage1_regression.py
  python scripts/stage1_regression.py --encoder gte-base
  python scripts/stage1_regression.py --encoders e5-large-v2 gte-base
  python scripts/stage1_regression.py --skip 1.4 1.5     # partial rerun
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WorkflowResult:
    task_id: str          # e.g. "1.6"
    label: str            # short human label
    encoder: str | None   # None for encoder-agnostic scripts
    cmd: list[str]
    rc: int
    verdict_line: str
    passed: bool
    raw_stdout: str = field(default="", repr=False)


# task_id, short_label, script_path, takes_encoder
WORKFLOWS: list[tuple[str, str, str, bool]] = [
    ("1.1",  "intent grammar smoke",         "scripts/stage1_intent_grammar_smoke.py",        False),
    ("1.3",  "intent coverage report",       "scripts/stage1_intent_coverage_report.py",      False),
    ("1.4",  "intent Tier-2 classifier",     "scripts/stage1_intent_classifier_train.py",     True),
    ("1.5",  "intent Tier-2 calibration",    "scripts/stage1_intent_classifier_calibrate.py", True),
    ("1.6",  "beam-search planner",          "scripts/stage1_planner_beam_smoke.py",          True),
    ("1.7",  "cos-progress + step-bonus",    "scripts/stage1_planner_progress_smoke.py",      True),
    ("1.8",  "prior + chain_hint",           "scripts/stage1_planner_prior_train.py",         True),
    ("1.10", "verification gates",           "scripts/stage1_planner_verifier_smoke.py",      True),
    ("1.11", "Ψ-program trace round-trip",   "scripts/stage1_planner_trace_smoke.py",         True),
    ("1.12", "depth-3-to-5 validation",      "scripts/stage1_planner_depth_validation.py",    True),
]


VERDICT_RE = re.compile(r"→\s*Task\s*([0-9]+\.[0-9]+[a-z]?)\s*:\s*(PASS|FAIL)")


def extract_verdict_line(stdout: str) -> str:
    """Return the last `→ Task X.Y: PASS|FAIL` line in stdout, or '' if
    the script didn't print one. Some scripts (e.g. 1.10) print their
    sub-gate lines first, then the overall task line — we want the
    overall, which is always last."""
    matches = list(VERDICT_RE.finditer(stdout))
    if not matches:
        return ""
    last = matches[-1]
    # Return the full surrounding line for the table.
    line_start = stdout.rfind("\n", 0, last.start()) + 1
    line_end = stdout.find("\n", last.end())
    if line_end == -1:
        line_end = len(stdout)
    return stdout[line_start:line_end].strip()


def run_one(task_id: str, label: str, script: str, encoder: str | None) -> WorkflowResult:
    cmd = [sys.executable, script]
    if encoder is not None:
        cmd += ["--encoder", encoder]
    enc_str = f" [{encoder}]" if encoder else ""
    print(f"\n{'─' * 74}")
    print(f"Task {task_id}{enc_str}: {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'─' * 74}")
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0 and proc.stderr:
        print(f"[stderr]\n{proc.stderr}", end="")
    verdict = extract_verdict_line(proc.stdout)
    return WorkflowResult(
        task_id=task_id,
        label=label,
        encoder=encoder,
        cmd=cmd,
        rc=proc.returncode,
        verdict_line=verdict,
        passed=(proc.returncode == 0),
        raw_stdout=proc.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        "--encoder", default=None,
        help="Single encoder for all encoder-bearing scripts. "
             "Default: e5-large-v2. Mutually exclusive with --encoders.",
    )
    parser.add_argument(
        "--encoders", nargs="+", default=None,
        help="Run the gauntlet once per encoder in this list. Useful "
             "for the full Stage 1 close (e.g. e5-large-v2 gte-base).",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Task IDs to skip (e.g. --skip 1.4 1.5).",
    )
    parser.add_argument(
        "--out", default="results/stage1/regression.json",
    )
    args = parser.parse_args()

    if args.encoder is not None and args.encoders is not None:
        parser.error("--encoder and --encoders are mutually exclusive")
    encoders: list[str] = (
        args.encoders if args.encoders is not None
        else [args.encoder if args.encoder is not None else "e5-large-v2"]
    )
    skip = set(args.skip or [])

    print("Task 1.13 — Stage 1 closing regression")
    print("=" * 74)
    print(f"Encoders for encoder-bearing scripts: {encoders}")
    if skip:
        print(f"Skipping: {sorted(skip)}")

    all_results: list[WorkflowResult] = []
    for encoder in encoders:
        for task_id, label, script, takes_enc in WORKFLOWS:
            if task_id in skip:
                continue
            enc = encoder if takes_enc else None
            # Encoder-agnostic scripts only need to run once across the
            # whole runner, even when --encoders has multiple entries.
            if not takes_enc and any(
                r.task_id == task_id for r in all_results
            ):
                continue
            all_results.append(run_one(task_id, label, script, enc))

    # ---- Summary ----
    print("\n" + "=" * 74)
    print("STAGE 1 REGRESSION SUMMARY")
    print("=" * 74)
    print(
        f"  {'task':>5}  {'encoder':<14}  {'label':<30}  status   verdict"
    )
    print("  " + "-" * 80)
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        enc_cell = r.encoder if r.encoder else "—"
        verdict_cell = r.verdict_line if r.verdict_line else f"(rc={r.rc}, no verdict line)"
        print(
            f"  {r.task_id:>5}  {enc_cell:<14}  {r.label:<30}  "
            f"{status:<7} {verdict_cell}"
        )

    overall = bool(all_results) and all(r.passed for r in all_results)
    print(f"\n→ Task 1.13 (Stage 1 closing gate): "
          f"{'PASS' if overall else 'FAIL'}")
    if not overall:
        print("\nFailures:")
        for r in all_results:
            if not r.passed:
                enc_str = f" [{r.encoder}]" if r.encoder else ""
                print(f"  - Task {r.task_id}{enc_str}: rc={r.rc}, "
                      f"verdict='{r.verdict_line or '(missing)'}'")

    # ---- Save JSON ----
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "task": "1.13",
                "encoders": encoders,
                "skipped": sorted(skip),
                "results": [
                    {
                        "task_id": r.task_id,
                        "label": r.label,
                        "encoder": r.encoder,
                        "cmd": r.cmd,
                        "rc": r.rc,
                        "verdict_line": r.verdict_line,
                        "passed": r.passed,
                    }
                    for r in all_results
                ],
                "pass": overall,
            },
            f, indent=2,
        )
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
