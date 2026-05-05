"""Task 1.5.7 — Stage 1.5 closing regression runner.

Re-runs every Stage 1.5 acceptance script in order and reports a single
PASS/FAIL. Mirror of stage0_5_regression.py and stage1_regression.py
but for Stage 1.5 (concept discovery / wake-sleep library learning).

Each Stage 1.5 acceptance script self-reports via:
  - exit code 0 (PASS) / 1 (FAIL)
  - a final `→ Task X.Y.Z: PASS|FAIL` verdict line on stdout

so this runner doesn't re-extract metrics — it trusts each script's
own gate and surfaces the verdict line in the summary table. The
script-level gates were chosen and audited at the time the task
shipped (often after 1–2 reframings — see the relative-gates lesson
in feedback_relative_gates_in_encoder_space).

Workflows (in execution order):

    Task 1.5.1   wake replay buffer (4-channel)
    Task 1.5.2   Ψ-shift clustering + operator-consistency
    Task 1.5.3   three-criterion candidate validation gate
    Task 1.5.4   e-graph subchain refactor + macro promotion
    Task 1.5.5   versioned concept registry with LRU growth control
    Task 1.5.6   end-to-end sleep cycle orchestrator

Encoder policy: encoder-bearing scripts default to e5-large-v2 (Stage
1.5 evidence anchor — every smoke that needed an encoder passed on
e5 first). Pass `--encoder gte-base` to re-run on the alternate
encoder. Encoder-agnostic scripts (1.5.1, 1.5.5) execute once
regardless.

Stage 1 chain: by default this runner ALSO invokes
scripts/stage1_regression.py at the end on the chosen encoder, so
Stage 1.5 doesn't close without proving it didn't regress Stage 1.
Disable with --no-include-stage1-regression for partial reruns.

Acceptance gate (Task 1.5.7 + Stage 1.5 close):
  All Stage 1.5 workflows PASS on the chosen encoder(s) AND
  (if included) Stage 1 regression PASSES.

Run:
  python scripts/stage1_5_regression.py
  python scripts/stage1_5_regression.py --encoder gte-base
  python scripts/stage1_5_regression.py --no-include-stage1-regression
  python scripts/stage1_5_regression.py --skip 1.5.3 1.5.4
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
    task_id: str          # e.g. "1.5.6"
    label: str
    encoder: str | None
    cmd: list[str]
    rc: int
    verdict_line: str
    passed: bool
    raw_stdout: str = field(default="", repr=False)


# task_id, short_label, script_path, takes_encoder
WORKFLOWS: list[tuple[str, str, str, bool]] = [
    ("1.5.1", "wake replay buffer",          "scripts/stage1_5_wake_buffer_smoke.py", False),
    ("1.5.2", "Ψ-shift cluster + consistency", "scripts/stage1_5_cluster_smoke.py",     True),
    ("1.5.3", "three-criterion validation",  "scripts/stage1_5_validation_smoke.py",  True),
    ("1.5.4", "macro refactor + promotion",   "scripts/stage1_5_refactor_smoke.py",    True),
    ("1.5.5", "versioned concept registry",   "scripts/stage1_5_registry_smoke.py",    False),
    ("1.5.6", "end-to-end sleep cycle",       "scripts/stage1_5_sleep_smoke.py",       True),
]


VERDICT_RE = re.compile(r"→\s*Task\s*([0-9]+\.[0-9]+\.[0-9]+|[0-9]+\.[0-9]+[a-z]?)\s*:\s*(PASS|FAIL)")


def extract_verdict_line(stdout: str) -> str:
    """Return the last `→ Task X.Y.Z: PASS|FAIL` line in stdout, or '' if
    the script didn't print one."""
    matches = list(VERDICT_RE.finditer(stdout))
    if not matches:
        return ""
    last = matches[-1]
    line_start = stdout.rfind("\n", 0, last.start()) + 1
    line_end = stdout.find("\n", last.end())
    if line_end == -1:
        line_end = len(stdout)
    return stdout[line_start:line_end].strip()


def run_one(
    task_id: str, label: str, script: str, encoder: str | None,
) -> WorkflowResult:
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
             "for the full Stage 1.5 close (e.g. e5-large-v2 gte-base).",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Task IDs to skip (e.g. --skip 1.5.3 1.5.4).",
    )
    parser.add_argument(
        "--include-stage1-regression",
        dest="include_stage1_regression",
        action="store_true",
        default=True,
        help="At the end, also run scripts/stage1_regression.py to confirm "
             "Stage 1.5 did not regress Stage 1 (default: on).",
    )
    parser.add_argument(
        "--no-include-stage1-regression",
        dest="include_stage1_regression",
        action="store_false",
        help="Skip the chained Stage 1 regression (faster partial reruns).",
    )
    parser.add_argument(
        "--out", default="results/stage1_5/regression.json",
    )
    args = parser.parse_args()

    if args.encoder is not None and args.encoders is not None:
        parser.error("--encoder and --encoders are mutually exclusive")
    encoders: list[str] = (
        args.encoders if args.encoders is not None
        else [args.encoder if args.encoder is not None else "e5-large-v2"]
    )
    skip = set(args.skip or [])

    print("Task 1.5.7 — Stage 1.5 closing regression")
    print("=" * 74)
    print(f"Encoders for encoder-bearing scripts: {encoders}")
    if skip:
        print(f"Skipping: {sorted(skip)}")
    if args.include_stage1_regression:
        print("Will chain into Stage 1 regression after Stage 1.5 sub-tasks.")
    else:
        print("Stage 1 regression chain: SKIPPED.")

    all_results: list[WorkflowResult] = []
    for encoder in encoders:
        for task_id, label, script, takes_enc in WORKFLOWS:
            if task_id in skip:
                continue
            enc = encoder if takes_enc else None
            # Encoder-agnostic scripts only need to run once.
            if not takes_enc and any(
                r.task_id == task_id for r in all_results
            ):
                continue
            all_results.append(run_one(task_id, label, script, enc))

    # ---- Optional chain into Stage 1 regression ----
    stage1_result: WorkflowResult | None = None
    if args.include_stage1_regression:
        # Use the first encoder for the chained run — Stage 1 regression
        # itself can take a list, but for closing-gate purposes one
        # encoder is enough (the per-encoder evidence already shipped
        # in Task 1.13).
        chain_encoder = encoders[0]
        print(f"\n{'═' * 74}")
        print(f"Chained: Stage 1 regression on {chain_encoder}")
        print(f"{'═' * 74}")
        cmd = [
            sys.executable, "scripts/stage1_regression.py",
            "--encoder", chain_encoder,
        ]
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0 and proc.stderr:
            print(f"[stderr]\n{proc.stderr}", end="")
        stage1_result = WorkflowResult(
            task_id="1.13",
            label="Stage 1 closing regression (chained)",
            encoder=chain_encoder,
            cmd=cmd,
            rc=proc.returncode,
            verdict_line=extract_verdict_line(proc.stdout),
            passed=(proc.returncode == 0),
            raw_stdout=proc.stdout,
        )

    # ---- Summary ----
    print("\n" + "=" * 74)
    print("STAGE 1.5 REGRESSION SUMMARY")
    print("=" * 74)
    print(
        f"  {'task':>7}  {'encoder':<14}  {'label':<32}  status   verdict"
    )
    print("  " + "-" * 84)
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        enc_cell = r.encoder if r.encoder else "—"
        verdict_cell = r.verdict_line if r.verdict_line else f"(rc={r.rc}, no verdict line)"
        print(
            f"  {r.task_id:>7}  {enc_cell:<14}  {r.label:<32}  "
            f"{status:<7} {verdict_cell}"
        )
    if stage1_result is not None:
        s = stage1_result
        status = "PASS" if s.passed else "FAIL"
        verdict_cell = s.verdict_line if s.verdict_line else f"(rc={s.rc}, no verdict line)"
        print(
            f"  {s.task_id:>7}  {s.encoder:<14}  {s.label:<32}  "
            f"{status:<7} {verdict_cell}"
        )

    overall = (
        bool(all_results)
        and all(r.passed for r in all_results)
        and (stage1_result is None or stage1_result.passed)
    )
    print(f"\n→ Task 1.5.7 (Stage 1.5 closing gate): "
          f"{'PASS' if overall else 'FAIL'}")
    if not overall:
        print("\nFailures:")
        for r in all_results:
            if not r.passed:
                enc_str = f" [{r.encoder}]" if r.encoder else ""
                print(f"  - Task {r.task_id}{enc_str}: rc={r.rc}, "
                      f"verdict='{r.verdict_line or '(missing)'}'")
        if stage1_result is not None and not stage1_result.passed:
            print(f"  - Chained Stage 1 regression: rc={stage1_result.rc}, "
                  f"verdict='{stage1_result.verdict_line or '(missing)'}'")

    # ---- Save JSON ----
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "task": "1.5.7",
                "encoders": encoders,
                "skipped": sorted(skip),
                "include_stage1_regression": args.include_stage1_regression,
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
                "stage1_chained_result": (
                    {
                        "task_id": stage1_result.task_id,
                        "encoder": stage1_result.encoder,
                        "rc": stage1_result.rc,
                        "verdict_line": stage1_result.verdict_line,
                        "passed": stage1_result.passed,
                    }
                    if stage1_result is not None else None
                ),
                "pass": overall,
            },
            f, indent=2,
        )
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
