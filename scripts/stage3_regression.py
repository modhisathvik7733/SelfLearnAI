"""Sub-task 3.7 — Stage 3 closing regression runner.

Re-runs Stage 3 acceptance scripts and reports a single PASS/FAIL.
Mirror of stage0_5_regression.py, stage1_regression.py, stage1_5_regression.py
shape, adapted to Stage 3 (universal domain ingestion).

Stage 3 has both QUICK sub-tasks (artifact smokes, calibration on existing
checkpoints) and LONG sub-tasks (full domain retraining). For routine
regression we run only the quick subset that proves the SHIPPED artifacts
still produce the closing-gate verdicts:

  Quick (default, ~20-30 min total):
    3.3  registry smoke (CPU, ~5 sec)
    3.5f factored output composition (~5 min GPU — uses decoder_3a1.pt)
    3.6  conformal calibration (~5-10 min GPU — uses decoder_3a1.pt)

  Full (--full, adds long sub-tasks, ~6-7 hr total):
    3.1  cross-domain validation (~4 hr GPU — retrains the definitional decoder)
    3.2  per-domain energy model (~1 hr GPU — retrains energy models)
    3.4  orchestrator smoke (~30-45 min GPU)

The quick subset is the right gate for "did anything regress?" because
3.5f and 3.6 both load decoder_3a1.pt (Stage 3.1's shipped artifact)
and 3.5f uses the energy models indirectly via the orchestrator
machinery. If 3.1's checkpoint is corrupted or the architecture
diverged, the quick subset catches it.

Stage 1.5 chain: by default ALSO invokes scripts/stage1_5_regression.py
at the end. Stage 3 doesn't close without proving it didn't regress
Stage 1.5 (and transitively 1, 0.5).

Acceptance gate (Sub-task 3.7 + Stage 3 close):
  All Stage 3 sub-tasks PASS in the chosen mode AND
  (if included) Stage 1.5 regression PASSES.

Run:
  python scripts/stage3_regression.py
  python scripts/stage3_regression.py --full              # include long retrains
  python scripts/stage3_regression.py --no-chain          # skip Stage 1.5 chain
  python scripts/stage3_regression.py --skip 3.6
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
    task_id: str
    label: str
    encoder: str | None
    cmd: list[str]
    rc: int
    verdict_line: str
    passed: bool
    raw_stdout: str = field(default="", repr=False)


# task_id, label, script, takes_encoder, is_long
WORKFLOWS_QUICK: list[tuple[str, str, str, bool, bool]] = [
    ("3.3",  "domain registry smoke",        "scripts/stage3_3_registry_smoke.py",       False, False),
    ("3.5f", "factored output composition",  "scripts/stage3_5f_factored_output.py",     True,  False),
    ("3.6",  "conformal calibration",        "scripts/stage3_6_conformal.py",            True,  False),
]

WORKFLOWS_LONG: list[tuple[str, str, str, bool, bool]] = [
    ("3.1", "cross-domain validation",       "scripts/stage3_1_definitional.py",         True,  True),
    ("3.2", "per-domain energy model",       "scripts/stage3_2_energy_model.py",         True,  True),
    ("3.4", "orchestrator smoke",            "scripts/stage3_4_orchestrator_smoke.py",   True,  True),
]


VERDICT_RE = re.compile(
    r"(?:→\s*Stage\s*3(?:\.\d+[a-z]?)?:?|→\s*STAGE_3(?:_\d+[a-zA-Z]?)?_(PASS|FAIL))",
    re.IGNORECASE,
)
VERDICT_LINE_RE = re.compile(
    r"→\s*(?:Stage\s*3[\.\d]*[a-zA-Z]?\s*:|STAGE_3[\.\d]*[a-zA-Z]?_)\s*(PASS|FAIL)",
    re.IGNORECASE,
)


def extract_verdict_line(stdout: str) -> str:
    """Return the last verdict line in stdout. Looks for either:
      "→ Stage 3.X: PASS" / "→ Stage 3.X: FAIL"
      "→ STAGE_3_X_PASS" / "→ STAGE_3_X_FAIL"
    """
    matches = list(VERDICT_LINE_RE.finditer(stdout))
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
    print(f"Sub-task {task_id}{enc_str}: {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'─' * 74}")
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0 and proc.stderr:
        print(f"[stderr]\n{proc.stderr}", end="")
    return WorkflowResult(
        task_id=task_id,
        label=label,
        encoder=encoder,
        cmd=cmd,
        rc=proc.returncode,
        verdict_line=extract_verdict_line(proc.stdout),
        passed=(proc.returncode == 0),
        raw_stdout=proc.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        "--encoder", default="e5-large-v2",
        help="Encoder for encoder-bearing scripts. Default: e5-large-v2.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Include long retraining sub-tasks (3.1, 3.2, 3.4). "
             "Default off — quick mode (3.3, 3.5f, 3.6) is the routine gate.",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Sub-task IDs to skip (e.g. --skip 3.6).",
    )
    parser.add_argument(
        "--chain", dest="chain", action="store_true", default=True,
        help="At the end, also run scripts/stage1_5_regression.py (default on).",
    )
    parser.add_argument(
        "--no-chain", dest="chain", action="store_false",
        help="Skip the chained Stage 1.5 regression.",
    )
    parser.add_argument(
        "--out", default="results/stage3/regression.json",
    )
    args = parser.parse_args()

    workflows = list(WORKFLOWS_QUICK)
    if args.full:
        workflows = list(WORKFLOWS_LONG) + workflows
    skip = set(args.skip or [])

    print("Sub-task 3.7 — Stage 3 closing regression")
    print("=" * 74)
    print(f"Mode: {'full (long retrains included)' if args.full else 'quick'}")
    print(f"Encoder: {args.encoder}")
    if skip:
        print(f"Skipping: {sorted(skip)}")
    if args.chain:
        print("Will chain into Stage 1.5 regression after Stage 3 sub-tasks.")
    else:
        print("Stage 1.5 regression chain: SKIPPED.")

    all_results: list[WorkflowResult] = []
    for task_id, label, script, takes_enc, _is_long in workflows:
        if task_id in skip:
            continue
        enc = args.encoder if takes_enc else None
        all_results.append(run_one(task_id, label, script, enc))

    # ---- Optional chain into Stage 1.5 regression -------------------
    chained: WorkflowResult | None = None
    if args.chain:
        print(f"\n{'═' * 74}")
        print(f"Chained: Stage 1.5 regression on {args.encoder}")
        print(f"{'═' * 74}")
        cmd = [
            sys.executable, "scripts/stage1_5_regression.py",
            "--encoder", args.encoder,
        ]
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0 and proc.stderr:
            print(f"[stderr]\n{proc.stderr}", end="")
        # Stage 1.5 regression uses its own verdict format ("→ Task 1.5.7: PASS")
        m = re.search(r"→\s*Task\s*1\.5\.7[^:]*:\s*(PASS|FAIL)", proc.stdout)
        verdict_str = m.group(0).strip() if m else ""
        chained = WorkflowResult(
            task_id="1.5.7",
            label="Stage 1.5 closing regression (chained)",
            encoder=args.encoder,
            cmd=cmd,
            rc=proc.returncode,
            verdict_line=verdict_str,
            passed=(proc.returncode == 0),
            raw_stdout=proc.stdout,
        )

    # ---- Summary ----------------------------------------------------
    print("\n" + "=" * 74)
    print("STAGE 3 REGRESSION SUMMARY")
    print("=" * 74)
    print(f"  {'task':>5}  {'encoder':<14}  {'label':<32}  status   verdict")
    print("  " + "-" * 86)
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        enc_cell = r.encoder if r.encoder else "—"
        verdict_cell = r.verdict_line if r.verdict_line else f"(rc={r.rc}, no verdict line)"
        print(
            f"  {r.task_id:>5}  {enc_cell:<14}  {r.label:<32}  "
            f"{status:<7} {verdict_cell}"
        )
    if chained is not None:
        s = chained
        status = "PASS" if s.passed else "FAIL"
        verdict_cell = s.verdict_line if s.verdict_line else f"(rc={s.rc}, no verdict)"
        print(
            f"  {s.task_id:>5}  {s.encoder:<14}  {s.label:<32}  "
            f"{status:<7} {verdict_cell}"
        )

    overall = (
        bool(all_results)
        and all(r.passed for r in all_results)
        and (chained is None or chained.passed)
    )
    print(f"\n→ Sub-task 3.7 (Stage 3 closing gate): "
          f"{'PASS' if overall else 'FAIL'}")
    if not overall:
        print("\nFailures:")
        for r in all_results:
            if not r.passed:
                enc_str = f" [{r.encoder}]" if r.encoder else ""
                print(f"  - {r.task_id}{enc_str}: rc={r.rc}, "
                      f"verdict='{r.verdict_line or '(missing)'}'")
        if chained is not None and not chained.passed:
            print(f"  - Chained Stage 1.5 regression: rc={chained.rc}")

    # ---- Save JSON --------------------------------------------------
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "3.7",
        "mode": "full" if args.full else "quick",
        "encoder": args.encoder,
        "skipped": sorted(skip),
        "chain_stage1_5": args.chain,
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
        "chained_stage1_5_result": (
            {
                "task_id": chained.task_id,
                "encoder": chained.encoder,
                "rc": chained.rc,
                "verdict_line": chained.verdict_line,
                "passed": chained.passed,
            }
            if chained is not None else None
        ),
        "overall_pass": overall,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
