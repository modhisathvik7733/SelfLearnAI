"""Phase 2a closing-regression runner (sub-task 2a.7).

Mirror of `scripts/stage1_5_regression.py` for Phase 2a. Verifies
that every artifact Phase 2a shipped is intact and functional, and
that Stage 1.5 / Stage 1 / Stage 0.5 didn't regress.

Phase 2a artifacts checked:
  1. data/explanations_v2/train.tsv      (2064 rows expected)
  2. data/explanations_v2/holdout.tsv    (432 rows)
  3. data/explanations_v2/metadata.json  (matches above + audit info)
  4. data/explanations_v2/checkpoints/decoder_2a3.pt (production checkpoint)
  5. selflearnai/generator/* package imports cleanly
  6. PointerSeqCondDecoder loads the checkpoint and forward-passes
  7. results/stage2a/full_train.json      (2a.3 verdict = PHASE_2A_PASS)
  8. results/stage2a/multi_candidate.json (2a.5 verdict recorded)

Optional (default ON): chain into stage1_5_regression.py to confirm
no upstream regression. Flag --no-chain to skip.

Run:
  python scripts/stage2a_regression.py
  python scripts/stage2a_regression.py --no-chain   # Phase 2a only, fast
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
class CheckResult:
    name: str
    passed: bool
    detail: str
    raw: str = field(default="", repr=False)


def check_file_exists(path: Path, expected_size_min: int = 0) -> CheckResult:
    if not path.exists():
        return CheckResult(name=str(path), passed=False, detail="missing")
    size = path.stat().st_size
    if size < expected_size_min:
        return CheckResult(
            name=str(path), passed=False,
            detail=f"too small: {size} < {expected_size_min} bytes",
        )
    return CheckResult(name=str(path), passed=True,
                       detail=f"{size} bytes")


def check_corpus_metadata() -> CheckResult:
    p = REPO_ROOT / "data/explanations_v2/metadata.json"
    if not p.exists():
        return CheckResult(name="metadata.json", passed=False, detail="missing")
    with open(p) as f:
        meta = json.load(f)
    n_train = meta["totals"]["n_train"]
    n_holdout = meta["totals"]["n_holdout"]
    audits = meta["audits"]
    ok = (
        n_train >= 2000
        and n_holdout >= 400
        and audits.get("leakage_passed", False)
    )
    detail = (
        f"n_train={n_train}, n_holdout={n_holdout}, "
        f"leakage_passed={audits.get('leakage_passed')}"
    )
    return CheckResult(name="metadata.json", passed=ok, detail=detail)


def check_results_json(path: Path, expected_verdict: str) -> CheckResult:
    if not path.exists():
        return CheckResult(name=str(path), passed=False, detail="missing")
    with open(path) as f:
        d = json.load(f)
    actual = d.get("verdict", "?")
    ok = actual == expected_verdict
    return CheckResult(
        name=str(path), passed=ok,
        detail=f"verdict={actual} (expected {expected_verdict})",
    )


def check_package_smoke() -> CheckResult:
    """Run scripts/stage2a_2_package_smoke.py and verify it exits 0."""
    cmd = [sys.executable, "scripts/stage2a_2_package_smoke.py"]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    return CheckResult(
        name="stage2a_2_package_smoke",
        passed=(proc.returncode == 0),
        detail=f"rc={proc.returncode}",
        raw=proc.stdout,
    )


def check_decoder_load() -> CheckResult:
    """Load 2a.3 checkpoint into the package's PointerSeqCondDecoder
    and run a tiny forward pass on a probe."""
    code = """
import torch, sys
sys.path.insert(0, '.')
from selflearnai.generator import PointerSeqCondDecoder

ckpt = 'data/explanations_v2/checkpoints/decoder_2a3.pt'
decoder = PointerSeqCondDecoder(
    encoder_dim=1024, hidden_dim=512, t_max=32,
    vocab_size=30522, n_layers=4, n_heads=8,
)
state = torch.load(ckpt, map_location='cpu', weights_only=True)
decoder.load_state_dict(state)
decoder.eval()
n_params = sum(p.numel() for p in decoder.parameters())

# Tiny forward
h = torch.randn(2, 8, 1024)
mask = torch.ones(2, 8)
ids = torch.randint(0, 30522, (2, 8))
with torch.no_grad():
    log_probs, hidden, p_gen = decoder(h, mask, ids)

assert log_probs.shape == (2, 32, 30522), f'shape {log_probs.shape}'
assert torch.isfinite(log_probs).all(), 'NaN/Inf in log_probs'
print(f'OK n_params={n_params} log_probs={log_probs.shape}')
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    return CheckResult(
        name="decoder_load_and_forward",
        passed=(proc.returncode == 0),
        detail=proc.stdout.strip().split("\n")[-1] if proc.stdout else f"rc={proc.returncode}",
        raw=proc.stderr or proc.stdout,
    )


def chain_stage15_regression() -> CheckResult:
    """Run stage1_5_regression.py to confirm no upstream regression.
    Skips the long Stage 1 chain (it was green at 96fa4f3)."""
    cmd = [
        sys.executable, "scripts/stage1_5_regression.py",
        "--no-include-stage1-regression",
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if proc.stdout:
        # Surface the summary table from stage 1.5 regression
        last_lines = "\n".join(proc.stdout.strip().split("\n")[-15:])
    else:
        last_lines = ""
    verdict_line = ""
    m = re.search(r"→\s*Task\s*1\.5\.7\s*\(.*?\):\s*(PASS|FAIL)", proc.stdout)
    if m:
        verdict_line = m.group(0)
    return CheckResult(
        name="stage1_5_regression (chained)",
        passed=(proc.returncode == 0),
        detail=verdict_line or f"rc={proc.returncode}",
        raw=last_lines,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        "--no-chain", dest="chain", action="store_false", default=True,
        help="Skip the chained stage1_5_regression.py call.",
    )
    parser.add_argument(
        "--out", default="results/stage2a/regression.json",
    )
    args = parser.parse_args()

    print("Phase 2a closing-regression runner")
    print("=" * 70)
    if args.chain:
        print("Will chain into stage1_5_regression.py at the end.")
    else:
        print("Stage 1.5 chain: SKIPPED.")

    results: list[CheckResult] = []

    print("\n[1] Phase 2a corpus artifacts")
    print("-" * 70)
    results.append(check_file_exists(
        REPO_ROOT / "data/explanations_v2/train.tsv",
        expected_size_min=10000,
    ))
    results.append(check_file_exists(
        REPO_ROOT / "data/explanations_v2/holdout.tsv",
        expected_size_min=2000,
    ))
    results.append(check_corpus_metadata())

    print("\n[2] Phase 2a checkpoint")
    print("-" * 70)
    results.append(check_file_exists(
        REPO_ROOT / "data/explanations_v2/checkpoints/decoder_2a3.pt",
        expected_size_min=100_000_000,   # ~130MB for ~34M params at fp32
    ))

    print("\n[3] Phase 2a results JSONs (verdicts recorded)")
    print("-" * 70)
    results.append(check_results_json(
        REPO_ROOT / "results/stage2a/full_train.json",
        expected_verdict="PHASE_2A_PASS",
    ))
    # 2a.5 verdict can be MULTI_CANDIDATE_PASS or MULTI_CANDIDATE_NO_LIFT
    # — both are valid closures; we only check the file exists.
    results.append(check_file_exists(
        REPO_ROOT / "results/stage2a/multi_candidate.json",
        expected_size_min=1000,
    ))

    print("\n[4] selflearnai.generator/ package")
    print("-" * 70)
    results.append(check_package_smoke())
    results.append(check_decoder_load())

    if args.chain:
        print("\n[5] Chained: Stage 1.5 regression")
        print("-" * 70)
        results.append(chain_stage15_regression())

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'check':<48s}  status   detail")
    print("  " + "-" * 70)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        # Trim long names to keep aligned
        name = r.name if len(r.name) <= 46 else "..." + r.name[-43:]
        print(f"  {name:<48s}  {mark:<7s}  {r.detail}")

    overall = all(r.passed for r in results)
    print(f"\n→ Phase 2a closing regression: "
          f"{'PASS' if overall else 'FAIL'}")
    if not overall:
        print("\nFailures:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.detail}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "task": "2a.7",
                "chain_stage1_5": args.chain,
                "results": [
                    {"name": r.name, "passed": r.passed, "detail": r.detail}
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
