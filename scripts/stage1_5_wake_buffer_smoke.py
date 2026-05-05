"""Task 1.5.1 — wake replay buffer smoke.

Validates the wake-buffer mechanism without requiring an encoder load
or a GPU. The buffer is a structural piece (PsiProgram + envelope +
on-disk JSON), so the smoke is fully synthetic:

  1. Build 4 synthetic PsiPrograms covering the four expected
     channels (success, failed, inefficient, low_confidence).
  2. Route each via `route_program` and verify the channel decision
     matches the expected channel — the precedence rules are
     exercised end-to-end.
  3. Append all four to a fresh WakeBuffer on disk.
  4. Read them back via `WakeBuffer.read()` and `read(channel=...)`.
  5. Field-level equality check: the reloaded BufferEntry must equal
     the original (program fields, channel, routing_reason,
     routing_metadata, entry_id, appended_at).
  6. Channel size accounting: `size_all()` returns 1 per channel.

Acceptance gate (Task 1.5.1):
  - 4/4 routing decisions match expected channel (HARD).
  - 4/4 reloaded entries field-equal the originals (HARD).
  - size_all() == {success: 1, failed: 1, inefficient: 1,
                   low_confidence: 1} (HARD).

No GPU. No encoder. Smoke runs in seconds and the JSON files written
under the temp dir are removed at the end.

Run:
  python scripts/stage1_5_wake_buffer_smoke.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selflearnai.discovery import (
    BufferEntry,
    CHANNELS,
    WakeBuffer,
    route_program,
)
from selflearnai.planner import PsiProgram, PsiProgramStep


# ---------------------------------------------------------------------------
# Synthetic PsiProgram fixtures
# ---------------------------------------------------------------------------

def make_synthetic_program(
    *,
    source: str,
    target: str,
    chain: list[str],
    final_top1_word: str | None,
    final_cos_to_goal: float | None,
) -> PsiProgram:
    """Cheap PsiProgram with 8-dim embeddings — small enough to inspect
    in JSON, no encoder dependency. Each step's psi_after is a
    deterministic function of (source, target, op_name) so two different
    cases produce visibly different traces."""
    dim = 8
    psi_initial = [float((ord(source[0]) + i) % 7) / 7.0 for i in range(dim)]
    psi_goal = [float((ord(target[0]) + i) % 5) / 5.0 for i in range(dim)]
    steps: list[PsiProgramStep] = []
    for i, op_name in enumerate(chain):
        psi_after = [
            float((ord(source[0]) + ord(op_name[0]) + i + j) % 11) / 11.0
            for j in range(dim)
        ]
        steps.append(PsiProgramStep(
            step_index=i,
            op_name=op_name,
            psi_after=psi_after,
            cos_to_goal_after=0.7 + 0.05 * i,
            verification={"all_passed": True, "gates": {}},
        ))
    return PsiProgram(
        source=source,
        target=target,
        encoder_name="synthetic",
        encoder_dim=dim,
        psi_initial=psi_initial,
        psi_goal=psi_goal,
        chain=chain,
        steps=steps,
        final_top1_word=final_top1_word,
        final_cos_to_goal=final_cos_to_goal,
        timestamp=PsiProgram.now_timestamp(),
        metadata={"task": "1.5.1", "synthetic": True},
    )


# ---------------------------------------------------------------------------
# Routing case definitions
# ---------------------------------------------------------------------------

CASES = [
    {
        "name": "success",
        "expected_channel": "success",
        "program_kwargs": dict(
            source="paint", target="painters",
            chain=["agentive", "plural"],
            final_top1_word="painters",
            final_cos_to_goal=0.92,
        ),
        "route_kwargs": dict(
            reached_goal=True,
            min_known_depth=2,
            coverage_width=0.05,
        ),
    },
    {
        "name": "failed",
        "expected_channel": "failed",
        "program_kwargs": dict(
            source="quobble", target="zorps",
            chain=["plural"],
            final_top1_word=None,
            final_cos_to_goal=0.31,
        ),
        "route_kwargs": dict(
            reached_goal=False,
        ),
    },
    {
        "name": "inefficient",
        "expected_channel": "inefficient",
        "program_kwargs": dict(
            # Goal reached but with a longer chain than the known minimum.
            source="drive", target="drivers",
            chain=["agentive", "plural", "agentive"],
            final_top1_word="drivers",
            final_cos_to_goal=0.88,
        ),
        "route_kwargs": dict(
            reached_goal=True,
            min_known_depth=2,           # known minimum
            coverage_width=0.05,
        ),
    },
    {
        "name": "low_confidence",
        "expected_channel": "low_confidence",
        "program_kwargs": dict(
            source="cook", target="cooks",
            chain=["plural"],
            final_top1_word="cooks",
            final_cos_to_goal=0.84,
        ),
        "route_kwargs": dict(
            reached_goal=True,
            min_known_depth=1,
            coverage_width=0.35,         # above default 0.20
        ),
    },
]


# ---------------------------------------------------------------------------
# Field-level equality check
# ---------------------------------------------------------------------------

def entries_equal(a: BufferEntry, b: BufferEntry) -> tuple[bool, list[str]]:
    """Strict field-level comparison; returns (ok, diffs)."""
    diffs: list[str] = []
    if a.entry_id != b.entry_id:
        diffs.append(f"entry_id {a.entry_id!r} != {b.entry_id!r}")
    if a.channel != b.channel:
        diffs.append(f"channel {a.channel!r} != {b.channel!r}")
    if a.routing_reason != b.routing_reason:
        diffs.append("routing_reason mismatch")
    if a.routing_metadata != b.routing_metadata:
        diffs.append("routing_metadata mismatch")
    if a.appended_at != b.appended_at:
        diffs.append(f"appended_at {a.appended_at!r} != {b.appended_at!r}")
    # Program field-level (subset of fields that matter for sleep mining):
    pa, pb = a.program, b.program
    for field in (
        "source", "target", "encoder_name", "encoder_dim",
        "psi_initial", "psi_goal", "chain", "final_top1_word",
        "final_cos_to_goal", "timestamp", "metadata",
    ):
        if getattr(pa, field) != getattr(pb, field):
            diffs.append(f"program.{field} mismatch")
    if len(pa.steps) != len(pb.steps):
        diffs.append(f"program.steps len {len(pa.steps)} != {len(pb.steps)}")
    else:
        for i, (sa, sb) in enumerate(zip(pa.steps, pb.steps)):
            if (
                sa.step_index != sb.step_index
                or sa.op_name != sb.op_name
                or sa.psi_after != sb.psi_after
                or sa.cos_to_goal_after != sb.cos_to_goal_after
                or sa.verification != sb.verification
            ):
                diffs.append(f"program.steps[{i}] mismatch")
    return (not diffs, diffs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument(
        "--keep-tmp", action="store_true",
        help="Keep the temp buffer dir for inspection (default: cleanup).",
    )
    parser.add_argument(
        "--out", default="results/stage1_5/wake_buffer_smoke.json",
    )
    args = parser.parse_args()

    print("Task 1.5.1 — wake replay buffer smoke")
    print("=" * 78)

    tmp_dir = Path(tempfile.mkdtemp(prefix="wake_buffer_smoke_"))
    print(f"Buffer root: {tmp_dir}")
    buf = WakeBuffer(tmp_dir)

    # -- 1+2: build programs, route them, and check routing decisions --
    print("\n[1] Routing decisions")
    print("-" * 78)
    n_route_pass = 0
    originals: list[BufferEntry] = []
    case_records: list[dict] = []
    for case in CASES:
        program = make_synthetic_program(**case["program_kwargs"])
        channel, reason, meta = route_program(program, **case["route_kwargs"])
        ok = channel == case["expected_channel"]
        if ok:
            n_route_pass += 1
        mark = "✓" if ok else "✗"
        print(
            f"  {mark} {case['name']:<16}  routed -> {channel:<14} "
            f"(expected {case['expected_channel']})"
        )
        if not ok:
            print(f"      reason: {reason}")
            print(f"      meta: {meta}")
        # -- 3: append --
        entry = buf.append(
            program,
            channel=channel,
            routing_reason=reason,
            routing_metadata=meta,
        )
        originals.append(entry)
        case_records.append({
            "name": case["name"],
            "expected_channel": case["expected_channel"],
            "actual_channel": channel,
            "routed_correctly": ok,
            "routing_reason": reason,
            "routing_metadata": meta,
            "entry_id": entry.entry_id,
        })

    # -- 4: read back and compare --
    print("\n[2] Round-trip (read + field-level equality)")
    print("-" * 78)
    reloaded_all = buf.read()
    by_id = {e.entry_id: e for e in reloaded_all}
    n_eq_pass = 0
    for orig in originals:
        if orig.entry_id not in by_id:
            print(f"  ✗ {orig.channel:<14} entry_id {orig.entry_id} missing on reload")
            continue
        ok, diffs = entries_equal(orig, by_id[orig.entry_id])
        if ok:
            n_eq_pass += 1
            print(f"  ✓ {orig.channel:<14} entry_id {orig.entry_id} round-trip equal")
        else:
            print(f"  ✗ {orig.channel:<14} entry_id {orig.entry_id} diffs:")
            for d in diffs:
                print(f"      - {d}")

    # -- 5: per-channel read + size_all --
    print("\n[3] Per-channel size accounting")
    print("-" * 78)
    sizes = buf.size_all()
    expected_sizes = {ch: 1 for ch in CHANNELS}
    sizes_match = sizes == expected_sizes
    for ch in CHANNELS:
        per_ch = buf.read(channel=ch)
        mark = "✓" if sizes[ch] == 1 and len(per_ch) == 1 else "✗"
        print(f"  {mark} {ch:<14} size={sizes[ch]} read(channel={ch})={len(per_ch)}")

    # -- Acceptance --
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.5.1)")
    print("=" * 78)
    n_cases = len(CASES)
    route_ok = n_route_pass == n_cases
    eq_ok = n_eq_pass == n_cases
    print(
        f"  Routing decisions: {n_route_pass}/{n_cases} "
        f"(HARD: all)  → {'PASS' if route_ok else 'FAIL'}"
    )
    print(
        f"  Round-trip equality: {n_eq_pass}/{n_cases} "
        f"(HARD: all)  → {'PASS' if eq_ok else 'FAIL'}"
    )
    print(
        f"  size_all() == {{ch: 1 for ch in CHANNELS}}: "
        f"{sizes}  → {'PASS' if sizes_match else 'FAIL'}"
    )
    overall = route_ok and eq_ok and sizes_match
    print(f"\n→ Task 1.5.1: {'PASS' if overall else 'FAIL'}")

    # -- Save summary JSON --
    payload = {
        "task": "1.5.1",
        "buffer_root": str(tmp_dir),
        "n_cases": n_cases,
        "n_route_pass": n_route_pass,
        "n_eq_pass": n_eq_pass,
        "sizes": sizes,
        "expected_sizes": expected_sizes,
        "cases": case_records,
        "pass": overall,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved summary JSON to {out_path}")

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"→ cleaned up temp buffer dir {tmp_dir}")
    else:
        print(f"→ kept temp buffer dir at {tmp_dir} (--keep-tmp)")

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
