"""Task 1.5.5 — versioned concept registry with growth control smoke.

Validates the registry mechanism in five sub-cases. All in-memory:
no encoder load, no concept training — operators are tiny untrained
ConceptOperator instances with seeded random init, exercising the
storage / versioning / pruning / persistence machinery.

Sub-cases:

  1. Register + list + state_dict round-trip
     Register 3 concepts. list_active() returns all 3. get_op()
     returns operators that produce identical forward pass to the
     originals (state_dict integrity).

  2. LRU growth control
     Cap=3. Touch concepts in deterministic order, then register
     a 4th without touching the first. The least-recently-used
     concept is retired; remaining 3 are the most recent.

  3. Version bump
     Register concept "X" v2 over the existing v1. current_version
     becomes 2. Both versions' state_dicts persist on disk. get_op
     returns v2 weights.

  4. Rollback
     Rollback "X" to v1. current_version becomes 1. get_op now
     returns v1 weights, identical to the original v1 forward pass.

  5. Persistence (fresh registry instance)
     Open a SECOND ConceptRegistry pointing at the same root dir.
     All entries reload from disk with correct metadata. get_op on
     the reloaded registry produces the same forward pass as the
     original.

Acceptance gates (Task 1.5.5, all HARD):

  - All 5 sub-cases pass.
  - State_dict round-trip on every get_op: cos(orig(z), loaded(z))
    ≥ 0.9999 on the same random input.

No GPU, no torch CUDA — runs on CPU in seconds. Under tempfile dir
which is cleaned up at exit.

Run:
  python scripts/stage1_5_registry_smoke.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.concepts.operator import ConceptOperator
from selflearnai.discovery import ConceptRegistry


# Small dim — registry mechanism is dim-agnostic; smoke uses a tiny
# vector to keep state_dict files small and tests fast.
DIM = 16
COS_ROUND_TRIP_MIN = 0.9999


def make_op(seed: int) -> ConceptOperator:
    """Untrained ConceptOperator with seeded random init."""
    torch.manual_seed(seed)
    op = ConceptOperator(dim=DIM)
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op


def forward_signature(op: ConceptOperator, z: torch.Tensor) -> torch.Tensor:
    """Deterministic forward pass for round-trip equality checks."""
    with torch.no_grad():
        return op(z)


def cos_one_to_one(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine over batch dimension."""
    return float(F.cosine_similarity(a, b, dim=-1).mean().item())


def _bump_lru_apart() -> None:
    """Microsecond timestamps already keep _now_iso strictly ordered, but
    sleep an extra 1ms between actions just to be unambiguous in the
    smoke output (timestamps in the JSON visibly differ)."""
    time.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep registry root for inspection.")
    parser.add_argument("--out", default="results/stage1_5/registry_smoke.json")
    args = parser.parse_args()

    print("Task 1.5.5 — versioned concept registry smoke")
    print("=" * 78)

    tmp = Path(tempfile.mkdtemp(prefix="registry_smoke_"))
    print(f"Registry root: {tmp}")

    # Test probe: random tensor used for forward-pass equality checks.
    torch.manual_seed(12345)
    probe = torch.randn(4, DIM)

    sub_records: dict[str, dict] = {}

    # =====================================================================
    # Sub-case 1: Register + list + state_dict round-trip
    # =====================================================================
    print("\n[1] Register + list + state_dict round-trip")
    print("-" * 78)
    reg = ConceptRegistry(tmp, dim=DIM, max_active=3)
    originals: dict[str, ConceptOperator] = {}
    sigs_orig: dict[str, torch.Tensor] = {}
    for i, cid in enumerate(["alpha", "beta", "gamma"]):
        op = make_op(seed=10 + i)
        originals[cid] = op
        sigs_orig[cid] = forward_signature(op, probe)
        reg.register(cid, op, provenance={"source": "smoke", "i": i}, support_count=10 + i)
        _bump_lru_apart()

    listing = reg.list_active()
    list_ids = sorted([e.concept_id for e in listing])
    list_pass = list_ids == ["alpha", "beta", "gamma"] and len(listing) == 3
    print(f"  list_active ids = {list_ids}   "
          f"{'PASS' if list_pass else 'FAIL'}")

    rt_min_cos = float("inf")
    rt_pass_all = True
    for cid in ["alpha", "beta", "gamma"]:
        loaded = reg.get_op(cid)
        sig_loaded = forward_signature(loaded, probe)
        c = cos_one_to_one(sigs_orig[cid], sig_loaded)
        rt_min_cos = min(rt_min_cos, c)
        ok = c >= COS_ROUND_TRIP_MIN
        rt_pass_all = rt_pass_all and ok
        print(f"  round-trip {cid}: cos={c:.6f}  "
              f"{'PASS' if ok else 'FAIL (need ≥ %.4f)' % COS_ROUND_TRIP_MIN}")
    sub_pass_1 = list_pass and rt_pass_all
    sub_records["1_register_and_round_trip"] = {
        "list_ids": list_ids,
        "round_trip_min_cos": rt_min_cos,
        "pass": sub_pass_1,
    }

    # =====================================================================
    # Sub-case 2: LRU growth control
    # =====================================================================
    print("\n[2] LRU growth control (cap=3, register 4th)")
    print("-" * 78)
    # Touch alpha and beta so they're more-recent than gamma.
    # gamma is the LRU when delta is registered.
    _bump_lru_apart()
    _ = reg.get_op("alpha")
    _bump_lru_apart()
    _ = reg.get_op("beta")
    # gamma is now the least-recently-used.
    _bump_lru_apart()
    op_delta = make_op(seed=99)
    sig_delta_orig = forward_signature(op_delta, probe)
    reg.register("delta", op_delta, provenance={"source": "smoke", "step": "lru_test"})

    after_listing = reg.list_active()
    after_ids = sorted([e.concept_id for e in after_listing])
    expected_ids = ["alpha", "beta", "delta"]   # gamma should be retired
    lru_pass = after_ids == expected_ids and len(after_listing) == 3
    archived_path = tmp / "archive" / "gamma.json"
    archive_pass = archived_path.exists()
    print(f"  active after register=delta: {after_ids}   "
          f"{'PASS' if lru_pass else 'FAIL (expected %s)' % expected_ids}")
    print(f"  retired entry archived to disk: gamma.json   "
          f"{'PASS' if archive_pass else 'FAIL'}")
    # delta operator must round-trip.
    sig_delta_loaded = forward_signature(reg.get_op("delta"), probe)
    delta_cos = cos_one_to_one(sig_delta_orig, sig_delta_loaded)
    delta_rt_pass = delta_cos >= COS_ROUND_TRIP_MIN
    print(f"  delta round-trip cos={delta_cos:.6f}   "
          f"{'PASS' if delta_rt_pass else 'FAIL'}")
    sub_pass_2 = lru_pass and archive_pass and delta_rt_pass
    sub_records["2_lru_prune"] = {
        "active_after": after_ids,
        "retired_archived": archive_pass,
        "delta_round_trip_cos": delta_cos,
        "pass": sub_pass_2,
    }

    # =====================================================================
    # Sub-case 3: Version bump
    # =====================================================================
    print("\n[3] Version bump on existing concept")
    print("-" * 78)
    op_alpha_v2 = make_op(seed=42)
    sig_alpha_v2 = forward_signature(op_alpha_v2, probe)
    _bump_lru_apart()
    reg.bump_version("alpha", op_alpha_v2,
                     provenance={"source": "smoke", "step": "v2"},
                     support_count=20)
    entry_alpha = reg.get_entry("alpha")
    bump_version_pass = (
        entry_alpha.current_version == 2
        and len(entry_alpha.versions) == 2
        and {v.version for v in entry_alpha.versions} == {1, 2}
    )
    print(f"  alpha.versions = {[v.version for v in entry_alpha.versions]}   "
          f"current = {entry_alpha.current_version}   "
          f"{'PASS' if bump_version_pass else 'FAIL'}")

    # get_op should return v2 weights now.
    sig_alpha_current = forward_signature(reg.get_op("alpha"), probe)
    bump_load_cos = cos_one_to_one(sig_alpha_v2, sig_alpha_current)
    bump_v2_pass = bump_load_cos >= COS_ROUND_TRIP_MIN
    print(f"  get_op('alpha') returns v2 weights: cos to v2 fixture = "
          f"{bump_load_cos:.6f}   {'PASS' if bump_v2_pass else 'FAIL'}")
    sub_pass_3 = bump_version_pass and bump_v2_pass
    sub_records["3_bump_version"] = {
        "current_version": entry_alpha.current_version,
        "versions": [v.version for v in entry_alpha.versions],
        "v2_round_trip_cos": bump_load_cos,
        "pass": sub_pass_3,
    }

    # =====================================================================
    # Sub-case 4: Rollback
    # =====================================================================
    print("\n[4] Rollback to previous version")
    print("-" * 78)
    sig_alpha_v1_orig = sigs_orig["alpha"]   # the very first registered op
    _bump_lru_apart()
    reg.rollback("alpha", target_version=1)
    entry_alpha = reg.get_entry("alpha")
    rollback_state_pass = (
        entry_alpha.current_version == 1
        and len(entry_alpha.versions) == 2  # both versions still on file
    )
    sig_alpha_after_rollback = forward_signature(reg.get_op("alpha"), probe)
    rollback_load_cos = cos_one_to_one(sig_alpha_v1_orig, sig_alpha_after_rollback)
    rollback_load_pass = rollback_load_cos >= COS_ROUND_TRIP_MIN
    print(f"  current_version after rollback = "
          f"{entry_alpha.current_version}   "
          f"{'PASS' if rollback_state_pass else 'FAIL'}")
    print(f"  get_op('alpha') returns v1 weights: cos to v1 fixture = "
          f"{rollback_load_cos:.6f}   "
          f"{'PASS' if rollback_load_pass else 'FAIL'}")
    sub_pass_4 = rollback_state_pass and rollback_load_pass
    sub_records["4_rollback"] = {
        "current_version_after_rollback": entry_alpha.current_version,
        "v1_round_trip_cos": rollback_load_cos,
        "pass": sub_pass_4,
    }

    # =====================================================================
    # Sub-case 5: Persistence — fresh registry instance reloads state
    # =====================================================================
    print("\n[5] Persistence (fresh registry, same root)")
    print("-" * 78)
    reg2 = ConceptRegistry(tmp, dim=DIM, max_active=3)
    fresh_listing = reg2.list_active()
    fresh_ids = sorted([e.concept_id for e in fresh_listing])
    persist_ids_pass = fresh_ids == ["alpha", "beta", "delta"]
    print(f"  fresh registry list_active: {fresh_ids}   "
          f"{'PASS' if persist_ids_pass else 'FAIL'}")
    fresh_alpha = reg2.get_entry("alpha")
    persist_versions_pass = (
        fresh_alpha.current_version == 1
        and {v.version for v in fresh_alpha.versions} == {1, 2}
    )
    print(f"  alpha versions reloaded: "
          f"{[v.version for v in fresh_alpha.versions]}, "
          f"current={fresh_alpha.current_version}   "
          f"{'PASS' if persist_versions_pass else 'FAIL'}")
    sig_alpha_fresh = forward_signature(reg2.get_op("alpha"), probe)
    persist_load_cos = cos_one_to_one(sig_alpha_v1_orig, sig_alpha_fresh)
    persist_load_pass = persist_load_cos >= COS_ROUND_TRIP_MIN
    print(f"  fresh get_op('alpha') round-trip cos: {persist_load_cos:.6f}   "
          f"{'PASS' if persist_load_pass else 'FAIL'}")
    sub_pass_5 = persist_ids_pass and persist_versions_pass and persist_load_pass
    sub_records["5_persistence"] = {
        "fresh_ids": fresh_ids,
        "fresh_alpha_current": fresh_alpha.current_version,
        "fresh_alpha_versions": [v.version for v in fresh_alpha.versions],
        "round_trip_cos": persist_load_cos,
        "pass": sub_pass_5,
    }

    # =====================================================================
    # Acceptance
    # =====================================================================
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.5.5)")
    print("=" * 78)
    sub_pass_flags = [sub_pass_1, sub_pass_2, sub_pass_3, sub_pass_4, sub_pass_5]
    sub_labels = [
        "Register + list + state_dict round-trip",
        "LRU growth control (cap=3)",
        "Version bump",
        "Rollback",
        "Persistence (fresh registry instance)",
    ]
    for label, ok in zip(sub_labels, sub_pass_flags):
        print(f"  {label:<48s}  {'PASS' if ok else 'FAIL'}")
    overall = all(sub_pass_flags)
    print(f"\n→ Task 1.5.5: {'PASS' if overall else 'FAIL'}")

    payload = {
        "task": "1.5.5",
        "registry_root": str(tmp),
        "dim": DIM,
        "round_trip_min_cos_threshold": COS_ROUND_TRIP_MIN,
        "sub_cases": sub_records,
        "pass": overall,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")

    if not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"→ cleaned up registry root {tmp}")
    else:
        print(f"→ kept registry root at {tmp} (--keep-tmp)")

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
