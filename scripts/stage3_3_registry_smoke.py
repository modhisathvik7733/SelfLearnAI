"""Stage 3 / Sub-task 3.3 — domain registry smoke test.

Validates the DomainRegistry mechanism in 5 sub-cases. CPU-only,
no encoder, ~5 seconds. Mirror of scripts/stage1_5_registry_smoke.py
which validated Stage 1.5's ConceptRegistry — same pattern, applied
to domains instead of concepts.

Sub-cases (all in tempfile dir, cleaned up at exit):

  1. Register + list + artifact stash
     Register 3 domains with synthetic decoder + energy artifacts.
     list_active() returns all 3. Each artifact dir contains
     decoder.pt + energy_*.pt + provenance.json.

  2. LRU growth control
     Cap = 3. Touch domains in deterministic order, then register
     a 4th without touching the first. The least-recently-used
     domain is retired; the 3 most recent remain.

  3. Version bump
     Register domain "alpha" v2 (different decoder ckpt). v1 stays
     archived; v2 is current. get_entry()→current() returns v2's
     paths.

  4. Rollback
     Rollback "alpha" to v1. current_version becomes 1. Both
     versions' artifact dirs are still on disk (artifacts/alpha__v1/
     and artifacts/alpha__v2/).

  5. Persistence (fresh registry instance)
     Open a SECOND DomainRegistry pointing at the same root.
     All entries reload from disk with correct metadata. get_entry
     on the reloaded registry returns identical paths.

Acceptance: all 5 sub-cases pass.

Run:
  python scripts/stage3_3_registry_smoke.py
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

from selflearnai.domain import DomainRegistry


def make_dummy_artifact(path: Path, label: str) -> None:
    """Write a tiny torch state-dict so the registry has something
    to copy. Real artifacts (decoder, energy) are exactly this shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "DummyArtifact",
            "label": label,
            "data": torch.randn(8),
        },
        path,
    )


def check(label: str, ok: bool, *, fatal: bool = True) -> None:
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}")
    if not ok and fatal:
        raise SystemExit(1)


def _bump_lru() -> None:
    """1ms sleep to keep timestamps unambiguous in the JSON output."""
    time.sleep(0.001)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep registry root for inspection (default: cleanup).")
    parser.add_argument("--out", default="results/stage3/registry_smoke.json")
    args = parser.parse_args()

    print("Stage 3 / Sub-task 3.3 — domain registry smoke")
    print("=" * 78)

    tmp = Path(tempfile.mkdtemp(prefix="domain_registry_smoke_"))
    print(f"Registry root: {tmp}")

    sub_records: dict[str, dict] = {}

    # =====================================================================
    # Sub-case 1: Register + list + artifact stash
    # =====================================================================
    print("\n[1] Register + list + artifact stash")
    print("-" * 78)
    reg = DomainRegistry(tmp, max_active=3)

    # Source artifact files (in a separate "src" dir to keep the
    # registry-internal copies clean).
    src = tmp / "src"
    src.mkdir(parents=True, exist_ok=True)

    expected_paths: dict[str, dict] = {}
    for cid, kind in [("alpha", "GaussianEnergy"),
                      ("beta",  "MLPEnergyModel"),
                      ("gamma", "GaussianEnergy")]:
        decoder_src = src / f"{cid}_decoder.pt"
        energy_src = src / f"{cid}_energy.pt"
        make_dummy_artifact(decoder_src, label=f"{cid}_decoder_v1")
        make_dummy_artifact(energy_src,  label=f"{cid}_energy_v1")
        reg.register(
            cid,
            decoder_src_path=decoder_src,
            energy_src_path=energy_src,
            energy_kind=kind,
            provenance={"source": "smoke", "domain": cid, "version": 1},
        )
        _bump_lru()
        # Confirm the registry copied the files into artifacts/<cid>__v1/
        artifact_dir = tmp / "artifacts" / f"{cid}__v1"
        expected_paths[cid] = {
            "artifact_dir": artifact_dir,
            "decoder_path": artifact_dir / "decoder.pt",
            "energy_path": artifact_dir / f"energy_{kind.lower()}.pt",
            "provenance_path": artifact_dir / "provenance.json",
        }

    listing = reg.list_active()
    list_ids = sorted([e.domain_id for e in listing])
    list_ok = list_ids == ["alpha", "beta", "gamma"] and len(listing) == 3
    check(f"list_active ids = {list_ids}", list_ok)

    # Verify artifact files were actually copied
    artifacts_ok = True
    for cid, paths in expected_paths.items():
        for kind, p in paths.items():
            if kind == "artifact_dir":
                continue
            if not p.exists():
                print(f"    missing: {p}")
                artifacts_ok = False
    check("artifacts stashed (decoder + energy + provenance per domain)",
          artifacts_ok)

    sub_records["1_register_and_stash"] = {
        "list_ids": list_ids, "pass": list_ok and artifacts_ok,
    }

    # =====================================================================
    # Sub-case 2: LRU growth control
    # =====================================================================
    print("\n[2] LRU growth control (cap=3, register 4th)")
    print("-" * 78)
    # Touch alpha + beta — gamma becomes LRU
    _bump_lru()
    _ = reg.get_entry("alpha")
    _bump_lru()
    _ = reg.get_entry("beta")
    _bump_lru()

    # Register delta
    decoder_d = src / "delta_decoder.pt"
    energy_d = src / "delta_energy.pt"
    make_dummy_artifact(decoder_d, label="delta_decoder_v1")
    make_dummy_artifact(energy_d,  label="delta_energy_v1")
    reg.register(
        "delta",
        decoder_src_path=decoder_d,
        energy_src_path=energy_d,
        energy_kind="MLPEnergyModel",
        provenance={"source": "smoke", "step": "lru_test"},
    )

    after = sorted([e.domain_id for e in reg.list_active()])
    expected_after = ["alpha", "beta", "delta"]
    lru_ok = after == expected_after
    archived_path = tmp / "archive" / "gamma.json"
    archive_ok = archived_path.exists()
    check(f"active after register=delta: {after}",
          lru_ok, fatal=False)
    check(f"retired entry archived: gamma.json", archive_ok)
    sub_records["2_lru_prune"] = {
        "after": after, "archive_ok": archive_ok,
        "pass": lru_ok and archive_ok,
    }

    # =====================================================================
    # Sub-case 3: Version bump
    # =====================================================================
    print("\n[3] Version bump on existing domain")
    print("-" * 78)
    decoder_v2 = src / "alpha_decoder_v2.pt"
    energy_v2 = src / "alpha_energy_v2.pt"
    make_dummy_artifact(decoder_v2, label="alpha_decoder_v2")
    make_dummy_artifact(energy_v2,  label="alpha_energy_v2")
    _bump_lru()
    reg.bump_version(
        "alpha",
        decoder_src_path=decoder_v2,
        energy_src_path=energy_v2,
        energy_kind="MLPEnergyModel",
        provenance={"source": "smoke", "step": "v2"},
    )
    entry_alpha = reg.get_entry("alpha")
    bump_ok = (
        entry_alpha.current_version == 2
        and len(entry_alpha.versions) == 2
        and {v.version for v in entry_alpha.versions} == {1, 2}
    )
    check(f"alpha.versions = {[v.version for v in entry_alpha.versions]}, "
          f"current = {entry_alpha.current_version}", bump_ok)

    v2_artifact_dir = tmp / "artifacts" / "alpha__v2"
    v1_artifact_dir = tmp / "artifacts" / "alpha__v1"
    artifacts_v2_ok = (
        v2_artifact_dir.exists()
        and (v2_artifact_dir / "decoder.pt").exists()
        and v1_artifact_dir.exists()        # v1 still there
    )
    check("v1 + v2 artifacts both on disk", artifacts_v2_ok)

    # Verify v2 paths point to the new artifacts
    v2 = entry_alpha.current()
    v2_decoder = tmp / v2.decoder_path
    v2_loaded = torch.load(v2_decoder, map_location="cpu", weights_only=True)
    v2_label_ok = v2_loaded["label"] == "alpha_decoder_v2"
    check(f"current() decoder loads with v2 label: {v2_loaded['label']!r}",
          v2_label_ok)
    sub_records["3_bump_version"] = {
        "current_version": entry_alpha.current_version,
        "versions": [v.version for v in entry_alpha.versions],
        "pass": bump_ok and artifacts_v2_ok and v2_label_ok,
    }

    # =====================================================================
    # Sub-case 4: Rollback
    # =====================================================================
    print("\n[4] Rollback to previous version")
    print("-" * 78)
    _bump_lru()
    reg.rollback("alpha", target_version=1)
    entry_alpha = reg.get_entry("alpha")
    rollback_ok = (
        entry_alpha.current_version == 1
        and len(entry_alpha.versions) == 2
    )
    check(f"current_version after rollback = {entry_alpha.current_version}",
          rollback_ok)
    v1 = entry_alpha.current()
    v1_decoder = tmp / v1.decoder_path
    v1_loaded = torch.load(v1_decoder, map_location="cpu", weights_only=True)
    v1_label_ok = v1_loaded["label"] == "alpha_decoder_v1"
    check(f"current() decoder loads with v1 label: {v1_loaded['label']!r}",
          v1_label_ok)
    sub_records["4_rollback"] = {
        "current_version_after_rollback": entry_alpha.current_version,
        "pass": rollback_ok and v1_label_ok,
    }

    # =====================================================================
    # Sub-case 5: Persistence
    # =====================================================================
    print("\n[5] Persistence (fresh registry, same root)")
    print("-" * 78)
    reg2 = DomainRegistry(tmp, max_active=3)
    fresh_ids = sorted([e.domain_id for e in reg2.list_active()])
    persist_ok = fresh_ids == ["alpha", "beta", "delta"]
    check(f"fresh registry list_active: {fresh_ids}", persist_ok)

    fresh_alpha = reg2.get_entry("alpha")
    fresh_versions_ok = (
        fresh_alpha.current_version == 1
        and {v.version for v in fresh_alpha.versions} == {1, 2}
    )
    check(f"alpha versions reloaded: "
          f"{[v.version for v in fresh_alpha.versions]}, "
          f"current={fresh_alpha.current_version}",
          fresh_versions_ok)

    # Verify provenance roundtrips
    fresh_v1 = fresh_alpha.current()
    prov = fresh_v1.provenance
    prov_ok = prov.get("source") == "smoke" and prov.get("domain") == "alpha"
    check(f"provenance roundtrips: {prov}", prov_ok)
    sub_records["5_persistence"] = {
        "fresh_ids": fresh_ids, "fresh_alpha_current": fresh_alpha.current_version,
        "pass": persist_ok and fresh_versions_ok and prov_ok,
    }

    # =====================================================================
    # Acceptance
    # =====================================================================
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK")
    print("=" * 78)
    sub_pass_flags = [r["pass"] for r in sub_records.values()]
    sub_labels = [
        "Register + list + artifact stash",
        "LRU growth control",
        "Version bump",
        "Rollback",
        "Persistence (fresh registry)",
    ]
    for label, ok in zip(sub_labels, sub_pass_flags):
        print(f"  {label:<48s}  {'PASS' if ok else 'FAIL'}")
    overall = all(sub_pass_flags)
    print(f"\n→ Stage 3.3: {'PASS' if overall else 'FAIL'}")

    payload = {
        "task": "3.3",
        "registry_root": str(tmp),
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
        print(f"→ cleaned up {tmp}")
    else:
        print(f"→ kept {tmp} (--keep-tmp)")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
