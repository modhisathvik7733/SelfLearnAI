"""Task 0.5.6 — adapter framework smoke test.

Pure unit-style validation that the adapter framework imports and
behaves correctly, with NO concrete adapter trained (per locked
reactive policy).

Six checks:
  1. Module imports cleanly.
  2. DEFAULT_ADAPTER_REGISTRY is empty by default.
  3. AdapterHead instantiates at multiple dims; forward pass shapes match.
  4. Untrained AdapterHead is approximately identity (residual init).
  5. OperatorWithAdapter wraps a ConceptOperator and forward works.
  6. AdapterRegistry rejects unfrozen adapters and family mismatches.

Acceptance gate: all 6 checks PASS. No model loading, no GPU needed —
this is a structural test. Regression that "existing concepts work
unchanged" is verified separately by running the operator pipeline
regressions (RUNBOOK / scripts/multi_head_opposite.py etc).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from selflearnai.encoders import (
    AdapterHead,
    AdapterRegistry,
    DEFAULT_ADAPTER_REGISTRY,
    OperatorWithAdapter,
)
from selflearnai.concepts import ConceptOperator


def check_1_imports() -> tuple[bool, str]:
    # Imports succeeded if we reached here.
    return True, "imports clean"


def check_2_default_registry_empty() -> tuple[bool, str]:
    if len(DEFAULT_ADAPTER_REGISTRY) == 0:
        return True, f"DEFAULT_ADAPTER_REGISTRY empty ({len(DEFAULT_ADAPTER_REGISTRY)} adapters)"
    return False, (
        f"DEFAULT_ADAPTER_REGISTRY has {len(DEFAULT_ADAPTER_REGISTRY)} adapters; "
        f"reactive policy requires zero at startup"
    )


def check_3_adapter_shapes() -> tuple[bool, str]:
    for dim in (384, 768, 1024):
        a = AdapterHead(dim=dim, family="test")
        z = torch.randn(7, dim)
        out = a(z)
        if out.shape != z.shape:
            return False, f"dim={dim}: in-shape {z.shape} != out-shape {out.shape}"
    return True, "shapes preserved across dims 384, 768, 1024"


def check_4_untrained_adapter_near_identity() -> tuple[bool, str]:
    """An untrained AdapterHead should be CLOSE to identity (residual
    + LayerNorm-zeroed-residual init). cos(z, adapter(z)) > 0.85 across
    random inputs is a reasonable lower bound for a fresh init."""
    torch.manual_seed(0)
    dim = 768
    a = AdapterHead(dim=dim, family="test")
    a.eval()
    z = torch.randn(64, dim)
    with torch.no_grad():
        out = a(z)
    cos = torch.nn.functional.cosine_similarity(z, out, dim=-1)
    mean_cos = cos.mean().item()
    if mean_cos > 0.85:
        return True, f"untrained adapter ~ identity: mean cos(z, adapter(z)) = {mean_cos:.3f}"
    return False, (
        f"untrained adapter NOT near identity: mean cos(z, adapter(z)) = {mean_cos:.3f}; "
        f"residual init may be too aggressive"
    )


def check_5_operator_with_adapter() -> tuple[bool, str]:
    dim = 768
    adapter = AdapterHead(dim=dim, family="test")
    op = ConceptOperator(dim=dim)
    wrapped = OperatorWithAdapter(adapter=adapter, operator=op)
    z = torch.randn(3, dim)
    with torch.no_grad():
        out = wrapped(z)
    if out.shape != z.shape:
        return False, f"wrapped op: in-shape {z.shape} != out-shape {out.shape}"
    # Confirm wrapper exposes both submodules so .parameters() works.
    n_params = sum(p.numel() for p in wrapped.parameters())
    if n_params <= 0:
        return False, "wrapped op has no trainable parameters"
    return True, f"OperatorWithAdapter wraps ConceptOperator; total params={n_params:,}"


def check_6_registry_rejects_invalid() -> tuple[bool, str]:
    reg = AdapterRegistry()
    a = AdapterHead(dim=768, family="antonym")
    # 6a. Reject unfrozen adapter.
    try:
        reg.register("antonym", a)
        return False, "registry accepted unfrozen adapter (should have raised)"
    except ValueError:
        pass
    # 6b. Reject family mismatch even after freezing.
    a.freeze()
    try:
        reg.register("paraphrase", a)
        return False, "registry accepted family mismatch (should have raised)"
    except ValueError:
        pass
    # 6c. Accept frozen + matched family.
    reg.register("antonym", a)
    if "antonym" not in reg:
        return False, "registry contains-check failed after register()"
    if reg.get("nonexistent") is not None:
        return False, "registry returned non-None for unregistered family"
    return True, "rejects unfrozen + mismatched, accepts valid, returns None for missing"


def main() -> None:
    print("Task 0.5.6 — adapter framework smoke test")
    print("=" * 60)

    checks = [
        ("1. Module imports cleanly",                 check_1_imports),
        ("2. DEFAULT_ADAPTER_REGISTRY is empty",      check_2_default_registry_empty),
        ("3. AdapterHead shapes preserved",           check_3_adapter_shapes),
        ("4. Untrained adapter ≈ identity",           check_4_untrained_adapter_near_identity),
        ("5. OperatorWithAdapter wraps ConceptOperator", check_5_operator_with_adapter),
        ("6. AdapterRegistry rejects invalid registrations", check_6_registry_rejects_invalid),
    ]

    n_pass = 0
    fail_lines: list[str] = []
    for label, fn in checks:
        ok, detail = fn()
        tag = "✓" if ok else "✗"
        print(f"  {tag} {label}")
        print(f"      {detail}")
        if ok:
            n_pass += 1
        else:
            fail_lines.append(f"{label}: {detail}")

    print("\n" + "=" * 60)
    print("ACCEPTANCE CHECK (Task 0.5.6)")
    print("=" * 60)
    print(f"  {n_pass} / {len(checks)} checks passed")
    overall = n_pass == len(checks)
    print(f"\n→ Task 0.5.6: {'PASS' if overall else 'FAIL'}")
    if fail_lines:
        print("\nFailures:")
        for fl in fail_lines:
            print(f"  - {fl}")

    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
