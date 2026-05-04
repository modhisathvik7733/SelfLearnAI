"""Failure-mode detectors — locked from plan section 6c.

Run after every stage. Any failure here = blocker.

  1. Embedding collapse: per-dim stddev < 0.2.
  2. "Good cosine, bad semantics" — three orthogonal probes.
  3. Modality leakage / shortcut: visual perturbation insensitivity.
  4. Per-noun memorization: cross-category transfer gap > threshold.
  5. Latent geometry drift: prior-stage metrics regressed > 10%.
  6. Anchor drift: cos(adapter_c at start, at now) < 0.85.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class FailureModeReport:
    has_collapse: bool
    has_anchor_drift: bool
    has_modality_leakage: bool
    has_per_noun_memorization: bool
    has_geometry_drift: bool
    notes: list[str]

    @property
    def passed(self) -> bool:
        return not any([
            self.has_collapse,
            self.has_anchor_drift,
            self.has_modality_leakage,
            self.has_per_noun_memorization,
            self.has_geometry_drift,
        ])


def check_collapse(per_dim_std: dict, threshold: float = 0.2) -> tuple[bool, str]:
    """Flag if any adapter has min per-dim stddev < threshold."""
    failures = []
    for adapter_name, stats in per_dim_std.items():
        if stats["min_std"] < threshold:
            failures.append(
                f"{adapter_name}: min_std={stats['min_std']:.3f} < {threshold}"
            )
    if failures:
        return True, "Collapse: " + "; ".join(failures)
    return False, "No collapse."


def check_anchor_drift(anchor_cos: float, threshold: float = 0.85) -> tuple[bool, str]:
    """Flag if cos(adapter_c at start, at now) drops below threshold."""
    if anchor_cos < threshold:
        return True, f"Anchor drift: cos={anchor_cos:.3f} < {threshold} (star topology compromised)"
    return False, f"Anchor stable (cos={anchor_cos:.3f})."


def check_modality_leakage(
    perturbation_sensitivity: float,
    threshold: float = 0.05,
) -> tuple[bool, str]:
    """Flag if visual perturbation barely changes the embedding (vision being ignored)."""
    if perturbation_sensitivity < threshold:
        return True, (
            f"Modality leakage: visual perturbation sensitivity = "
            f"{perturbation_sensitivity:.3f} < {threshold}. Vision is being ignored."
        )
    return False, f"Vision is responsive (Δ={perturbation_sensitivity:.3f})."


def check_per_noun_memorization(
    cross_category_gap: float,
    threshold: float = 0.30,
) -> tuple[bool, str]:
    """Flag if there's a large gap between in-category and out-of-category transfer.
    Large gap → operator memorized a specific category, not a general concept."""
    if cross_category_gap is None:
        return False, "Cross-category gap not computed (missing held-out category data)."
    if cross_category_gap > threshold:
        return True, (
            f"Per-noun/category memorization: in-cat - out-of-cat = "
            f"{cross_category_gap:.3f} > {threshold}. Operator does not generalize."
        )
    return False, f"Operator generalizes across categories (gap={cross_category_gap:.3f})."


def check_geometry_drift(
    prior_metric_now: float,
    prior_metric_at_freeze: float,
    name: str,
    threshold: float = 0.10,
) -> tuple[bool, str]:
    """Flag if a prior-stage metric regressed by more than `threshold` (relative)."""
    if prior_metric_at_freeze == 0:
        return False, f"{name}: skipped (baseline=0)"
    delta = (prior_metric_at_freeze - prior_metric_now) / prior_metric_at_freeze
    if delta > threshold:
        return True, (
            f"Geometry drift: {name} regressed {delta:.1%} "
            f"({prior_metric_at_freeze:.3f} → {prior_metric_now:.3f})"
        )
    return False, f"{name}: stable ({prior_metric_now:.3f})"


def report_failure_modes(
    *,
    per_dim_std: dict,
    anchor_cos: float,
    perturbation_sensitivity: float,
    cross_category_gap: float | None = None,
    prior_metric_baselines: dict[str, tuple[float, float]] | None = None,  # name → (now, baseline)
) -> FailureModeReport:
    """Run all failure-mode detectors. Each returns (failed, note)."""
    notes: list[str] = []

    has_collapse, n = check_collapse(per_dim_std);            notes.append(n)
    has_anchor_drift, n = check_anchor_drift(anchor_cos);     notes.append(n)
    has_leakage, n = check_modality_leakage(perturbation_sensitivity);  notes.append(n)
    has_memorization, n = check_per_noun_memorization(cross_category_gap); notes.append(n)

    has_drift = False
    if prior_metric_baselines is not None:
        for name, (now, baseline) in prior_metric_baselines.items():
            failed, n = check_geometry_drift(now, baseline, name)
            notes.append(n)
            has_drift = has_drift or failed

    return FailureModeReport(
        has_collapse=has_collapse,
        has_anchor_drift=has_anchor_drift,
        has_modality_leakage=has_leakage,
        has_per_noun_memorization=has_memorization,
        has_geometry_drift=has_drift,
        notes=notes,
    )
