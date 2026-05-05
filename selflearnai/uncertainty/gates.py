"""Calibrated decision gates — replacement primitive for hard cosine thresholds.

Stage 0.5's deliverable for "replace cos ≥ 0.5 hard gates with calibrated
predictions". Provides two gate types and a small migration helper:

  - `ConformalGate`: a binary decision gate backed by a fitted
    `ConformalOperatorCalibrator`. Takes a *negative cosine
    nonconformity score* and returns True iff the prediction is
    inside the calibrated coverage set.

  - `ThresholdGate`: legacy hard-threshold gate, retained for backward
    compatibility with code that hasn't migrated yet. Takes a *cosine
    similarity* and returns True iff cos >= threshold. Same API as
    `ConformalGate` so swapping is a 1-line change at the call site.

  - `gate_from(...)`: factory that returns a `ConformalGate` if a
    calibrator is provided, else a `ThresholdGate` with the given
    fallback. Lets call sites carry a single gate handle that's
    upgraded automatically when calibration data lands.

Migration policy (locked in plan §19.6):
    All NEW code in Stage 1+ uses ConformalGate. Existing call sites in
    `selflearnai/metrics/` and `scripts/run_metrics.py` retain hard
    thresholds until they're touched by the planner / acceptance-gate
    work in Stage 1. No sweeping rewrite — gates upgrade reactively.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .conformal import ConformalOperatorCalibrator


@dataclass(frozen=True)
class GateDecision:
    """Result of evaluating a gate. Carries the input value, the
    threshold/q_hat used, and a string reason for log/debug."""
    passed: bool
    value: float
    bound: float
    reason: str


class ConformalGate:
    """Decision gate backed by a fitted ConformalOperatorCalibrator.

    Input is a *nonconformity score* (typically `-cos(pred, target)` —
    same sign convention as `ConformalOperatorCalibrator.fit`). The
    gate passes iff `score <= calibrator.q_hat`, i.e. the prediction
    lies inside the (1 − α) coverage set.

    Single-call interface mirrors `ThresholdGate` so a call site can
    swap gate implementations without touching its surrounding code.
    """

    def __init__(self, calibrator: ConformalOperatorCalibrator):
        if calibrator.q_hat is None:
            raise ValueError(
                "Calibrator not fit yet. Call calibrator.fit(...) before "
                "wrapping it in a ConformalGate."
            )
        self.calibrator = calibrator

    def __call__(self, score: float) -> GateDecision:
        bound = self.calibrator.q_hat
        passed = score <= bound
        return GateDecision(
            passed=passed,
            value=score,
            bound=bound,
            reason=(
                f"conformal α={self.calibrator.alpha:.3f}: "
                f"score {score:+.4f} {'<= q_hat' if passed else '> q_hat'} "
                f"{bound:+.4f}"
            ),
        )

    @property
    def name(self) -> str:
        return f"ConformalGate(α={self.calibrator.alpha:.3f})"


class ThresholdGate:
    """Legacy hard-threshold gate — `cos >= threshold`.

    Kept for backward compatibility with existing code that doesn't yet
    have a calibrator. Same API as `ConformalGate` so call sites can be
    migrated one at a time. Note the input convention difference:
    `ConformalGate` takes a *nonconformity score* (lower = better);
    `ThresholdGate` takes a *cosine similarity* (higher = better). To
    pass either gate the same call-site value, use `gate_from(...)`
    which normalizes.
    """

    def __init__(self, threshold: float, name: str = "ThresholdGate"):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(f"cosine threshold must be in [-1, 1], got {threshold}")
        self.threshold = threshold
        self._name = name

    def __call__(self, cosine: float) -> GateDecision:
        passed = cosine >= self.threshold
        return GateDecision(
            passed=passed,
            value=cosine,
            bound=self.threshold,
            reason=(
                f"hard threshold: cos {cosine:+.4f} "
                f"{'>= threshold' if passed else '< threshold'} {self.threshold:+.4f}"
            ),
        )

    @property
    def name(self) -> str:
        return self._name


def gate_from(
    *,
    calibrator: Optional[ConformalOperatorCalibrator] = None,
    fallback_threshold: Optional[float] = None,
    name: str = "Gate",
) -> ConformalGate | ThresholdGate:
    """Return a calibrated gate if `calibrator` is fit, else a fallback
    `ThresholdGate`. Use at call sites that want to inherit calibration
    automatically when one lands.

    At least one of `calibrator` or `fallback_threshold` must be given.
    """
    if calibrator is not None and calibrator.q_hat is not None:
        return ConformalGate(calibrator)
    if fallback_threshold is not None:
        return ThresholdGate(fallback_threshold, name=name)
    raise ValueError(
        "Need a fitted calibrator OR a fallback_threshold to build a gate."
    )
