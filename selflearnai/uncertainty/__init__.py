"""Uncertainty quantification for ConceptOperators (Stage 0.5).

Implements split-conformal calibration on top of trained operators,
giving distribution-free, finite-sample coverage guarantees on
prediction sets.

Replaces hard cosine thresholds (e.g. `cos > 0.5`) with calibrated
coverage sets at user-chosen confidence level (1 − α).

Reference: Vovk, Shafer & Gammerman (2005); gentle intro Angelopoulos
& Bates (2021), arXiv:2107.07511. Compositional extension for
neuro-symbolic chains: arXiv:2405.15912 (planned for `compositional.py`,
Task 0.5.3).
"""
from .conformal import ConformalOperatorCalibrator, coverage_curve
from .compositional import BonferroniChainCalibrator, compose_operators

__all__ = [
    "ConformalOperatorCalibrator",
    "coverage_curve",
    "BonferroniChainCalibrator",
    "compose_operators",
]
