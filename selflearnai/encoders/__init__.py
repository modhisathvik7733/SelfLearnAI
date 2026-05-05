"""Encoder calibration + adapter framework (Stage 0.5).

- `calibration.py`: scores a frozen encoder against the
  `data/encoder_diagnostics/` task-family suite, producing per-family
  scores and per-family recommendations (best encoder, adapter-watch
  flag).
- `adapters.py` (Task 0.5.6, not yet built): typed `AdapterHead`
  framework. Reactive policy — concrete adapters trained only when
  evidence demands.
"""
from .calibration import (
    DiagnosticFamily,
    DiagnosticTriple,
    EncoderCalibrator,
    encoder_recommendations,
    load_default_suite,
    read_family,
    score_family,
)
from .adapters import (
    AdapterHead,
    AdapterRegistry,
    DEFAULT_ADAPTER_REGISTRY,
    OperatorWithAdapter,
)

__all__ = [
    # calibration
    "DiagnosticFamily",
    "DiagnosticTriple",
    "EncoderCalibrator",
    "encoder_recommendations",
    "load_default_suite",
    "read_family",
    "score_family",
    # adapters
    "AdapterHead",
    "AdapterRegistry",
    "DEFAULT_ADAPTER_REGISTRY",
    "OperatorWithAdapter",
]
