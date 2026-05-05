"""Concept discovery — wake/sleep library learning (Stage 1.5).

Grows the concept library *without human authoring*. The wake phase
records planner activity (successes, failures, inefficient solves,
low-confidence answers) into a tagged replay buffer. The sleep phase
mines the buffer for new concept candidates and compound macros, then
gates them through a strict three-criterion validation (cos→truth,
non-triviality, planner-utility) before promoting to the live registry.

Module layout (built in §19.9 sub-task order):

  - `wake.py`      — replay buffer + channel routing (Task 1.5.1)
  - `cluster.py`   — Ψ-shift clustering + per-cluster operator-
                     consistency check (Task 1.5.2)
  - `refactor.py`  — e-graph subchain mining + macro promotion
                     (Task 1.5.5)
  - `registry.py`  — versioned concept registry with growth control
                     (Task 1.5.6)
  - `sleep.py`     — orchestrator (Task 1.5.7)

Safeguards (locked design rules — see plan §19.9 + memory):
  1. Cluster consistency check, not just centroid similarity.
  2. "Unexplained" = failed OR inefficient OR low_confidence.
  3. Macro promotion needs frequency AND utility gates.
  4. Validation = cos ≥ 0.85 AND non-triviality AND planner-utility.
  5. Registry has explicit growth control (cap + utility-decay prune).
  6. Sleep cadence is N=50–100 tasks per cycle.
"""
from .wake import (
    WakeBuffer,
    BufferEntry,
    Channel,
    CHANNELS,
    route_program,
)
from .cluster import (
    ConsistencyResult,
    KMeansResult,
    SweepResult,
    cluster_with_silhouette_sweep,
    compute_psi_shifts,
    kmeans_cluster,
    operator_consistency,
    silhouette_score,
    train_quick_operator,
)
from .validate import (
    ValidationResult,
    make_identity_candidate,
    validate_candidate,
)
from .refactor import (
    MacroCandidate,
    MacroPromotionResult,
    enumerate_subsequences,
    evaluate_macro_utility,
    macro_name_from_chain,
    make_macro_op,
    mine_subchain_candidates,
    promote_macros,
)

__all__ = [
    "WakeBuffer",
    "BufferEntry",
    "Channel",
    "CHANNELS",
    "route_program",
    "ConsistencyResult",
    "KMeansResult",
    "SweepResult",
    "cluster_with_silhouette_sweep",
    "compute_psi_shifts",
    "kmeans_cluster",
    "operator_consistency",
    "silhouette_score",
    "train_quick_operator",
    "ValidationResult",
    "make_identity_candidate",
    "validate_candidate",
    "MacroCandidate",
    "MacroPromotionResult",
    "enumerate_subsequences",
    "evaluate_macro_utility",
    "macro_name_from_chain",
    "make_macro_op",
    "mine_subchain_candidates",
    "promote_macros",
]
