"""End-to-end sleep cycle orchestrator (Task 1.5.6).

Wires the Stage 1.5 primitives into a single end-to-end cycle:

    wake buffer
       │
       ├── unexplained channels ──→ cluster ──→ operator-consistency
       │                                     │
       │                                     ▼
       │                          train + train/holdout split
       │                                     │
       │                                     ▼
       │                          three-criterion validation
       │                                     │
       │                                     ▼ (PASS)
       │                                registry.register
       │
       └── success channel ─────→ mine sub-chains ──→ utility eval
                                                  │
                                                  ▼ (PASS)
                                          (count promoted macros;
                                           macro persistence deferred
                                           to a future sub-task)

Safeguards inherited from upstream (no new safeguards introduced):
  - #2 unexplained = failed OR inefficient OR low_confidence (wake.py)
  - #1 cluster consistency check (cluster.py)
  - #4 three-criterion validation (validate.py)
  - #3 macro frequency + utility gates (refactor.py)
  - #5 registry growth control (registry.py)
  - #6 cadence is `sleep_every=50` tasks per cycle (driven by caller —
       this module exposes one cycle; the cadence wrapper is one
       `if (task_idx + 1) % sleep_every == 0:` line at the call site,
       not worth its own abstraction).

Idempotency: an empty wake buffer produces an empty SleepCycleResult
with no errors. A buffer with only success entries (no unexplained)
runs only the macro promotion path. A buffer with only unexplained
entries runs only the concept discovery path.

Note on macro persistence: macros are *function compositions*, not
nn.Modules. The registry stores nn.Module state_dicts, so macros
don't fit. Promoted macros are reported in SleepCycleResult and are
expected to be persisted by the caller (e.g., a separate macro
registry that stores chains as text). Implemented in a follow-up
sub-task.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from selflearnai.discovery.cluster import (
    cluster_with_silhouette_sweep,
    compute_psi_shifts,
    operator_consistency,
    train_quick_operator,
)
from selflearnai.discovery.refactor import (
    evaluate_macro_utility,
    make_macro_op,
    mine_subchain_candidates,
)
from selflearnai.discovery.registry import ConceptRegistry
from selflearnai.discovery.validate import validate_candidate
from selflearnai.discovery.wake import WakeBuffer


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------

@dataclass
class SleepConfig:
    """Tunable thresholds for one sleep cycle. Defaults match the
    individual primitives' defaults (consistent with Tasks 1.5.2,
    1.5.3, 1.5.4)."""
    # Clustering
    cluster_k_min: int = 2
    cluster_k_max: int = 6
    cluster_n_init: int = 8
    min_cluster_size: int = 5            # below this we skip the cluster
    consistency_mean_min: float = 0.80
    consistency_var_max: float = 0.10
    # Validation (three-criterion gate from 1.5.3)
    cos_truth_improvement_min: float = 0.01
    non_triviality_max_cos: float = 0.99
    planner_cos_improvement_min: float = 0.01
    per_task_improvement_min: float = 0.01
    n_tasks_improved_min: int = 1
    # Macro promotion (1.5.4)
    min_chain_length: int = 2
    max_chain_length: int = 3
    frequency_min: int = 5
    macro_depth: int = 1
    # Operator training
    operator_epochs: int = 2000
    operator_seed: int = 0
    # Cadence (driven by caller; here for completeness)
    sleep_every: int = 50


@dataclass
class ClusterAuditRow:
    """Per-cluster audit trail for the cycle JSON. Surfaces why a
    proposal was accepted or rejected — sleep should never silently
    drop candidates."""
    cluster_id: int
    n_members: int
    consistency_passed: bool
    consistency_mean: float
    consistency_var: float
    validation_attempted: bool
    validation_passed: bool
    failing_criteria: list[str] = field(default_factory=list)
    registered_as: Optional[str] = None


@dataclass
class MacroAuditRow:
    chain: list[str]
    n_occurrences: int
    buildable: bool
    utility_passed: Optional[bool]
    failing_reasons: list[str] = field(default_factory=list)


@dataclass
class SleepCycleResult:
    # Concept discovery
    n_unexplained_entries: int
    n_clusters_proposed: int
    n_clusters_consistent: int
    n_clusters_validated: int
    n_concepts_registered: int
    registered_concept_ids: list[str] = field(default_factory=list)
    cluster_audit: list[ClusterAuditRow] = field(default_factory=list)
    # Macro promotion
    n_success_entries: int = 0
    n_macro_candidates: int = 0
    n_macros_buildable: int = 0
    n_macros_promoted: int = 0
    promoted_macro_chains: list[list[str]] = field(default_factory=list)
    macro_audit: list[MacroAuditRow] = field(default_factory=list)
    # Registry state
    registry_size_before: int = 0
    registry_size_after: int = 0


# ---------------------------------------------------------------------------
# Internal: extract pairs from buffer entries
# ---------------------------------------------------------------------------

def _collect_unexplained_pairs(
    wake_buffer: WakeBuffer,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Return (pairs, channels) — pairs is [(source, target), ...] from
    every buffer entry whose program has both, channels is the channel
    name per pair (kept for audit)."""
    pairs: list[tuple[str, str]] = []
    channels: list[str] = []
    for ch in ("failed", "inefficient", "low_confidence"):
        for entry in wake_buffer.read(channel=ch):
            target = entry.program.target
            source = entry.program.source
            if source and target:
                pairs.append((source, target))
                channels.append(ch)
    return pairs, channels


def _make_concept_id(
    cluster_id: int,
    cycle_tag: str,
    registry: ConceptRegistry,
) -> str:
    """Stable, unique-per-cycle id. cycle_tag should be unique to a
    sleep invocation (timestamp or counter); the cluster_id distinguishes
    multiple concepts proposed by the same cycle."""
    return f"discovered_cluster{cluster_id}__{cycle_tag}"


# ---------------------------------------------------------------------------
# Public: one sleep cycle
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_consistency_gate(
    z_src: torch.Tensor, z_tgt: torch.Tensor, config: SleepConfig,
    *, dim: int, device: str,
) -> tuple[bool, float, float]:
    """Train operator on full cluster, return (passes, mean_cos, var_cos)."""
    op = train_quick_operator(
        z_src, z_tgt,
        dim=dim, device=device,
        seed=config.operator_seed, epochs=config.operator_epochs,
    )
    cons = operator_consistency(
        op, z_src.to(device), z_tgt.to(device),
        mean_threshold=config.consistency_mean_min,
        var_threshold=config.consistency_var_max,
    )
    return cons.passes, cons.mean_cos, cons.var_cos


def run_sleep_cycle(
    wake_buffer: WakeBuffer,
    registry: ConceptRegistry,
    *,
    encode_fn: Callable[[list[str]], torch.Tensor],
    operators: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    macro_holdout_tasks: Sequence[tuple[torch.Tensor, torch.Tensor]] = (),
    dim: int,
    device: str = "cuda",
    config: Optional[SleepConfig] = None,
    cycle_tag: Optional[str] = None,
) -> SleepCycleResult:
    """Run one sleep cycle end-to-end.

    `operators`: the planner's current operator library — used both as
    the validation baseline (criterion 3 of validate_candidate) and
    as the macro construction pool (make_macro_op resolves chain names
    against this dict). In production these are the same: operators
    register makes available to the planner.

    `macro_holdout_tasks`: (psi_start, psi_goal) pairs used by the
    macro utility gate. Empty → macro path runs but utility eval is
    skipped (every candidate fails utility because n_tasks=0).

    `cycle_tag`: label used to disambiguate concept_ids across
    multiple cycles. If not provided, generated from time.
    """
    if config is None:
        config = SleepConfig()
    if cycle_tag is None:
        from time import time as _time
        cycle_tag = f"t{int(_time() * 1000)}"

    size_before = len(registry.list_active())

    # =====================================================================
    # CONCEPT DISCOVERY
    # =====================================================================
    pairs, _channels = _collect_unexplained_pairs(wake_buffer)
    n_unexplained = len(pairs)

    cluster_audit: list[ClusterAuditRow] = []
    registered_ids: list[str] = []
    n_clusters_proposed = 0
    n_clusters_consistent = 0
    n_clusters_validated = 0
    n_concepts_registered = 0

    # Need enough pairs to cluster meaningfully. Minimum: 2 clusters of
    # min_cluster_size each.
    if n_unexplained >= max(config.cluster_k_min * config.min_cluster_size,
                            config.cluster_k_min + 1):
        z_src, z_tgt, shifts = compute_psi_shifts(encode_fn, pairs)

        sweep = cluster_with_silhouette_sweep(
            shifts.cpu(),
            k_min=config.cluster_k_min,
            k_max=min(config.cluster_k_max, n_unexplained - 1),
            n_init=config.cluster_n_init,
            seed=config.operator_seed,
            normalize_inputs=True,
        )
        labels = sweep.best_result.labels.tolist()
        n_clusters_proposed = sweep.best_k

        for cid in sorted(set(labels)):
            members = [i for i, lab in enumerate(labels) if lab == cid]
            row = ClusterAuditRow(
                cluster_id=cid,
                n_members=len(members),
                consistency_passed=False,
                consistency_mean=float("nan"),
                consistency_var=float("nan"),
                validation_attempted=False,
                validation_passed=False,
            )
            if len(members) < config.min_cluster_size:
                row.failing_criteria.append(
                    f"cluster too small: {len(members)} < min_cluster_size={config.min_cluster_size}"
                )
                cluster_audit.append(row)
                continue

            z_src_c = z_src[members]
            z_tgt_c = z_tgt[members]

            # Consistency gate (safeguard #1)
            consistent, mean_cos, var_cos = _evaluate_consistency_gate(
                z_src_c, z_tgt_c, config, dim=dim, device=device,
            )
            row.consistency_mean = mean_cos
            row.consistency_var = var_cos
            row.consistency_passed = consistent
            if not consistent:
                row.failing_criteria.append(
                    f"consistency: mean={mean_cos:.3f} (≥{config.consistency_mean_min:.2f}) "
                    f"AND var={var_cos:.4f} (<{config.consistency_var_max:.2f})"
                )
                cluster_audit.append(row)
                continue
            n_clusters_consistent += 1

            # Train/held-out split for three-criterion validation
            n_ho = max(2, len(members) // 3)
            split = len(members) - n_ho
            train_idx = members[:split]
            ho_idx = members[split:]
            z_src_train = z_src[train_idx]
            z_tgt_train = z_tgt[train_idx]
            z_src_ho = z_src[ho_idx]
            z_tgt_ho = z_tgt[ho_idx]

            # Re-train on training portion only (so holdout is genuinely held)
            op_validate = train_quick_operator(
                z_src_train, z_tgt_train,
                dim=dim, device=device,
                seed=config.operator_seed, epochs=config.operator_epochs,
            )

            concept_id = _make_concept_id(cid, cycle_tag, registry)
            row.validation_attempted = True

            val = validate_candidate(
                op_validate, concept_id,
                z_src_holdout=z_src_ho, z_tgt_holdout=z_tgt_ho,
                z_src_for_triviality=z_src_train,
                baseline_operators=operators,
                planner_holdout_tasks=[
                    (z_src_ho[i], z_tgt_ho[i]) for i in range(len(ho_idx))
                ],
                cos_truth_improvement_min=config.cos_truth_improvement_min,
                non_triviality_max_cos=config.non_triviality_max_cos,
                planner_cos_improvement_min=config.planner_cos_improvement_min,
                per_task_improvement_min=config.per_task_improvement_min,
                n_tasks_improved_min=config.n_tasks_improved_min,
            )
            row.validation_passed = val.passes
            row.failing_criteria.extend(val.failing_criteria)
            if val.passes:
                n_clusters_validated += 1
                registry.register(
                    concept_id, op_validate,
                    provenance={
                        "source": "sleep_cycle",
                        "cycle_tag": cycle_tag,
                        "cluster_id": cid,
                        "n_support": len(members),
                        "consistency_mean": mean_cos,
                        "consistency_var": var_cos,
                        "validation": {
                            "cos_truth_improvement": val.cos_truth_improvement,
                            "cos_to_input_max": val.cos_to_input_max,
                            "planner_cos_improvement_mean": val.planner_cos_improvement_mean,
                            "n_tasks_improved": val.n_tasks_improved,
                        },
                    },
                    support_count=len(members),
                )
                row.registered_as = concept_id
                registered_ids.append(concept_id)
                n_concepts_registered += 1
            cluster_audit.append(row)

    # =====================================================================
    # MACRO PROMOTION
    # =====================================================================
    success_entries = wake_buffer.read(channel="success")
    n_success = len(success_entries)
    n_macro_candidates = 0
    n_macros_buildable = 0
    n_macros_promoted = 0
    promoted_macro_chains: list[list[str]] = []
    macro_audit: list[MacroAuditRow] = []

    if n_success > 0:
        candidates = mine_subchain_candidates(
            success_entries,
            min_chain_length=config.min_chain_length,
            max_chain_length=config.max_chain_length,
            frequency_min=config.frequency_min,
        )
        n_macro_candidates = len(candidates)
        for cand in candidates:
            row = MacroAuditRow(
                chain=list(cand.chain),
                n_occurrences=cand.n_occurrences,
                buildable=False,
                utility_passed=None,
            )
            try:
                macro_op = make_macro_op(cand.chain, operators)
                row.buildable = True
                n_macros_buildable += 1
            except KeyError as e:
                row.failing_reasons.append(f"unbuildable: {e}")
                macro_audit.append(row)
                continue
            if not macro_holdout_tasks:
                row.utility_passed = False
                row.failing_reasons.append(
                    "no macro_holdout_tasks provided; utility eval skipped"
                )
                macro_audit.append(row)
                continue
            result = evaluate_macro_utility(
                cand, macro_op,
                operators=operators,
                holdout_tasks=macro_holdout_tasks,
                macro_depth=config.macro_depth,
                cos_improvement_min=config.planner_cos_improvement_min,
                per_task_improvement_min=config.per_task_improvement_min,
                n_tasks_improved_min=config.n_tasks_improved_min,
            )
            row.utility_passed = result.passes_utility
            row.failing_reasons.extend(result.failing_reasons)
            if result.passes_utility:
                n_macros_promoted += 1
                promoted_macro_chains.append(list(cand.chain))
            macro_audit.append(row)

    return SleepCycleResult(
        n_unexplained_entries=n_unexplained,
        n_clusters_proposed=n_clusters_proposed,
        n_clusters_consistent=n_clusters_consistent,
        n_clusters_validated=n_clusters_validated,
        n_concepts_registered=n_concepts_registered,
        registered_concept_ids=registered_ids,
        cluster_audit=cluster_audit,
        n_success_entries=n_success,
        n_macro_candidates=n_macro_candidates,
        n_macros_buildable=n_macros_buildable,
        n_macros_promoted=n_macros_promoted,
        promoted_macro_chains=promoted_macro_chains,
        macro_audit=macro_audit,
        registry_size_before=size_before,
        registry_size_after=len(registry.list_active()),
    )
