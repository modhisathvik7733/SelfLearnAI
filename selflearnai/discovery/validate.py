"""Three-criterion candidate validation gate (Task 1.5.3).

Per safeguard #4 (plan §19.9, memory feedback_stage1_5_safeguards), a
candidate concept enters the live registry only if it passes ALL three
criteria. Cosine-to-truth alone admits two known failure modes:

  - identity-like operators that pass cos→truth on datasets where
    target ≈ source by chance (or where the encoder collapses some
    pair distinctions);
  - operators with high training accuracy but no off-cluster transfer.

The three criteria together rule both out:

  1. GENERALIZATION (RELATIVE) — operator beats the no-op baseline:
       mean(cos(op(src_h), tgt_h)) - mean(cos(src_h, tgt_h)) ≥ τ_rel
     on held-out pairs. Absolute thresholds fail here because
     source/target cosines are domain-dependent (plural pairs sit
     at ~0.93 cos in E5 before any operator runs) — identity and
     near-identity ops trivially pass any reasonable absolute gate.
     The relative gate measures the only thing we care about: does
     applying the operator move us CLOSER to the goal than not
     applying it? Same reframing as silhouette in Task 1.5.2.

  2. NON-TRIVIALITY (max cos(op(src), src) < τ_triv) — operator does
     SOMETHING. An identity operator (output ≈ input) fails this;
     so does any operator whose residual MLP collapsed during
     training. τ_triv = 0.99 by default — well below typical
     concept-shift cosines.

  3. PLANNER-UTILITY (RELATIVE) — augmented planner reaches HIGHER
     cos_to_goal than baseline planner across held-out tasks:
       mean(aug_cos - baseline_cos) ≥ τ_planner
     AND ≥ τ_n_tasks tasks individually improved by ≥ τ_per_task.
     Absolute thresholds fail here for the same reason as criterion 1:
     depth-0 PlanState (no ops applied) already crosses any
     reasonable absolute cos in cases where source and target are
     close. The relative gate measures whether ADDING the candidate
     to the library changes what the planner can reach.

All three are HARD. If any criterion fails, the candidate is logged
with the failing reason and not promoted. The registry never sees
unvalidated entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from selflearnai.concepts.operator import ConceptOperator
from selflearnai.planner import BeamSearchPlanner


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Per-candidate audit record. `passes` is the AND of all three.

    All three criteria use RELATIVE gates (improvement over no-op /
    over baseline planner) — see module docstring for why absolute
    thresholds fail in encoder-space cosine.
    """
    candidate_name: str
    # Criterion 1: generalization (relative; informational absolute mean kept)
    n_holdout: int
    cos_op_truth_mean: float                # mean cos(op(src_h), tgt_h)  [informational]
    cos_src_truth_mean: float               # mean cos(src_h, tgt_h)  — no-op baseline
    cos_truth_improvement: float            # cos_op_truth_mean - cos_src_truth_mean
    cos_truth_improvement_threshold: float
    passes_cos_truth: bool
    # Criterion 2: non-triviality
    n_triv: int
    cos_to_input_mean: float
    cos_to_input_max: float
    non_triviality_threshold: float
    passes_non_triviality: bool
    # Criterion 3: planner-utility (relative)
    n_planner_tasks: int
    planner_baseline_cos_mean: float
    planner_augmented_cos_mean: float
    planner_cos_improvement_mean: float
    planner_cos_improvement_threshold: float
    n_tasks_improved: int
    n_tasks_improved_min: int
    per_task_improvement_threshold: float
    passes_planner_utility: bool
    # Overall
    passes: bool
    failing_criteria: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Three-criterion validator
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_candidate(
    candidate_op: ConceptOperator,
    candidate_name: str,
    *,
    z_src_holdout: torch.Tensor,
    z_tgt_holdout: torch.Tensor,
    z_src_for_triviality: torch.Tensor,
    baseline_operators: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    planner_holdout_tasks: Sequence[tuple[torch.Tensor, torch.Tensor]],
    cos_truth_improvement_min: float = 0.01,
    non_triviality_max_cos: float = 0.99,
    planner_cos_improvement_min: float = 0.01,
    per_task_improvement_min: float = 0.01,
    n_tasks_improved_min: int = 1,
    planner_beam_width: int = 4,
    planner_max_depth: int = 3,
    planner_step_bonus: float = 0.015,
) -> ValidationResult:
    """Run all three criteria; return a self-explaining ValidationResult.

    All three gates are RELATIVE. See module docstring.

    Inputs:
      candidate_op: the trained ConceptOperator under test.
      candidate_name: registry-bound name (used in planner library).
      z_src_holdout / z_tgt_holdout: held-out pairs from the candidate's
        concept, NOT used during candidate training. Shape [n_h, dim].
      z_src_for_triviality: source embeddings used for the identity-
        check (typically the candidate's own training sources).
        Shape [n_t, dim].
      baseline_operators: the operator library WITHOUT the candidate.
      planner_holdout_tasks: list of (psi_start, psi_goal) tensors.
        Each is a plain 1-D tensor of shape [dim].
      cos_truth_improvement_min: criterion 1 threshold (default 0.01,
        i.e. operator must improve mean cos→truth by ≥ 1% absolute
        over the no-op baseline).
      non_triviality_max_cos: criterion 2 threshold (default 0.99).
      planner_cos_improvement_min: criterion 3 mean-improvement
        threshold (default 0.01).
      per_task_improvement_min: per-task improvement threshold for
        counting "tasks improved" in criterion 3 (default 0.01).
      n_tasks_improved_min: criterion 3 also requires this many
        tasks to be individually improved (default 1) — guards
        against a single outlier task carrying the mean.
      planner_beam_width / planner_max_depth / planner_step_bonus:
        planner config, identical between baseline and augmented runs;
        only the operator library differs.
    """
    failing: list[str] = []

    # --- Criterion 1: generalization (relative) --------------------------
    pred_h = candidate_op(z_src_holdout)
    cos_op_truth = F.cosine_similarity(pred_h, z_tgt_holdout, dim=-1)
    cos_src_truth = F.cosine_similarity(z_src_holdout, z_tgt_holdout, dim=-1)
    cos_op_truth_mean = float(cos_op_truth.mean().item())
    cos_src_truth_mean = float(cos_src_truth.mean().item())
    cos_truth_improvement = cos_op_truth_mean - cos_src_truth_mean
    passes_cos_truth = cos_truth_improvement >= cos_truth_improvement_min
    if not passes_cos_truth:
        failing.append(
            f"cos_truth_improvement={cos_truth_improvement:+.4f} "
            f"< threshold={cos_truth_improvement_min:.4f} "
            f"(op_truth={cos_op_truth_mean:.3f}, "
            f"src_truth_baseline={cos_src_truth_mean:.3f})"
        )

    # --- Criterion 2: non-triviality -------------------------------------
    pred_t = candidate_op(z_src_for_triviality)
    cos_self = F.cosine_similarity(pred_t, z_src_for_triviality, dim=-1)
    cos_self_mean = float(cos_self.mean().item())
    cos_self_max = float(cos_self.max().item())
    passes_non_triviality = cos_self_max < non_triviality_max_cos
    if not passes_non_triviality:
        failing.append(
            f"cos_to_input_max={cos_self_max:.3f} >= threshold="
            f"{non_triviality_max_cos:.3f} (operator is identity-like)"
        )

    # --- Criterion 3: planner-utility (relative) -------------------------
    n_tasks = len(planner_holdout_tasks)
    baseline_cos: list[float] = []
    augmented_cos: list[float] = []
    n_tasks_improved = 0
    if n_tasks > 0:
        baseline_planner = BeamSearchPlanner(
            operators=baseline_operators,
            beam_width=planner_beam_width,
            max_depth=planner_max_depth,
            step_bonus=planner_step_bonus,
        )
        augmented_ops = {**baseline_operators, candidate_name: candidate_op}
        augmented_planner = BeamSearchPlanner(
            operators=augmented_ops,
            beam_width=planner_beam_width,
            max_depth=planner_max_depth,
            step_bonus=planner_step_bonus,
        )
        for psi_start, psi_goal in planner_holdout_tasks:
            base = baseline_planner.search(psi_start, psi_goal)
            aug = augmented_planner.search(psi_start, psi_goal)
            baseline_cos.append(base.cos_to_goal)
            augmented_cos.append(aug.cos_to_goal)
            if aug.cos_to_goal - base.cos_to_goal >= per_task_improvement_min:
                n_tasks_improved += 1
    planner_baseline_mean = (
        float(sum(baseline_cos) / len(baseline_cos)) if baseline_cos else 0.0
    )
    planner_augmented_mean = (
        float(sum(augmented_cos) / len(augmented_cos)) if augmented_cos else 0.0
    )
    planner_cos_improvement_mean = planner_augmented_mean - planner_baseline_mean
    passes_planner_utility = (
        planner_cos_improvement_mean >= planner_cos_improvement_min
        and n_tasks_improved >= n_tasks_improved_min
    )
    if not passes_planner_utility:
        failing.append(
            f"planner-utility: improvement_mean="
            f"{planner_cos_improvement_mean:+.4f} "
            f"(threshold ≥ {planner_cos_improvement_min:.4f}), "
            f"tasks_improved={n_tasks_improved}/{n_tasks} "
            f"(threshold ≥ {n_tasks_improved_min})"
        )

    overall = passes_cos_truth and passes_non_triviality and passes_planner_utility

    return ValidationResult(
        candidate_name=candidate_name,
        n_holdout=int(z_src_holdout.shape[0]),
        cos_op_truth_mean=cos_op_truth_mean,
        cos_src_truth_mean=cos_src_truth_mean,
        cos_truth_improvement=cos_truth_improvement,
        cos_truth_improvement_threshold=cos_truth_improvement_min,
        passes_cos_truth=passes_cos_truth,
        n_triv=int(z_src_for_triviality.shape[0]),
        cos_to_input_mean=cos_self_mean,
        cos_to_input_max=cos_self_max,
        non_triviality_threshold=non_triviality_max_cos,
        passes_non_triviality=passes_non_triviality,
        n_planner_tasks=n_tasks,
        planner_baseline_cos_mean=planner_baseline_mean,
        planner_augmented_cos_mean=planner_augmented_mean,
        planner_cos_improvement_mean=planner_cos_improvement_mean,
        planner_cos_improvement_threshold=planner_cos_improvement_min,
        n_tasks_improved=n_tasks_improved,
        n_tasks_improved_min=n_tasks_improved_min,
        per_task_improvement_threshold=per_task_improvement_min,
        passes_planner_utility=passes_planner_utility,
        passes=overall,
        failing_criteria=failing,
    )


# ---------------------------------------------------------------------------
# Helpers for crafting test candidates (used by the smoke + future sleep)
# ---------------------------------------------------------------------------

def make_identity_candidate(dim: int, device: str = "cpu") -> ConceptOperator:
    """ConceptOperator with alpha=0 and residual MLP zero'd → forward(z) = z.

    Used by the smoke to verify the non-triviality gate fires. Also
    useful as a baseline "do-nothing" reference if a future sleep
    cycle wants to bound the residual-MLP magnitude of every
    candidate.
    """
    op = ConceptOperator(dim=dim).to(device)
    with torch.no_grad():
        op.alpha.zero_()
        for p in op.residual.parameters():
            p.zero_()
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op
