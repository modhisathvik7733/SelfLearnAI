"""Three-criterion candidate validation gate (Task 1.5.3).

Per safeguard #4 (plan §19.9, memory feedback_stage1_5_safeguards), a
candidate concept enters the live registry only if it passes ALL three
criteria. Cosine-to-truth alone admits two known failure modes:

  - identity-like operators that pass cos→truth on datasets where
    target ≈ source by chance (or where the encoder collapses some
    pair distinctions);
  - operators with high training accuracy but no off-cluster transfer.

The three criteria together rule both out:

  1. GENERALIZATION (cos→truth ≥ τ_cos) — operator's mean output
     cosine to held-out targets meets threshold. Tests that the
     learned shift transfers off the training cluster, not just
     within it (operator-consistency from Task 1.5.2 was on training
     members; this is on held-out).

  2. NON-TRIVIALITY (max cos(op(src), src) < τ_triv) — operator does
     SOMETHING. An identity operator (output ≈ input) fails this;
     so does any operator whose residual MLP collapsed during
     training. τ_triv = 0.99 by default — well below typical
     plural / past-tense shift cosines.

  3. PLANNER-UTILITY (n_solved_with > n_solved_without) — registering
     the candidate measurably improves the planner. Run the planner
     against a held-out task slice with the candidate's expected
     concept (e.g., a held-out plural pair pool when validating a
     plural candidate) — once with the baseline operator library,
     once with the candidate added. The candidate must let the
     planner reach goals it couldn't reach before. This rules out
     candidates that are technically learnable but redundant
     (already covered by an existing operator chain) or harmful
     (degrade beam search by adding distractor ops).

All three are HARD. If any criterion fails, the candidate is logged
with the failing reason and not promoted. The registry never sees
unvalidated entries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from selflearnai.concepts.operator import ConceptOperator
from selflearnai.planner import BeamSearchPlanner


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Per-candidate audit record. `passes` is the AND of all three."""
    candidate_name: str
    # Criterion 1: generalization (held-out cos→truth)
    n_holdout: int
    cos_truth_mean: float
    cos_truth_min: float
    cos_truth_threshold: float
    passes_cos_truth: bool
    # Criterion 2: non-triviality
    n_triv: int
    cos_to_input_mean: float
    cos_to_input_max: float
    non_triviality_threshold: float
    passes_non_triviality: bool
    # Criterion 3: planner-utility
    n_planner_tasks: int
    n_baseline_solved: int
    n_augmented_solved: int
    utility_improvement: int
    planner_cos_threshold: float
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
    cos_truth_min: float = 0.85,
    non_triviality_max_cos: float = 0.99,
    planner_cos_threshold: float = 0.80,
    planner_beam_width: int = 4,
    planner_max_depth: int = 3,
    planner_step_bonus: float = 0.015,
) -> ValidationResult:
    """Run all three criteria; return a self-explaining ValidationResult.

    Inputs:
      candidate_op: the trained ConceptOperator under test.
      candidate_name: registry-bound name (used in planner library).
      z_src_holdout / z_tgt_holdout: held-out pairs from the candidate's
        concept, NOT used during candidate training. Shape [n_h, dim].
      z_src_for_triviality: source embeddings used for the identity-
        check (typically the candidate's own training sources;
        running on training data is the strongest test — if the
        operator can't do anything even on the data it saw, it's
        certainly trivial). Shape [n_t, dim].
      baseline_operators: the operator library WITHOUT the candidate.
      planner_holdout_tasks: list of (psi_start, psi_goal) tensors.
        Each is a plain 1-D tensor of shape [dim].
      cos_truth_min, non_triviality_max_cos, planner_cos_threshold:
        per-criterion thresholds.
      planner_beam_width, planner_max_depth, planner_step_bonus:
        planner config used for both baseline and augmented runs.
        Identical config — only the operator library changes.
    """
    failing: list[str] = []

    # --- Criterion 1: generalization on held-out -------------------------
    pred_h = candidate_op(z_src_holdout)
    cos_t = F.cosine_similarity(pred_h, z_tgt_holdout, dim=-1)
    cos_truth_mean = float(cos_t.mean().item())
    cos_truth_min_val = float(cos_t.min().item())
    passes_cos_truth = cos_truth_mean >= cos_truth_min
    if not passes_cos_truth:
        failing.append(
            f"cos_truth_mean={cos_truth_mean:.3f} < threshold={cos_truth_min:.3f}"
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

    # --- Criterion 3: planner-utility ------------------------------------
    n_tasks = len(planner_holdout_tasks)
    n_baseline_solved = 0
    n_augmented_solved = 0
    if n_tasks > 0:
        # Baseline planner.
        baseline_planner = BeamSearchPlanner(
            operators=baseline_operators,
            beam_width=planner_beam_width,
            max_depth=planner_max_depth,
            step_bonus=planner_step_bonus,
        )
        # Augmented planner: same config + candidate added under name.
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
            if base.cos_to_goal >= planner_cos_threshold:
                n_baseline_solved += 1
            if aug.cos_to_goal >= planner_cos_threshold:
                n_augmented_solved += 1
    utility_improvement = n_augmented_solved - n_baseline_solved
    passes_planner_utility = utility_improvement > 0
    if not passes_planner_utility:
        failing.append(
            f"planner_utility: augmented_solved={n_augmented_solved} "
            f"<= baseline_solved={n_baseline_solved} "
            f"(no measurable improvement)"
        )

    overall = passes_cos_truth and passes_non_triviality and passes_planner_utility

    return ValidationResult(
        candidate_name=candidate_name,
        n_holdout=int(z_src_holdout.shape[0]),
        cos_truth_mean=cos_truth_mean,
        cos_truth_min=cos_truth_min_val,
        cos_truth_threshold=cos_truth_min,
        passes_cos_truth=passes_cos_truth,
        n_triv=int(z_src_for_triviality.shape[0]),
        cos_to_input_mean=cos_self_mean,
        cos_to_input_max=cos_self_max,
        non_triviality_threshold=non_triviality_max_cos,
        passes_non_triviality=passes_non_triviality,
        n_planner_tasks=n_tasks,
        n_baseline_solved=n_baseline_solved,
        n_augmented_solved=n_augmented_solved,
        utility_improvement=utility_improvement,
        planner_cos_threshold=planner_cos_threshold,
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
