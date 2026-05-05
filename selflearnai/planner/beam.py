"""Beam-search planner skeleton (Task 1.6).

Given:
  - a starting embedding ψ_start in encoder space,
  - a goal embedding ψ_goal in the same space,
  - a library of named operators (any callable z → z transformation),

the planner explores operator chains up to `max_depth`, keeping the top
`beam_width` partial chains by cosine-to-goal at each depth. Returns
the highest-scoring `PlanState` found across ALL depths (including
depth 0 — i.e., the no-op chain — so the planner can correctly say
"no transformation needed" when ψ_start ≈ ψ_goal).

This module is the SKELETON only. Heuristics (operator-prior,
value-function), per-step verification, and Ψ-program trace
serialization arrive in Tasks 1.7–1.11. Today, scoring is pure
cosine-to-goal — adequate for short-chain validation (depth ≤ 5)
on small operator libraries.

Score function: `cos(ψ_current, ψ_goal)`. Higher = better. The
nonconformity / energy view (Stage 5) will swap this for a calibrated
distance once it lands; the API is shape-stable for that swap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


# Type alias: any callable that maps a (D,) or (B, D) tensor to the
# same shape. Concept operators (selflearnai.concepts.ConceptOperator),
# wrapped operators, and lambdas all satisfy this.
Operator = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class PlanState:
    """A single node in the beam-search tree.

    Fields are immutable conceptually (we never mutate after creation);
    using a regular dataclass — not frozen=True — only because torch
    tensors don't hash well and frozen dataclasses require hashable
    fields.

    Attributes:
        psi: current embedding, shape (D,).
        chain: tuple of operator names applied so far, in order.
        score: heuristic score used for beam ordering. Higher = better.
            With step_bonus > 0 this is `cos_to_goal + step_bonus * depth`;
            with step_bonus = 0 it equals `cos_to_goal`.
        cos_to_goal: raw cosine similarity to goal at this state.
            Surfaced separately so reports / diagnostics can show the
            unmodified geometric distance independent of the heuristic.
        depth: length of `chain`. depth == 0 ⇒ chain == ().
    """
    psi: torch.Tensor
    chain: tuple[str, ...]
    score: float
    cos_to_goal: float
    depth: int

    @property
    def is_initial(self) -> bool:
        return self.depth == 0


class BeamSearchPlanner:
    """Plain beam search over an operator library in latent space.

    Construction:
        planner = BeamSearchPlanner(
            operators={"agentive": op_a, "plural": op_p},
            beam_width=4,
            max_depth=5,
        )

    Search:
        result = planner.search(psi_start, psi_goal)
        # result.chain   → e.g. ("agentive", "plural")
        # result.psi     → final embedding
        # result.score   → cos(final, goal)

    The search keeps the top `beam_width` partial chains by score at
    each depth, plus the global best across all depths (including the
    no-op depth-0 baseline). With B operators and depth D, total work
    is at most beam_width × B × D operator applications — bounded and
    fast for B, D < 10.
    """

    def __init__(
        self,
        operators: dict[str, Operator],
        beam_width: int = 4,
        max_depth: int = 5,
        step_bonus: float = 0.0,
    ):
        """
        Args:
            operators: name → callable operator dict.
            beam_width: top-K states to keep at each depth.
            max_depth: max chain length to explore (0 = no-op only).
            step_bonus: tie-break / heuristic bonus added to the cos
                score per operator applied. With step_bonus = 0 the
                planner is pure cos-to-goal (the Task 1.6 baseline).
                With small positive step_bonus (~0.01-0.02) the planner
                prefers multi-step chains that produce nearly-tied
                cos scores — the standard "informative-when-cos-margin
                -is-tiny" tie-breaker, validated on the agentive ∘
                plural test in Task 1.7.
        """
        if not operators:
            raise ValueError("Need at least one operator in the library.")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        if step_bonus < 0:
            raise ValueError(f"step_bonus must be >= 0, got {step_bonus}")
        self.operators: dict[str, Operator] = dict(operators)
        self.beam_width: int = beam_width
        self.max_depth: int = max_depth
        self.step_bonus: float = step_bonus

    def _score(self, psi: torch.Tensor, psi_goal: torch.Tensor, depth: int) -> tuple[float, float]:
        """Compute (combined_score, cos_to_goal) for a state.

        combined_score = cos_to_goal + step_bonus * depth — the value
        used for beam ordering. cos_to_goal is the raw geometric
        similarity, surfaced for reporting.
        """
        cos = float(
            F.cosine_similarity(
                psi.unsqueeze(0), psi_goal.unsqueeze(0), dim=-1
            ).item()
        )
        score = cos + self.step_bonus * depth
        return score, cos

    @torch.no_grad()
    def search(
        self,
        psi_start: torch.Tensor,
        psi_goal: torch.Tensor,
    ) -> PlanState:
        """Find the highest-scoring operator chain from psi_start to
        psi_goal under the planner's score function.

        Returns the best PlanState seen across all depths from 0 to
        max_depth (inclusive). Including the depth-0 baseline means a
        start state already at the goal returns chain = ().
        """
        psi_start = psi_start.detach().flatten()
        psi_goal = psi_goal.detach().flatten()
        if psi_start.shape != psi_goal.shape:
            raise ValueError(
                f"psi_start shape {psi_start.shape} != psi_goal shape {psi_goal.shape}"
            )

        score0, cos0 = self._score(psi_start, psi_goal, depth=0)
        initial = PlanState(
            psi=psi_start,
            chain=(),
            score=score0,
            cos_to_goal=cos0,
            depth=0,
        )
        best = initial

        # No expansion needed if max_depth=0; just return the baseline.
        if self.max_depth == 0:
            return best

        beam: list[PlanState] = [initial]

        for d in range(1, self.max_depth + 1):
            candidates: list[PlanState] = []
            for state in beam:
                for op_name, op in self.operators.items():
                    next_psi = op(state.psi.unsqueeze(0)).squeeze(0)
                    score, cos = self._score(next_psi, psi_goal, depth=d)
                    new_state = PlanState(
                        psi=next_psi,
                        chain=state.chain + (op_name,),
                        score=score,
                        cos_to_goal=cos,
                        depth=d,
                    )
                    candidates.append(new_state)
                    if new_state.score > best.score:
                        best = new_state

            if not candidates:
                break

            candidates.sort(key=lambda s: s.score, reverse=True)
            beam = candidates[: self.beam_width]

        return best

    @torch.no_grad()
    def search_top_k(
        self,
        psi_start: torch.Tensor,
        psi_goal: torch.Tensor,
        k: int = 5,
    ) -> list[PlanState]:
        """Return the top-K PlanStates encountered during search,
        sorted by combined score (highest first).

        Useful for diagnosing why the best chain was chosen — inspect
        the runner-up chains to see whether the planner is making a
        clear decision or hovering between alternatives.
        """
        psi_start = psi_start.detach().flatten()
        psi_goal = psi_goal.detach().flatten()
        if psi_start.shape != psi_goal.shape:
            raise ValueError(
                f"psi_start shape {psi_start.shape} != psi_goal shape {psi_goal.shape}"
            )

        score0, cos0 = self._score(psi_start, psi_goal, depth=0)
        initial = PlanState(
            psi=psi_start, chain=(), score=score0, cos_to_goal=cos0, depth=0,
        )
        all_seen: list[PlanState] = [initial]
        beam: list[PlanState] = [initial]

        for d in range(1, self.max_depth + 1):
            candidates: list[PlanState] = []
            for state in beam:
                for op_name, op in self.operators.items():
                    next_psi = op(state.psi.unsqueeze(0)).squeeze(0)
                    score, cos = self._score(next_psi, psi_goal, depth=d)
                    new_state = PlanState(
                        psi=next_psi,
                        chain=state.chain + (op_name,),
                        score=score,
                        cos_to_goal=cos,
                        depth=d,
                    )
                    candidates.append(new_state)
                    all_seen.append(new_state)

            if not candidates:
                break

            candidates.sort(key=lambda s: s.score, reverse=True)
            beam = candidates[: self.beam_width]

        all_seen.sort(key=lambda s: s.score, reverse=True)
        return all_seen[:k]
