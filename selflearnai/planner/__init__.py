"""Multi-step planner over the concept-operator library (Stage 1).

Given a start embedding and a goal embedding (or goal description),
the planner searches for an operator chain whose composed application
to the start lands close to the goal in latent space. The output is
a `PlanState` carrying the final embedding, the operator names applied
in order, and per-step / final scores.

Stage 1 build-out (per plan §19.7):
  - `beam.py` (Task 1.6, this file): plain beam search with cosine-to-
    goal as the only ordering signal. Skeleton; will be extended.
  - `prior.py`   (Task 1.8): operator-prior MLP — learns p(op | ψ) so
    the beam expands promising operators first.
  - `value.py`   (Task 1.9): value-function MLP — learns V(ψ, ψ_goal)
    expected steps remaining, used to prune unlikely subtrees early.
  - `verifier.py` (Task 1.10): per-step type / axiom / drift / conformal
    gates; failing chains are halted and surfaced honestly.
  - `trace.py`    (Task 1.11): Ψ-program serialization + replay.
  - `macros.py`   (later): compound operators discovered by Stage 1.5
    wake/sleep are flattened here for shorter effective depths.
"""
from .beam import BeamSearchPlanner, PlanState

__all__ = ["BeamSearchPlanner", "PlanState"]
