"""Operator prior `P(op | ψ)` — small MLP over encoder embeddings.

Given a state ψ in the encoder's latent space, predicts a distribution
over operators in the library. Used by the beam-search planner to
prune the operator-expansion fanout: instead of expanding all K
operators at every state, expand only the top-N by prior probability.

For a 7-operator library at depth 5:
  - Without prior: 7^5 = 16807 expansion paths.
  - With prior, top-N=3: 3^5 = 243 paths.  (~70× pruning.)

The prior is small (~50K params for D=768 input) and trained on
(source_word_embedding, correct_concept) pairs synthesized from each
concept's existing training data. After training the prior is frozen
and registered with the planner as an optional component.

Important caveat: the prior is trained on **source-word embeddings**,
which sit in the same region of latent space as the *first state* in
a planning trace. After one or more operators have been applied, the
intermediate ψ may have moved into a region the prior wasn't trained
on. The prior's predictions on intermediates are therefore noisier
than at the start. We mitigate by keeping `top_k_operators` modest
(>= 3) so good operators still survive even if the prior misranks
them mid-chain. Stage 1.9's value function will eventually take over
the role of "is this state on a good path" for non-initial states.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OperatorPriorPrediction:
    """One prior prediction. Mostly for readability / JSON dumping."""
    op_name: str
    probability: float


class OperatorPriorMLP(nn.Module):
    """Small frozen-after-training MLP that scores operators given a state.

    Architecture: Linear(D, hidden) → GELU → Linear(hidden, n_ops).

    For D=768, n_ops=7, hidden=64:
      params = 768·64 + 64 + 64·7 + 7 = 49,591 ≈ 50K.

    The model is **encoder-agnostic**: callers pass embeddings of any
    fixed dimension; the prior was trained for that specific dim.

    Usage::

        prior = OperatorPriorMLP(encoder_dim=768, operator_names=("plural", "past_tense", ...))
        # ... train on (source_psi, concept_name) pairs ...
        prior.freeze()
        prior.predict_dict(psi)  # {op_name: prob}
        prior.predict_top_k(psi, k=3)  # [(op_name, prob), ...]

    Pass the bound `predict_dict` to BeamSearchPlanner via the
    `prior=` kwarg (it satisfies `Callable[[Tensor], dict[str, float]]`).
    """

    def __init__(
        self,
        encoder_dim: int,
        operator_names: tuple[str, ...] | list[str],
        hidden_dim: int = 64,
    ):
        super().__init__()
        if encoder_dim <= 0:
            raise ValueError(f"encoder_dim must be positive, got {encoder_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not operator_names:
            raise ValueError("operator_names must be non-empty")
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.operator_names: tuple[str, ...] = tuple(operator_names)
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.operator_names)),
        )

    @property
    def num_operators(self) -> int:
        return len(self.operator_names)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """Logits over operators. psi shape (B, D) or (D,) → (B, n_ops)."""
        if psi.dim() == 1:
            psi = psi.unsqueeze(0)
        return self.net(psi)

    @torch.no_grad()
    def predict_dict(self, psi: torch.Tensor) -> dict[str, float]:
        """Single-state prediction: {op_name: probability}."""
        if psi.dim() > 1:
            psi = psi.flatten()
        logits = self.forward(psi).squeeze(0)
        probs = F.softmax(logits, dim=-1)
        return {
            self.operator_names[i]: float(probs[i].item())
            for i in range(self.num_operators)
        }

    @torch.no_grad()
    def predict_top_k(
        self, psi: torch.Tensor, k: int,
    ) -> list[tuple[str, float]]:
        """Top-k operators by prior probability, descending."""
        d = self.predict_dict(psi)
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

    def freeze(self) -> "OperatorPriorMLP":
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_prior_callable(prior: OperatorPriorMLP) -> Callable[[torch.Tensor], dict[str, float]]:
    """Return a closure suitable for `BeamSearchPlanner(prior=...)`.

    Equivalent to `prior.predict_dict` but exposes a clean callable
    type signature; useful at planner-construction time.
    """
    @torch.no_grad()
    def _call(psi: torch.Tensor) -> dict[str, float]:
        return prior.predict_dict(psi)
    return _call
