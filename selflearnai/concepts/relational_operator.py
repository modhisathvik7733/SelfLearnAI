"""RelationalOperator — ψ-space operator for atomic relational reasoning.

The Stage 0 ConceptOperator handles single-input morphology:
    plural(ψ_cat) → ψ_cats

The RelationalOperator extends this to atomic relations:
    property(ψ_plant, axis="function")    → ψ_make_food
    property(ψ_plant, axis="requires")    → ψ_sunlight
    property(ψ_cat,   axis="sound")       → ψ_meow

Architecturally identical to ConceptOperator but with an axis-conditioned
direction vector instead of a single shared one. Each axis has its own
learnable direction; the residual MLP is shared across axes (so axes
can transfer knowledge — the operator can learn that 'sound axis' has
a similar abstract structure regardless of which animal).

Critical invariants:
  - Operates ENTIRELY in ψ-space. Inputs/outputs are vectors, never text.
  - Atomic: ONE relation per call. For multi-step reasoning ('why does
    X need Y?'), the planner composes multiple RelationalOperator
    calls — never one fat operator.
  - Held-out generalization is the validation criterion: a query about
    an entity not seen at training (e.g. 'snake → ?' for the 'sound'
    axis) must produce ψ near the right pool word ('hiss').

Per memory `feedback_relative_gates_in_encoder_space`: validate via
relative top-1 retrieval, not absolute cosine. The encoder's natural
geometry puts many words at high baseline cosine; what matters is
whether the operator's output is closer to the right value than to
distractors.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class RelationalOperator(nn.Module):
    """Axis-conditioned ψ-space operator for atomic relations.

    Architecture (mirrors Stage-0 ConceptOperator, with axis conditioning):

        for axis i:
            v_i      ∈ R^D       (per-axis learned direction)
            α_i      ∈ R         (per-axis magnitude scalar)
        shared MLP: 2D → mlp_hidden → D (content-sensitive residual)

        forward(z, axis_idx):
            v       = v_{axis_idx}
            α       = α_{axis_idx}
            delta   = α · v + MLP([z; v])
            return z + delta

    ~150K params for D=1024, mlp_hidden=192, num_axes=8 — within the
    plan §9.3 per-operator budget.

    The shared MLP allows cross-axis transfer (the network learns a
    generic 'apply axis-direction with content-sensitive residual'
    function). The per-axis v_i and α_i carry the axis-specific signal.
    """

    def __init__(
        self,
        dim: int = 1024,
        num_axes: int = 8,
        mlp_hidden: int = 192,
    ) -> None:
        super().__init__()
        if num_axes < 1:
            raise ValueError(f"num_axes must be >= 1, got {num_axes}")
        self.dim = dim
        self.num_axes = num_axes

        # Per-axis learnable directions. Initialized small so each axis
        # starts near identity; training pulls them apart.
        self.v = nn.Parameter(torch.randn(num_axes, dim) * 0.02)
        # Per-axis magnitude scalars.
        self.alpha = nn.Parameter(torch.ones(num_axes))
        # Shared content-sensitive residual (cross-axis transfer here).
        self.residual = nn.Sequential(
            nn.Linear(2 * dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(
        self,
        z_entity: torch.Tensor,
        axis_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        """ψ-space forward: (z_entity, axis) → z_result.

        Args:
            z_entity: (B, D) or (D,) entity ψ-vector(s).
            axis_idx: int or LongTensor of shape (B,). Picks per-axis
                direction + magnitude.

        Returns:
            z_result: same shape as z_entity.

        Invariant: this method NEVER touches strings. Pure ψ-space.
        """
        single_input = z_entity.dim() == 1
        if single_input:
            z_entity = z_entity.unsqueeze(0)
        B = z_entity.size(0)
        if isinstance(axis_idx, int):
            axis_idx = torch.full(
                (B,), axis_idx, dtype=torch.long, device=z_entity.device,
            )
        if axis_idx.dim() == 0:
            axis_idx = axis_idx.unsqueeze(0).expand(B)
        if axis_idx.shape != (B,):
            raise ValueError(
                f"axis_idx shape {tuple(axis_idx.shape)} does not match "
                f"batch dim {B}"
            )

        v = self.v[axis_idx]                      # (B, D)
        alpha = self.alpha[axis_idx].unsqueeze(-1)  # (B, 1)
        combined = torch.cat([z_entity, v], dim=-1)
        delta = alpha * v + self.residual(combined)
        z_result = z_entity + delta
        if single_input:
            z_result = z_result.squeeze(0)
        return z_result


class AxisVocabulary:
    """Mapping from axis names to integer indices, persisted alongside
    the trained RelationalOperator.

    Usage:
        vocab = AxisVocabulary.from_pairs(train_pairs)
        op = RelationalOperator(dim=1024, num_axes=len(vocab))
        idx = vocab["function"]
    """

    def __init__(self, axes: Iterable[str]) -> None:
        unique = sorted(set(axes))
        if not unique:
            raise ValueError("Cannot build AxisVocabulary on empty axes")
        self._axis_to_idx: dict[str, int] = {a: i for i, a in enumerate(unique)}
        self._idx_to_axis: list[str] = unique

    @classmethod
    def from_pairs(cls, pairs: list[dict]) -> "AxisVocabulary":
        return cls(p["axis"] for p in pairs)

    def __len__(self) -> int:
        return len(self._idx_to_axis)

    def __contains__(self, axis: str) -> bool:
        return axis in self._axis_to_idx

    def __getitem__(self, axis: str) -> int:
        if axis not in self._axis_to_idx:
            raise KeyError(
                f"axis {axis!r} not in vocabulary; known: {self._idx_to_axis}"
            )
        return self._axis_to_idx[axis]

    def name_of(self, idx: int) -> str:
        return self._idx_to_axis[idx]

    def axis_names(self) -> list[str]:
        return list(self._idx_to_axis)

    def to_dict(self) -> dict:
        return {"axes": list(self._idx_to_axis)}

    @classmethod
    def from_dict(cls, d: dict) -> "AxisVocabulary":
        return cls(d["axes"])
