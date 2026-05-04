"""ConceptOperator — the core concept-learning primitive.

A concept (plural, past tense, negation, comparative, possessive, ...) is
modeled as a *learned operator* in the shared latent space:

    forward(z)  =  z + α · v + residual_MLP([z; v])
    inverse(z)  =  z − α · v + inverse_residual_MLP([z; v])

where:
    v        : a single learned 384-dim direction (the "concept axis").
    α        : a learned scalar (allows the operator to modulate strength).
    residual : a small MLP that captures content-sensitive non-linearities.

KEY ARCHITECTURAL CONSTRAINTS (locked, see plan section 3):
  • ONE shared `v` for all inputs — no per-noun lookup.
  • ONE shared MLP — no per-noun MLP.
  • Inverse trained separately (different params) on reverse pairs.

This is the Ladder-2 operator from selflearn.py, lifted into the SHARED_DIM
space. Same shape, every concept; the library is `{plural, past, neg, ...}`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from selflearnai import SHARED_DIM


class ConceptOperator(nn.Module):
    """Forward operator. Maps z → z' where z' encodes the concept applied.

    Plural example: forward(emb("cat")) ≈ emb("cats").
    Past example:   forward(emb("walk")) ≈ emb("walked").
    """

    def __init__(self, dim: int = SHARED_DIM, mlp_hidden: int = 192):
        super().__init__()
        # Single shared concept direction. THIS IS THE WHOLE CONCEPT
        # PARAMETERIZATION at the linear level. Per-noun behavior is forbidden.
        self.v = nn.Parameter(torch.randn(dim) * 0.02)
        # Learnable scalar magnitude — lets the operator scale itself
        # without changing direction.
        self.alpha = nn.Parameter(torch.ones(1))
        # Content-sensitive residual. Same MLP weights for ALL inputs.
        self.residual = nn.Sequential(
            nn.Linear(2 * dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., dim) → (..., dim).

        Adds the linear concept shift (α·v) plus a content-sensitive residual.
        The residual sees both z and v — so it can produce content-dependent
        adjustments while still being conditioned on the SAME direction
        vector across all inputs.
        """
        # Broadcast v over the leading batch dims of z.
        v_b = self.v.expand_as(z)
        delta = self.alpha * self.v + self.residual(torch.cat([z, v_b], dim=-1))
        return z + delta


class InverseConceptOperator(ConceptOperator):
    """Same architecture as ConceptOperator, but trained on REVERSED pairs
    to undo the concept.

    Why a separate class (rather than just `forward(.)` with a flag)?
    Because the inverse may not be a perfect linear-mirror of the forward —
    the residual MLP captures asymmetries (e.g., "boys → boy" might require
    different content adjustments than "boy → boys"). Independent params
    let inverse learn its own non-linearities.

    Validates plurality is structurally invertible (algebraic group-like
    property), which selflearn.py's compositionality probe demonstrated at
    toy scale.
    """
    pass


class ConceptLibrary(nn.Module):
    """Holds a registered set of named operators for sequential or composed use.

    Used by Stage 2+ and by the reasoning planner. Registering a concept does
    NOT couple it to others — each operator has its own params.

    Example:
        lib = ConceptLibrary()
        lib.register("plural", ConceptOperator())
        lib.register("plural_inv", InverseConceptOperator())
        lib.register("past",  ConceptOperator())
        ...
        z2 = lib["plural"](z1)
    """

    def __init__(self):
        super().__init__()
        self.ops = nn.ModuleDict()

    def register(self, name: str, op: ConceptOperator) -> None:
        if name in self.ops:
            raise ValueError(f"Concept '{name}' already registered.")
        self.ops[name] = op

    def __getitem__(self, name: str) -> ConceptOperator:
        return self.ops[name]

    def __contains__(self, name: str) -> bool:
        return name in self.ops

    def names(self) -> list[str]:
        return list(self.ops.keys())
