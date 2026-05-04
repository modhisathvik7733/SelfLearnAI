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


class MultiHeadConceptOperator(nn.Module):
    """Multi-head concept operator with input-conditioned routing.

    The architectural extension motivated by the cross-category-preserving
    failure (young-animal: ~50% ceiling). The hypothesis: a single shared
    direction can move embeddings into the target *region* but cannot
    preserve source-specific identity (horse → baby-region vs horse → foal
    specifically). K direction vectors + a small router lets different
    sources route to different shift directions.

    Architecture:
      • K learned 'concept directions' v_1...v_K of dim D.
      • K learned scalar magnitudes alpha_1...alpha_K.
      • A small Router MLP: D → K logits (softmax to get head weights).
      • A SHARED residual MLP that takes (z, weighted_v) → delta.

    Compared to ConceptOperator:
      Single-head : one v, one alpha, one residual MLP.
      Multi-head  : K v's + K alphas + router + shared residual.

    Sharing the residual MLP keeps parameter count modest. The router and
    K v's add only ~25K params over the single-head version (for K=3,
    D=384, router_hidden=64).

    Backward compatible: when K=1, behavior reduces to (approximately) the
    single-head ConceptOperator (the router becomes a constant).
    """

    def __init__(
        self,
        dim: int = SHARED_DIM,
        num_heads: int = 3,
        router_hidden: int = 64,
        mlp_hidden: int = 192,
    ):
        super().__init__()
        self.num_heads = num_heads
        # K learned direction vectors.
        self.v = nn.Parameter(torch.randn(num_heads, dim) * 0.02)
        # K learned scalar magnitudes.
        self.alpha = nn.Parameter(torch.ones(num_heads))
        # Router: input → soft head weights.
        self.router = nn.Sequential(
            nn.Linear(dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_heads),
        )
        # Shared content-sensitive residual.
        self.residual = nn.Sequential(
            nn.Linear(2 * dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., dim) → (..., dim)."""
        # Soft routing weights per input (default temperature is 1).
        logits = self.router(z)                                          # (..., K)
        weights = torch.softmax(logits, dim=-1)                          # (..., K)

        # Per-head direction = alpha[k] * v[k]  →  shape (K, D).
        head_dirs = self.v * self.alpha.unsqueeze(-1)                    # (K, D)

        # Weighted v: (..., K) @ (K, D) = (..., D).
        weighted_v = weights @ head_dirs                                 # (..., D)

        # Residual conditioned on (z, weighted_v).
        delta = weighted_v + self.residual(
            torch.cat([z, weighted_v], dim=-1)
        )
        return z + delta

    @torch.no_grad()
    def head_assignments(self, z: torch.Tensor) -> torch.Tensor:
        """Diagnostic: return the soft head weights for each input.
        Use to inspect which sources route to which heads after training."""
        return torch.softmax(self.router(z), dim=-1)


class MultiHeadInverseConceptOperator(MultiHeadConceptOperator):
    """Same architecture; trained on REVERSED pairs. Independent params from
    the forward operator so the inverse can learn its own routing pattern."""
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
