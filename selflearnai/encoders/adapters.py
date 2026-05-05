"""Adapter head framework for encoder-space refinement (Task 0.5.6).

**Framework only.** Per the locked Stage 0.5 reactive-adapter policy
(plan §16, §19.2), no concrete adapter heads are trained here. Concrete
adapters get trained reactively — only when:
  (a) the encoder calibration suite (Task 0.5.5) flags a family below
      the watch threshold, AND
  (b) a downstream task in that family fails its acceptance gate.

This module provides the **substrate** for that future training:

- `AdapterHead`: small encoder-space transformation (`Linear → GELU →
  Linear → LayerNorm`) with a residual connection so an *untrained*
  adapter is close to the identity function. Safe to insert without
  retraining anything.

- `OperatorWithAdapter`: convenience wrapper that composes
  `op(adapter(z))` for inference-time use after an adapter has been
  trained against an operator.

- `AdapterRegistry`: per-task-family registry with safe fallback —
  `get(family)` returns None if no adapter is registered, so concepts
  that opt-in for a non-existent adapter run unmodified.

All adapters are designed to be **frozen after training** (consistent
with the no-catastrophic-forgetting policy: new adapters never modify
existing operators or other adapters).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AdapterHead(nn.Module):
    """Small frozen-after-training transformation in encoder space.

    Architecture (LoRA-style zero-init residual):
        z + LayerNorm( down( GELU( up(z) ) ) )
    where `up: Linear(D, hidden)`, `down: Linear(hidden, D)` is
    **zero-initialized** (both weight and bias). At random init the
    residual branch outputs zero exactly, so an untrained AdapterHead
    is the identity function. This is critical: inserting an
    unregistered AdapterHead must never break an upstream operator
    that was trained without it.

    Default `hidden_dim = D` (no compression). Parameter count is
    roughly 2·D² + 2·D for D-dim input and `hidden_dim = D`. For
    D=768 (GTE-base) ~1.2M params; for D=1024 (E5-large-v2) ~2.1M.

    Training (NOT in this file): a separate routine pulls contrastive
    pairs from the diagnostic test set (Task 0.5.4) or domain examples
    and optimizes the adapter for the chosen task family. As `down`'s
    weights move away from zero during training, the residual branch
    becomes meaningful. After training, call `.freeze()` and register
    into `AdapterRegistry`.
    """

    def __init__(
        self,
        dim: int,
        family: str = "generic",
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.family = family
        h = hidden_dim if hidden_dim is not None else dim

        # Up-projection (and activation) keep their default init.
        self.up = nn.Linear(dim, h)
        self.act = nn.GELU()
        # Down-projection is zero-initialized: both weight and bias.
        # This makes the entire residual branch output 0 at init.
        self.down = nn.Linear(h, dim)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        # LayerNorm with default affine init (γ=1, β=0): LayerNorm(0) = 0.
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Residual: untrained adapter ≡ identity (down zero-init guarantees).
        return z + self.norm(self.down(self.act(self.up(z))))

    def freeze(self) -> "AdapterHead":
        """Lock the adapter for inference. After freeze() the adapter
        is permanently in eval mode and no parameters require grad."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self


class OperatorWithAdapter(nn.Module):
    """Compose an adapter and a concept operator: `op(adapter(z))`.

    Use this at inference time when an operator was trained against
    an adapter (i.e. the operator's training inputs were already
    adapter-transformed). The adapter shapes the latent for the
    operator's task family; the operator applies the concept shift.

    Both components are kept as submodules so module APIs (e.g.,
    `.parameters()`, `.state_dict()`) work uniformly.
    """

    def __init__(self, adapter: AdapterHead, operator: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.operator = operator

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.operator(self.adapter(z))


class AdapterRegistry:
    """Per-family adapter registry with safe fallback semantics.

    `get(family)` returns the registered AdapterHead for that family,
    or None if none is registered. Concepts that opt-in for an adapter
    family with no registered adapter run unmodified (identity).

    Empty by default; the user must explicitly `register()` after
    training. The reactive policy means the project ships with an
    empty registry until evidence demands otherwise.
    """

    def __init__(self):
        self._adapters: dict[str, AdapterHead] = {}

    def register(self, family: str, adapter: AdapterHead) -> None:
        if family in self._adapters:
            raise ValueError(
                f"Adapter for family {family!r} already registered. "
                f"Versioning + rollback would go here when needed."
            )
        if adapter.family != family:
            raise ValueError(
                f"Adapter.family {adapter.family!r} does not match "
                f"registered family {family!r}"
            )
        # Locked policy: registered adapters MUST be frozen.
        # Verify by checking no parameter requires grad.
        if any(p.requires_grad for p in adapter.parameters()):
            raise ValueError(
                "Cannot register an unfrozen adapter. Call .freeze() first."
            )
        self._adapters[family] = adapter

    def get(self, family: str) -> Optional[AdapterHead]:
        return self._adapters.get(family)

    def families(self) -> list[str]:
        return list(self._adapters.keys())

    def __contains__(self, family: str) -> bool:
        return family in self._adapters

    def __len__(self) -> int:
        return len(self._adapters)


# Default global registry — empty until adapters are trained reactively.
DEFAULT_ADAPTER_REGISTRY = AdapterRegistry()
