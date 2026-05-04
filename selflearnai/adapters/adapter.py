"""Trainable adapter networks — the only modules that get gradients in Stage 1.

Three adapters per the star-topology design (plan section 1):

  adapter_c (CLIP)   : 512  → 384.  Passive anchor (VICReg only, no alignment loss).
  adapter_t (GTE)    : 768  → 384.  Trained to match adapter_c via InfoNCE.
  adapter_v (V-JEPA) : 1024 → 384.  Trained to match adapter_c via InfoNCE,
                                     with explicit identity/quantity/relational
                                     factorization built into the architecture.

All adapters output to the locked shared dim (384). The factored visual adapter
is the architectural commitment that prevents identity-count entanglement at
scale (validated in toy via selflearn.py:VisualSceneEncoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from selflearnai import SHARED_DIM


# ---------------------------------------------------------------------------
# Generic adapter — used for CLIP (passive anchor) and GTE (text)
# ---------------------------------------------------------------------------
class Adapter(nn.Module):
    """Linear → GELU → Linear → LayerNorm.

    Used for: adapter_c (CLIP, 512→384) and adapter_t (GTE, 768→384).

    Shape discipline (locked): forward accepts (B, in_dim), returns (B, SHARED_DIM).
    """

    def __init__(self, in_dim: int, out_dim: int = SHARED_DIM, hidden: int | None = None):
        super().__init__()
        h = hidden if hidden is not None else max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Linear(h, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Factored visual adapter — three branches with different invariances
# ---------------------------------------------------------------------------
class FactoredVisualAdapter(nn.Module):
    """Three-branch visual adapter mapping V-JEPA-2 patch features → SHARED_DIM.

    Architecture (plan section 1):

      identity_branch  (max-pool over patches  → linear → SHARED_DIM)
      quantity_branch  (mean-pool over patches → linear → SHARED_DIM)
      relational_branch (attention-pool        → linear → SHARED_DIM)

      output = identity + quantity + relational      ← additive, NOT concat

    The additive combination is the architectural commitment: the embedding
    is structurally a sum of independently-pooled factors. Concept operators
    later in the pipeline can route their training pressure to whichever
    factor they're about (plurality → quantity branch, identity-of-noun →
    identity branch, spatial relations → relational branch).

    Shape discipline (locked):
      • Input: (B, N_patches, in_dim)   — V-JEPA-2 last_hidden_state.
      • Output: (B, SHARED_DIM)         — additive of three SHARED_DIM tensors.
      • Pooling operations have different INVARIANCES (max → count-invariant;
        mean → identity-invariant in expectation; attention → context-sensitive).
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = SHARED_DIM):
        super().__init__()
        # Each branch: native_dim → SHARED_DIM. Tiny LayerNorms anchor scale.
        self.identity_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim),
        )
        self.quantity_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim),
        )
        self.relational_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim),
        )
        # Attention-pool head: a single learnable query attends over all patches.
        self.attn_query = nn.Parameter(torch.randn(in_dim) * 0.02)
        self.attn_scale = (in_dim ** -0.5)

    def _attention_pool(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, N, D) → (B, D). Single learned query attends to all
        patches; weighted-sum the values. Captures spatial structure in a
        permutation-invariant way."""
        B, N, D = patches.shape
        # (B, N) attention scores
        scores = (patches @ self.attn_query) * self.attn_scale
        weights = torch.softmax(scores, dim=-1)                          # (B, N)
        return (patches * weights.unsqueeze(-1)).sum(dim=1)              # (B, D)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: (B, N, in_dim) → (B, SHARED_DIM)."""
        # Identity: max-pool over patches. Invariant to count: peak detector
        # activation is the same whether one stamp or seven are present.
        identity_feat, _ = patches.max(dim=1)                            # (B, in_dim)
        # Quantity: mean-pool. Scales with how filled the scene is.
        quantity_feat = patches.mean(dim=1)                              # (B, in_dim)
        # Relational: attention-pool. Captures spatial / contextual structure.
        relational_feat = self._attention_pool(patches)                  # (B, in_dim)

        return (
            self.identity_proj(identity_feat)
            + self.quantity_proj(quantity_feat)
            + self.relational_proj(relational_feat)
        )


# ---------------------------------------------------------------------------
# Convenience factory — builds the three adapters as a bundle
# ---------------------------------------------------------------------------
class AdapterBundle(nn.Module):
    """Holds adapter_c, adapter_t, adapter_v together. The training script
    optimizes only the GTE and V-JEPA adapters via alignment losses; the CLIP
    anchor adapter only sees VICReg (passive).
    """

    def __init__(
        self,
        clip_dim: int,
        gte_dim: int,
        vjepa_dim: int,
        shared_dim: int = SHARED_DIM,
    ):
        super().__init__()
        self.adapter_c = Adapter(clip_dim, shared_dim)
        self.adapter_t = Adapter(gte_dim, shared_dim)
        self.adapter_v = FactoredVisualAdapter(vjepa_dim, shared_dim)
