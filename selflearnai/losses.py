"""Loss functions for Stage 1 alignment and Stage 2 concept training.

Three losses in scope:
  • InfoNCE — symmetric contrastive loss with temperature.
  • VICReg — variance + covariance regularization, anti-collapse.
  • Projector — small projection head for VICReg (decouples reps from
                spreading pressure). Same pattern as selflearn.py:Projector.

NO autoregressive / next-token loss anywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# InfoNCE — symmetric contrastive
# ---------------------------------------------------------------------------
def info_nce_loss(
    a: torch.Tensor,           # (B, D) — query side, gradients flow
    b: torch.Tensor,           # (B, D) — key side, may or may not have gradients
    temperature: float = 0.07,
    extra_negatives: torch.Tensor | None = None,   # (M, D) — hard-negative bank
) -> torch.Tensor:
    """Symmetric InfoNCE in the style of CLIP.

    Args:
        a, b: paired embeddings. Row i of a corresponds to row i of b.
        temperature: standard contrastive temperature (CLIP uses 0.07).
        extra_negatives: optional bank of hard negatives (e.g., from a memory
                         buffer). If supplied, both directions get extra negatives.

    Returns:
        Scalar loss = 0.5 * (CE_a→b + CE_b→a).
    """
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    logits_ab = a_n @ b_n.T / temperature                    # (B, B)
    if extra_negatives is not None and extra_negatives.numel() > 0:
        neg_n = F.normalize(extra_negatives, dim=-1).detach()
        a_neg = a_n @ neg_n.T / temperature                  # (B, M)
        b_neg = b_n @ neg_n.T / temperature                  # (B, M)
        logits_a = torch.cat([logits_ab, a_neg], dim=1)      # (B, B+M)
        logits_b = torch.cat([logits_ab.T, b_neg], dim=1)    # (B, B+M)
    else:
        logits_a = logits_ab
        logits_b = logits_ab.T
    target = torch.arange(a.size(0), device=a.device)
    loss_a = F.cross_entropy(logits_a, target)
    loss_b = F.cross_entropy(logits_b, target)
    return 0.5 * (loss_a + loss_b)


# ---------------------------------------------------------------------------
# VICReg — variance + covariance regularization (anti-collapse)
# ---------------------------------------------------------------------------
def vicreg_loss(
    z: torch.Tensor,                # (N, D) — N samples, D features
    gamma: float = 1.0,             # variance hinge target
    var_coeff: float = 2.5,
    cov_coeff: float = 0.25,
) -> torch.Tensor:
    """VICReg variance + off-diagonal covariance penalty.

    Crucial shape rule: caller must reshape any (B, L, D) tensor to (N, D)
    BEFORE calling — variance is computed per-dim *across samples*, not
    across positions. This is the same locked rule as selflearn.py.

    Args:
        z: (N, D). Each row is one embedding vector.
        gamma: each per-dim stddev should be ≥ gamma. Hinge loss below.
        var_coeff: weight on variance term.
        cov_coeff: weight on off-diagonal covariance term.
    """
    N, D = z.shape
    z_centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    var_loss = torch.mean(F.relu(gamma - std))
    if N > 1:
        cov = (z_centered.T @ z_centered) / (N - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        cov_loss = (off ** 2).sum() / D
    else:
        cov_loss = torch.tensor(0.0, device=z.device)
    return var_coeff * var_loss + cov_coeff * cov_loss


# ---------------------------------------------------------------------------
# Projector — small head used before VICReg
# ---------------------------------------------------------------------------
class Projector(nn.Module):
    """Standard VICReg projection head. Input dim → proj_dim → proj_dim.

    Applied to embeddings before the VICReg regularizer. Decouples the
    encoder's representations from the spreading pressure: the encoder can
    produce compact informative reps while the projector absorbs the
    variance budget. Standard practice in VICReg / SimCLR / BYOL.
    """

    def __init__(self, in_dim: int, proj_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
