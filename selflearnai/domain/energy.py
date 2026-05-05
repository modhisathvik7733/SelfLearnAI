"""Per-domain energy/density models for Stage 3 (sub-task 3.2).

Two implementations, both small + fast + non-LLM:

  GaussianEnergy
    Mahalanobis distance to the in-domain ψ cluster's mean and
    covariance. Zero training (closed-form fit). Strong baseline for
    high-dim ψ vectors when the in-domain manifold is roughly Gaussian
    in encoder space.
    E(ψ) = (ψ−μ)ᵀ Σ⁻¹ (ψ−μ)

  MLPEnergyModel
    Small MLP (~250K params per plan §9.3 budget) trained with
    contrastive NCE-style loss. Positives = in-domain ψ. Negatives =
    self-supervised: Gaussian noise scaled to ψ_train statistics.
    More expressive than Gaussian when the in-domain manifold has
    non-elliptical structure.

Both expose the same interface (.fit(psi_train), .energy(psi),
.save/.load) so the orchestrator (3.4) can pick whichever scores
better on each domain's probe set.

Plan §19.14's "no LLMs in architecture" rule trivially holds — both
models are pure density estimators with no language-modeling head.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class EnergyModelBase(ABC):
    """Abstract interface for per-domain energy models.

    Convention: lower energy = more in-domain. Out-of-domain inputs
    should receive higher energy.
    """

    @abstractmethod
    def fit(self, psi_train: torch.Tensor) -> dict[str, Any]:
        """Fit on in-domain ψ. Returns a small dict of training stats."""

    @abstractmethod
    def energy(self, psi: torch.Tensor) -> torch.Tensor:
        """Score a batch of ψ. Returns 1-D tensor of energies."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "EnergyModelBase":
        ...


# ---------------------------------------------------------------------------
# Gaussian baseline (Mahalanobis distance)
# ---------------------------------------------------------------------------

class GaussianEnergy(EnergyModelBase):
    """Mahalanobis-distance energy.

    fit(): closed-form. Computes mean and a regularized inverse covariance
    of the training ψ. Reg coefficient defaults to 1e-3 of the trace —
    necessary because ψ_train is typically much smaller than D so the
    sample covariance is rank-deficient.

    energy(): (ψ−μ)ᵀ Σ⁻¹ (ψ−μ). Higher = more anomalous.

    Zero parameters to train. Fits in seconds.
    """

    def __init__(self, dim: int, ridge: float = 1e-3) -> None:
        self.dim = dim
        self.ridge = ridge
        self.mean: torch.Tensor | None = None
        self.precision: torch.Tensor | None = None
        self.training_n: int = 0

    def fit(self, psi_train: torch.Tensor) -> dict[str, Any]:
        psi_train = psi_train.float().detach()
        n, d = psi_train.shape
        if d != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {d}")
        self.mean = psi_train.mean(dim=0)
        centered = psi_train - self.mean
        # Sample covariance with ridge regularization on the diagonal.
        cov = (centered.T @ centered) / max(n - 1, 1)
        diag_mean = float(torch.diag(cov).mean().item())
        cov = cov + self.ridge * diag_mean * torch.eye(d, device=cov.device)
        # Cholesky-based inverse for stability.
        L = torch.linalg.cholesky(cov)
        self.precision = torch.cholesky_inverse(L)
        self.training_n = n
        return {
            "n_train": n,
            "ridge_used": self.ridge * diag_mean,
            "diag_mean": diag_mean,
        }

    def energy(self, psi: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.precision is None:
            raise RuntimeError("call fit() first")
        diff = psi.float() - self.mean
        # (ψ-μ)ᵀ P (ψ-μ) per row
        return ((diff @ self.precision) * diff).sum(dim=-1)

    def save(self, path: str | Path) -> None:
        if self.mean is None or self.precision is None:
            raise RuntimeError("nothing to save — call fit() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "GaussianEnergy",
                "dim": self.dim,
                "ridge": self.ridge,
                "mean": self.mean,
                "precision": self.precision,
                "training_n": self.training_n,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GaussianEnergy":
        d = torch.load(str(path), map_location="cpu", weights_only=True)
        if d.get("kind") != "GaussianEnergy":
            raise ValueError(f"checkpoint kind={d.get('kind')!r}, expected GaussianEnergy")
        m = cls(dim=d["dim"], ridge=d["ridge"])
        m.mean = d["mean"]
        m.precision = d["precision"]
        m.training_n = int(d["training_n"])
        return m


# ---------------------------------------------------------------------------
# MLP energy model (NCE-style contrastive training)
# ---------------------------------------------------------------------------

class MLPEnergyModel(nn.Module, EnergyModelBase):
    """Small MLP that scores a ψ vector to a scalar energy.

    Architecture: Linear(D, h1) → GELU → Linear(h1, h2) → GELU → Linear(h2, 1).
    Default h1=256, h2=64. ~263K params for D=1024 — within the §9.3 budget.

    Training loss (hinge-style contrastive):
      L = mean(relu(E(positive) - E(negative) + margin))
    Positives: training ψ.
    Negatives: synthesized as Gaussian noise scaled to match ψ_train stats.

    The synthetic-negative trick assumes the in-domain manifold is
    well-separated from the matched-statistics Gaussian envelope — true
    in encoder spaces where domain-specific structure dominates over
    marginal statistics. Plan §9.3 used the same negative-sampling
    pattern.
    """

    def __init__(
        self,
        dim: int = 1024,
        hidden1: int = 256,
        hidden2: int = 64,
    ) -> None:
        nn.Module.__init__(self)
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, 1),
        )
        # Stats remembered after fit() for save/load + neg sampling.
        self.register_buffer(
            "_train_mean", torch.zeros(dim), persistent=True,
        )
        self.register_buffer(
            "_train_std", torch.ones(dim), persistent=True,
        )
        self._fitted = False

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        return self.net(psi.float()).squeeze(-1)

    def energy(self, psi: torch.Tensor) -> torch.Tensor:
        return self.forward(psi)

    def fit(
        self,
        psi_train: torch.Tensor,
        *,
        steps: int = 3000,
        batch_size: int = 64,
        lr: float = 1e-3,
        margin: float = 1.0,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        log_every: int = 500,
        seed: int = 0,
    ) -> dict[str, Any]:
        torch.manual_seed(seed)
        psi_train = psi_train.float().detach().to(device)
        self.to(device)
        N = psi_train.size(0)
        if N == 0:
            raise ValueError("psi_train is empty")
        with torch.no_grad():
            self._train_mean = psi_train.mean(dim=0).clone()
            self._train_std = (psi_train.std(dim=0) + 1e-6).clone()
        opt = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay,
        )
        history: list[dict] = []
        self.train()
        for step in range(steps):
            opt.zero_grad()
            idx = torch.randint(0, N, (batch_size,), device=device)
            pos = psi_train[idx]                                # [B, D]
            # Negatives: Gaussian noise scaled to match ψ_train stats.
            neg = (
                self._train_mean.unsqueeze(0)
                + self._train_std.unsqueeze(0)
                * torch.randn(batch_size, self.dim, device=device)
            )                                                   # [B, D]
            E_pos = self.forward(pos)
            E_neg = self.forward(neg)
            loss = torch.relu(E_pos - E_neg + margin).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            opt.step()
            if step % log_every == 0 or step == steps - 1:
                history.append({
                    "step": step,
                    "loss": float(loss.item()),
                    "E_pos_mean": float(E_pos.mean().item()),
                    "E_neg_mean": float(E_neg.mean().item()),
                })
        self._fitted = True
        return {
            "n_train": N,
            "steps": steps,
            "loss_history": history,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "MLPEnergyModel",
                "dim": self.dim,
                "state_dict": self.state_dict(),
                "fitted": self._fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPEnergyModel":
        d = torch.load(str(path), map_location="cpu", weights_only=True)
        if d.get("kind") != "MLPEnergyModel":
            raise ValueError(f"checkpoint kind={d.get('kind')!r}, expected MLPEnergyModel")
        m = cls(dim=d["dim"])
        m.load_state_dict(d["state_dict"])
        m._fitted = bool(d.get("fitted", False))
        return m


# ---------------------------------------------------------------------------
# ROC-AUC (Mann-Whitney U via vectorized pair comparison)
# ---------------------------------------------------------------------------

def roc_auc(
    y_true: torch.Tensor,        # [N], 0/1 binary (1 = "positive class")
    scores: torch.Tensor,        # [N], higher = more "positive class"
) -> float:
    """Vectorized ROC-AUC. Positive class is the one we want scored higher.

    For energy-model evaluation we set positive class = "out-of-domain"
    (since out-of-domain should have HIGHER energy).
    """
    if y_true.numel() != scores.numel():
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, scores {scores.shape}")
    y_true = y_true.flatten().bool()
    scores = scores.flatten().float()
    pos_scores = scores[y_true]
    neg_scores = scores[~y_true]
    n_pos, n_neg = pos_scores.numel(), neg_scores.numel()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Pairwise differences. For 51,840 pairs (120 in-dom × 432 out-of-dom)
    # this is trivial.
    diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # [n_pos, n_neg]
    n_correct = (diff > 0).float().sum() + 0.5 * (diff == 0).float().sum()
    return float(n_correct.item() / (n_pos * n_neg))
