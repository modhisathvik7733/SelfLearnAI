"""Ψ-shift clustering + per-cluster operator-consistency check (Task 1.5.2).

Reads (source, target) pairs from the wake buffer's unexplained
channels, computes residual shifts ΔΨ = z_target − z_source, clusters
them in latent space, and for each cluster trains a fresh
ConceptOperator and verifies it captures the cluster's shift
*consistently across all members* — not just on the centroid.

This is safeguard #1 of Stage 1.5 (plan §19.9 + memory): centroid
similarity alone produces fake clusters because:
  - different concepts can overlap in vector space; their residuals
    don't have to;
  - noise creates clusters whose centroid is reasonable but whose
    members don't share a common shift.

The operator-consistency check is the principled fix: if a cluster
represents a real concept, a single learned operator should map every
(src_i, tgt_i) pair in it with similar fidelity. If `cos(op(src_i),
tgt_i)` has high variance across members, the cluster is heterogeneous
and is rejected.

Pure-torch implementation: KMeans with multiple inits + silhouette
score, both built in to avoid adding sklearn as a dependency for
the modest clustering loads (typically N=30-200 ΔΨ vectors per sleep
cycle).

Public API:
  - compute_psi_shifts(encode_fn, pairs)        -> Tensor[N, dim]
  - kmeans_cluster(x, k, ...)                   -> KMeansResult
  - silhouette_score(x, labels)                 -> float
  - cluster_with_silhouette_sweep(x, k_min, k_max, ...) -> SweepResult
  - train_quick_operator(z_src, z_tgt, ...)     -> ConceptOperator
  - operator_consistency(op, z_src, z_tgt)      -> ConsistencyResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from selflearnai.concepts.operator import ConceptOperator


# ---------------------------------------------------------------------------
# Encoding ΔΨ
# ---------------------------------------------------------------------------

def compute_psi_shifts(
    encode_fn: Callable[[list[str]], torch.Tensor],
    pairs: Sequence[tuple[str, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode pairs and return (z_src, z_tgt, shifts).

    Shifts are residuals z_tgt - z_src in the encoder's native space
    (no normalization). Sleep clustering uses these directly.
    """
    if not pairs:
        raise ValueError("pairs is empty")
    sources = [p[0] for p in pairs]
    targets = [p[1] for p in pairs]
    z_src = encode_fn(sources)
    z_tgt = encode_fn(targets)
    if z_src.shape != z_tgt.shape:
        raise ValueError(
            f"encoded source/target shape mismatch: {z_src.shape} vs {z_tgt.shape}"
        )
    return z_src, z_tgt, z_tgt - z_src


# ---------------------------------------------------------------------------
# KMeans (Lloyd's algorithm, multiple inits, pure torch)
# ---------------------------------------------------------------------------

@dataclass
class KMeansResult:
    labels: torch.Tensor          # [N], int
    centroids: torch.Tensor       # [k, dim]
    inertia: float                # sum of squared distances to nearest centroid
    iterations: int               # iters actually run on the winning init
    k: int


def _pairwise_sq_dist(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """[N, dim] x [k, dim] -> [N, k] pairwise squared distances."""
    return torch.cdist(x, c, p=2.0).pow(2)


def _kmeans_one_init(
    x: torch.Tensor, k: int, *, max_iter: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Single-init Lloyd's algorithm. Returns (labels, centroids, inertia, iters)."""
    n = x.shape[0]
    if n < k:
        raise ValueError(f"need n >= k, got n={n}, k={k}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    # k-means++-lite: random sample without replacement.
    init_idx = torch.randperm(n, generator=g)[:k]
    centroids = x[init_idx].clone()
    labels = torch.zeros(n, dtype=torch.long, device=x.device)
    last_labels = torch.full_like(labels, -1)
    iters = 0
    for it in range(max_iter):
        d2 = _pairwise_sq_dist(x, centroids)        # [n, k]
        labels = d2.argmin(dim=-1)
        if torch.equal(labels, last_labels):
            iters = it
            break
        # Recompute centroids; reseed empty clusters from the farthest point.
        new_centroids = centroids.clone()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = x[mask].mean(dim=0)
            else:
                # Empty cluster: take the point with the largest current
                # min-distance to any centroid (farthest from the pack).
                far_idx = int(d2.min(dim=-1).values.argmax().item())
                new_centroids[j] = x[far_idx]
        centroids = new_centroids
        last_labels = labels
        iters = it + 1
    d2_final = _pairwise_sq_dist(x, centroids)
    inertia = float(d2_final.gather(1, labels.unsqueeze(-1)).sum().item())
    return labels, centroids, inertia, iters


def kmeans_cluster(
    x: torch.Tensor,
    k: int,
    *,
    n_init: int = 8,
    max_iter: int = 100,
    seed: int = 0,
) -> KMeansResult:
    """KMeans with `n_init` random restarts; pick the lowest-inertia run."""
    best: Optional[tuple[torch.Tensor, torch.Tensor, float, int]] = None
    for s in range(n_init):
        run = _kmeans_one_init(x, k, max_iter=max_iter, seed=seed + s)
        if best is None or run[2] < best[2]:
            best = run
    assert best is not None
    labels, centroids, inertia, iters = best
    return KMeansResult(
        labels=labels.cpu(),
        centroids=centroids.cpu(),
        inertia=inertia,
        iterations=iters,
        k=k,
    )


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------

def silhouette_score(x: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean silhouette s_i = (b_i - a_i) / max(a_i, b_i).

    a_i: mean distance from i to other points in the same cluster.
    b_i: min over other clusters of mean distance from i to points
         in that cluster.

    Single-element clusters contribute s_i = 0 (degenerate). Pure
    Euclidean distance to match the KMeans objective.
    """
    n = x.shape[0]
    uniq = torch.unique(labels)
    if uniq.numel() < 2 or n < 2:
        return 0.0
    d = torch.cdist(x, x, p=2.0)        # [n, n]
    s = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        own = labels[i]
        same_mask = (labels == own).clone()
        same_mask[i] = False
        if same_mask.sum().item() == 0:
            s[i] = 0.0
            continue
        a_i = float(d[i, same_mask].mean().item())
        b_i = float("inf")
        for c in uniq.tolist():
            if c == int(own.item()):
                continue
            mask = labels == c
            if mask.any():
                mean_d = float(d[i, mask].mean().item())
                if mean_d < b_i:
                    b_i = mean_d
        denom = max(a_i, b_i)
        s[i] = 0.0 if denom == 0 else (b_i - a_i) / denom
    return float(s.mean().item())


# ---------------------------------------------------------------------------
# Sweep K by silhouette
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    best_k: int
    best_silhouette: float
    best_result: KMeansResult
    per_k: dict[int, dict[str, float]] = field(default_factory=dict)


def cluster_with_silhouette_sweep(
    x: torch.Tensor,
    *,
    k_min: int = 2,
    k_max: int = 6,
    n_init: int = 8,
    max_iter: int = 100,
    seed: int = 0,
    normalize_inputs: bool = True,
) -> SweepResult:
    """Run KMeans for k in [k_min, k_max], pick the k with the highest
    silhouette. Tiebreak by lower inertia.

    `normalize_inputs=True` (default): L2-normalize each row of `x`
    before clustering. ΔΨ residuals encode concept identity as a
    *direction* in encoder space — magnitude carries word-frequency
    and encoder-norm noise that's irrelevant to which concept is
    being expressed. On unit vectors, Euclidean distance is a monotone
    of cosine distance, so KMeans + silhouette + operator-consistency
    all score the same geometric structure.

    Setting `normalize_inputs=False` is for callers that need to
    distinguish concepts that share a direction but differ in
    magnitude (e.g. comparative vs superlative may end up on the
    same axis at different distances). For Stage 1.5 discovery
    smoke + sleep cycles, the default is correct.
    """
    n = x.shape[0]
    k_max_eff = min(k_max, n - 1)
    if k_max_eff < k_min:
        raise ValueError(f"too few points (n={n}) for k_min={k_min}")
    if normalize_inputs:
        x = F.normalize(x, dim=-1)
    per_k: dict[int, dict[str, float]] = {}
    best_k = k_min
    best_sil = -2.0
    best_result: Optional[KMeansResult] = None
    for k in range(k_min, k_max_eff + 1):
        res = kmeans_cluster(x, k, n_init=n_init, max_iter=max_iter, seed=seed)
        sil = silhouette_score(x, res.labels)
        per_k[k] = {"silhouette": sil, "inertia": res.inertia}
        is_better = (
            sil > best_sil
            or (sil == best_sil and best_result is not None and res.inertia < best_result.inertia)
        )
        if is_better:
            best_sil = sil
            best_k = k
            best_result = res
    assert best_result is not None
    return SweepResult(
        best_k=best_k,
        best_silhouette=best_sil,
        best_result=best_result,
        per_k=per_k,
    )


# ---------------------------------------------------------------------------
# Quick operator + consistency
# ---------------------------------------------------------------------------

def train_quick_operator(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    *,
    dim: int,
    device: str = "cuda",
    seed: int = 0,
    epochs: int = 300,
    lr: float = 1e-3,
) -> ConceptOperator:
    """Light operator fit for cluster-consistency checks. Same shape as
    the canonical `train_operator` from stage1_planner_beam_smoke.py
    but with a default 300 epochs (enough to evaluate consistency,
    not enough to fully validate generalization)."""
    torch.manual_seed(seed)
    op = ConceptOperator(dim=dim).to(device)
    z_src = z_src.to(device)
    z_tgt = z_tgt.to(device)
    opt = torch.optim.AdamW(op.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(op(z_src), z_tgt)
        loss.backward()
        opt.step()
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op


@dataclass
class ConsistencyResult:
    """Per-cluster consistency check (safeguard #1).

    `passes` requires both:
      - mean(cos) >= mean_threshold      (operator captures the shift)
      - var(cos)  <  var_threshold       (it captures it CONSISTENTLY)
    """
    n: int
    mean_cos: float
    var_cos: float
    min_cos: float
    max_cos: float
    mean_threshold: float
    var_threshold: float
    passes_mean: bool
    passes_var: bool
    passes: bool


@torch.no_grad()
def operator_consistency(
    op: ConceptOperator,
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
    *,
    mean_threshold: float = 0.80,
    var_threshold: float = 0.10,
) -> ConsistencyResult:
    pred = op(z_src)
    cos = F.cosine_similarity(pred, z_tgt, dim=-1)         # [n]
    mean = float(cos.mean().item())
    var = float(cos.var(unbiased=False).item())
    mn = float(cos.min().item())
    mx = float(cos.max().item())
    pm = mean >= mean_threshold
    pv = var < var_threshold
    return ConsistencyResult(
        n=int(cos.numel()),
        mean_cos=mean,
        var_cos=var,
        min_cos=mn,
        max_cos=mx,
        mean_threshold=mean_threshold,
        var_threshold=var_threshold,
        passes_mean=pm,
        passes_var=pv,
        passes=pm and pv,
    )
