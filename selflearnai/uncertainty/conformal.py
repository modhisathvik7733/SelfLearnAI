"""Score-based split-conformal calibration for a frozen ConceptOperator.

Given a trained operator and a calibration set disjoint from training,
the calibrator computes a single quantile q_hat such that prediction
sets

    Π(z_src) = { pool_j : -cos(op(z_src), z_pool_j) <= q_hat }

have empirical coverage >= (1 - alpha) on test data exchangeable with
the calibration set. Distribution-free, finite-sample. The guarantee
follows from the conformal coverage theorem (Vovk, Shafer & Gammerman
2005); the form used here is the standard split-conformal version
(Angelopoulos & Bates 2021, arXiv:2107.07511).

Replaces hardcoded `cos > 0.5`-style gates throughout the codebase
(Stage 0.5, Task 0.5.7) with calibrated set predictors.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F


class ConformalOperatorCalibrator:
    """Split-conformal calibration on top of a trained ConceptOperator.

    Nonconformity score: s(src, tgt) = -cos(op(z_src), z_tgt). Lower is
    better fit. Prediction sets at confidence (1 - alpha) include all
    pool words whose score is no worse than the (1 - alpha)-quantile of
    calibration scores.

    Important: calibration data MUST be disjoint from both training data
    (the operator's) and test data, or the coverage guarantee breaks.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha: float = alpha
        self.q_hat: float | None = None
        self.n_calib: int | None = None
        self.calib_scores: np.ndarray | None = None

    @torch.no_grad()
    def fit(
        self,
        op,
        encode_fn: Callable[[list[str]], torch.Tensor],
        calib_pairs: list[tuple[str, str]],
    ) -> "ConformalOperatorCalibrator":
        """Compute the calibration quantile q_hat from held-out pairs.

        Args:
            op: callable z_src -> z_pred (the trained operator, frozen
                and on the right device).
            encode_fn: list[str] -> Tensor of shape (n, d). Typically
                a closure around a foundation encoder's encode method.
            calib_pairs: list of (source, target) string pairs. MUST be
                disjoint from training and test pairs.

        Returns: self.
        """
        n = len(calib_pairs)
        if n < 2:
            raise ValueError(f"Need >= 2 calibration pairs, got {n}.")

        sources = [p[0] for p in calib_pairs]
        targets = [p[1] for p in calib_pairs]
        z_src = encode_fn(sources)
        z_tgt = encode_fn(targets)
        z_pred = op(z_src)
        cos = F.cosine_similarity(z_pred, z_tgt, dim=-1)
        scores = (-cos).detach().cpu().numpy().astype(float).ravel()

        # Conformal quantile: ⌈(n+1)(1-α)⌉-th order statistic on n samples.
        # If that exceeds n, alpha is too small for this calibration size
        # and the prediction set must contain everything (q_hat = +inf).
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        if k > n:
            self.q_hat = float("inf")
        else:
            self.q_hat = float(np.sort(scores)[k - 1])

        self.n_calib = n
        self.calib_scores = scores
        return self

    @torch.no_grad()
    def predict_set(
        self,
        op,
        encode_fn: Callable[[list[str]], torch.Tensor],
        src: str,
        pool: list[str],
    ) -> list[str]:
        """Calibrated prediction set for one source word."""
        if self.q_hat is None:
            raise RuntimeError("Call fit() before predict_set().")
        z_src = encode_fn([src])
        z_pred = op(z_src)
        z_pool = encode_fn(pool)
        cos = F.cosine_similarity(z_pred, z_pool, dim=-1)
        scores = (-cos).detach().cpu().numpy().astype(float).ravel()
        return [w for w, s in zip(pool, scores) if s <= self.q_hat]

    @torch.no_grad()
    def evaluate(
        self,
        op,
        encode_fn: Callable[[list[str]], torch.Tensor],
        test_pairs: list[tuple[str, str]],
        pool: list[str],
    ) -> dict:
        """Empirical coverage + set-size statistics on a disjoint test set."""
        if self.q_hat is None:
            raise RuntimeError("Call fit() before evaluate().")

        in_set_flags: list[int] = []
        set_sizes: list[int] = []
        for src, tgt in test_pairs:
            pred_set = self.predict_set(op, encode_fn, src, pool)
            in_set_flags.append(int(tgt in pred_set))
            set_sizes.append(len(pred_set))

        n_test = len(test_pairs)
        return {
            "alpha": self.alpha,
            "nominal_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(np.mean(in_set_flags)) if n_test else float("nan"),
            "n_test": n_test,
            "n_calib": self.n_calib,
            "in_set_flags": in_set_flags,
            "set_sizes": set_sizes,
            "mean_set_size": float(np.mean(set_sizes)) if n_test else float("nan"),
            "median_set_size": float(np.median(set_sizes)) if n_test else float("nan"),
            "max_set_size": int(np.max(set_sizes)) if n_test else 0,
            "q_hat": self.q_hat,
        }


def coverage_curve(
    op,
    encode_fn: Callable[[list[str]], torch.Tensor],
    calib_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    pool: list[str],
    alphas: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> dict:
    """Re-fit at multiple alpha levels; report coverage at each + ECE.

    ECE here = mean | empirical_coverage(α) − (1−α) | across the alpha
    grid. Lower is better. Theoretical floor is 1 / n_test (granularity
    of empirical coverage with discrete trials), so ECE much smaller
    than 1/n_test is not achievable without more test samples.
    """
    per_alpha = []
    for a in alphas:
        cal = ConformalOperatorCalibrator(alpha=a)
        cal.fit(op, encode_fn, calib_pairs)
        per_alpha.append(cal.evaluate(op, encode_fn, test_pairs, pool))

    ece = float(
        np.mean(
            [abs(r["empirical_coverage"] - r["nominal_coverage"]) for r in per_alpha]
        )
    )
    return {
        "per_alpha": per_alpha,
        "ece": ece,
        "alphas": list(alphas),
        "n_test": len(test_pairs),
        "ece_floor": 1.0 / max(len(test_pairs), 1),  # discretization floor
    }
