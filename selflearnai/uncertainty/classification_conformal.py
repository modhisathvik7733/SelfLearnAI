"""Split-conformal prediction for classification (Task 1.5).

Wraps the raw softmax of a classifier (e.g., the Tier-2 IntentClassifier)
in calibrated prediction sets with finite-sample coverage guarantees.

Standard nonconformity score for classification (Sadinle, Lei & Wasserman
2019):

    s(x, y) = 1 − softmax(f(x))[y]

Lower = more confident. After fit on a calibration set with N exchangeable
points, the prediction set for a new x at confidence (1 − α) is:

    Π(x) = { c : 1 − softmax(f(x))[c]  ≤  q_hat }

where q_hat is the ⌈(N+1)(1−α)⌉-th order statistic of calibration scores.
For data exchangeable with the calibration set, P(y_test ∈ Π(x_test))
≥ 1 − α — distribution-free, finite-sample.

In intent routing (Task 1.6), an empty prediction set is the system's
honest "I don't know" signal: refuse explicitly rather than guessing.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class ClassificationConformalCalibrator:
    """Split-conformal calibration on top of a classifier's softmax outputs.

    Coverage guarantee assumes calibration and test data are exchangeable.
    The most common violation in practice is distribution shift; that
    case is detectable by the per-(α) coverage curve diverging from the
    diagonal — `coverage_curve()` makes this visible.

    The classifier itself is treated as a black box; we only need its
    softmax output.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha: float = alpha
        self.q_hat: float | None = None
        self.n_calib: int | None = None
        self.calib_scores: np.ndarray | None = None

    def fit(
        self,
        calib_probs: np.ndarray,
        calib_labels: np.ndarray | list[int],
    ) -> "ClassificationConformalCalibrator":
        """Compute the calibration quantile q_hat.

        Args:
            calib_probs: shape (N, C) softmax probabilities.
            calib_labels: shape (N,) integer class labels (0 ≤ y < C).

        Returns: self.
        """
        probs = np.asarray(calib_probs, dtype=float)
        labels = np.asarray(calib_labels, dtype=int)
        if probs.ndim != 2:
            raise ValueError(f"calib_probs must be (N, C), got {probs.shape}")
        if labels.shape[0] != probs.shape[0]:
            raise ValueError(
                f"len(labels) {labels.shape[0]} != probs.shape[0] {probs.shape[0]}"
            )
        n, c = probs.shape
        if n < 2:
            raise ValueError(f"Need >= 2 calibration samples, got {n}")
        if labels.min() < 0 or labels.max() >= c:
            raise ValueError(
                f"labels out of range [0, {c}); got [{labels.min()}, {labels.max()}]"
            )

        scores = 1.0 - probs[np.arange(n), labels]
        scores_sorted = np.sort(scores)
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        if k > n:
            self.q_hat = float("inf")
        else:
            self.q_hat = float(scores_sorted[k - 1])

        self.n_calib = n
        self.calib_scores = scores
        return self

    def predict_set_indices(self, probs: np.ndarray) -> list[list[int]]:
        """Calibrated prediction sets as lists of class indices."""
        if self.q_hat is None:
            raise RuntimeError("Call fit() before predict_set_indices().")
        probs = np.asarray(probs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        scores = 1.0 - probs                            # (B, C)
        return [
            [int(c) for c in range(scores.shape[1]) if scores[b, c] <= self.q_hat]
            for b in range(scores.shape[0])
        ]

    def evaluate(
        self,
        test_probs: np.ndarray,
        test_labels: np.ndarray | list[int],
    ) -> dict:
        """Empirical coverage + set-size statistics on a disjoint test set."""
        if self.q_hat is None:
            raise RuntimeError("Call fit() before evaluate().")
        probs = np.asarray(test_probs, dtype=float)
        labels = np.asarray(test_labels, dtype=int)
        sets = self.predict_set_indices(probs)

        in_set_flags = [int(labels[i] in sets[i]) for i in range(len(labels))]
        set_sizes = [len(s) for s in sets]
        n = len(labels)

        return {
            "alpha": self.alpha,
            "nominal_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(np.mean(in_set_flags)) if n else float("nan"),
            "n_test": n,
            "n_calib": self.n_calib,
            "in_set_flags": in_set_flags,
            "set_sizes": set_sizes,
            "mean_set_size": float(np.mean(set_sizes)) if n else float("nan"),
            "median_set_size": float(np.median(set_sizes)) if n else float("nan"),
            "max_set_size": int(np.max(set_sizes)) if n else 0,
            "n_empty_sets": int(sum(1 for s in sets if not s)),
            "q_hat": self.q_hat,
        }


def classification_coverage_curve(
    train_probs: np.ndarray,
    train_labels: np.ndarray | list[int],
    test_probs: np.ndarray,
    test_labels: np.ndarray | list[int],
    alphas: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> dict:
    """Re-fit at multiple alpha levels and report empirical coverage at
    each level + ECE = mean | empirical_coverage − (1 − α) | over alphas.

    Use as: split your eval data into a calibration half and a test half;
    pass calibration as `train_probs/labels` and test as `test_probs/labels`.
    """
    per_alpha = []
    for a in alphas:
        cal = ClassificationConformalCalibrator(alpha=a)
        cal.fit(train_probs, train_labels)
        per_alpha.append(cal.evaluate(test_probs, test_labels))
    ece = float(
        np.mean([
            abs(r["empirical_coverage"] - r["nominal_coverage"])
            for r in per_alpha
        ])
    )
    n_test = per_alpha[0]["n_test"] if per_alpha else 0
    return {
        "per_alpha": per_alpha,
        "ece": ece,
        "alphas": list(alphas),
        "n_test": n_test,
        "ece_floor": (1.0 / n_test) if n_test else float("nan"),
    }
