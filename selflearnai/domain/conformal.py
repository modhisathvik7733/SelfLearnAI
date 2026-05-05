"""Per-domain decoder conformal calibration (Stage 3 / sub-task 3.6).

Mirror of Stage 0.5's ConformalOperatorCalibrator (selflearnai/uncertainty/
conformal.py) at the DECODER level instead of the operator level.

Stage 0.5 score (operator):
    s(src, tgt) = -cos(op(z_src), z_tgt)
Stage 3.6 score (per-domain decoder):
    s(input_sent, target_sent) = -cos(encode(decode(h_input)), z_target)

The decoder is treated as a black box that takes encoder activations and
emits text; we re-encode the emitted text and compare its pooled ψ to the
target sentence's pooled ψ. Lower score = higher fidelity. Quantile q_hat
at confidence (1-α) defines an admit/refuse threshold with finite-sample
coverage guarantee per Vovk-Shafer-Gammerman (2005); standard split-
conformal form per Angelopoulos & Bates (arXiv:2107.07511).

Per plan §19.17 sub-task 3.6 acceptance: ECE < 0.07 on each registered
domain's held-out, where ECE = mean |empirical_coverage(α) - (1-α)|
averaged over an α grid.

Interpretation: "the decoder's output for this input is within the
calibrated fidelity bar" → admit (use the decode); else refuse (escalate
to user clarification or out-of-domain message). Cleaner than
hand-tuned cosine thresholds, matches plan §9.4's principled refusal
semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

@dataclass
class DomainConformalCalibrator:
    """Split-conformal calibration on a per-domain decoder's fidelity.

    Calibration data MUST be disjoint from any data the decoder was
    trained on AND from test data — otherwise the coverage guarantee
    breaks.

    Usage:
      cal = DomainConformalCalibrator(alpha=0.10)
      cal.fit(decoder, encode_pool_fn, decode_fn,
              calib_input_sents, calib_target_sents)
      report = cal.evaluate(...)
    """
    alpha: float = 0.10
    q_hat: float | None = None
    n_calib: int | None = None
    calib_scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    @torch.no_grad()
    def fit(
        self,
        decode_fn: Callable[[list[str]], list[str]],
        encode_pool_fn: Callable[[list[str]], torch.Tensor],
        calib_input_sents: list[str],
        calib_target_sents: list[str],
    ) -> "DomainConformalCalibrator":
        """Compute the calibration quantile q_hat from held-out (input, target) pairs.

        Args:
            decode_fn: list[input_sent] -> list[generated_text]. The decoder's
                       inference path; encapsulates encoding + running the decoder.
            encode_pool_fn: list[str] -> Tensor of pooled ψ ∈ R^{n × D}. Typically a
                            closure around the foundation encoder.
            calib_input_sents: input sentences (held-out from training).
            calib_target_sents: target sentences (parallel; what the decoder SHOULD produce).
        """
        n = len(calib_input_sents)
        if n != len(calib_target_sents):
            raise ValueError("input and target lists must be parallel")
        if n < 2:
            raise ValueError(f"Need >= 2 calibration pairs, got {n}.")

        gen_texts = decode_fn(calib_input_sents)
        z_gen = encode_pool_fn(gen_texts)
        z_tgt = encode_pool_fn(calib_target_sents)
        cos = F.cosine_similarity(z_gen, z_tgt, dim=-1)
        scores = (-cos).detach().cpu().numpy().astype(float).ravel()

        # Conformal quantile.
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        if k > n:
            self.q_hat = float("inf")
        else:
            self.q_hat = float(np.sort(scores)[k - 1])
        self.n_calib = n
        self.calib_scores = scores
        return self

    @torch.no_grad()
    def admit(
        self,
        decode_fn: Callable[[list[str]], list[str]],
        encode_pool_fn: Callable[[list[str]], torch.Tensor],
        input_sents: list[str],
        target_sents: list[str],
    ) -> list[bool]:
        """Per-input admission decisions at the calibrated bar."""
        if self.q_hat is None:
            raise RuntimeError("Call fit() before admit().")
        gen_texts = decode_fn(input_sents)
        z_gen = encode_pool_fn(gen_texts)
        z_tgt = encode_pool_fn(target_sents)
        cos = F.cosine_similarity(z_gen, z_tgt, dim=-1)
        scores = (-cos).detach().cpu().numpy().astype(float).ravel()
        return [bool(s <= self.q_hat) for s in scores]

    @torch.no_grad()
    def evaluate(
        self,
        decode_fn: Callable[[list[str]], list[str]],
        encode_pool_fn: Callable[[list[str]], torch.Tensor],
        test_input_sents: list[str],
        test_target_sents: list[str],
    ) -> dict[str, Any]:
        """Empirical coverage on a disjoint test set."""
        if self.q_hat is None:
            raise RuntimeError("Call fit() before evaluate().")
        admit_flags = self.admit(
            decode_fn, encode_pool_fn, test_input_sents, test_target_sents,
        )
        # Re-compute scores for reporting (admit() already did, but evaluate()
        # is an entry point so do it freshly).
        gen_texts = decode_fn(test_input_sents)
        z_gen = encode_pool_fn(gen_texts)
        z_tgt = encode_pool_fn(test_target_sents)
        cos = F.cosine_similarity(z_gen, z_tgt, dim=-1)
        scores = (-cos).detach().cpu().numpy().astype(float).ravel()
        n_test = len(test_input_sents)
        return {
            "alpha": self.alpha,
            "nominal_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(np.mean(admit_flags)) if n_test else float("nan"),
            "n_test": n_test,
            "n_calib": self.n_calib,
            "admit_flags": admit_flags,
            "test_scores_min": float(np.min(scores)),
            "test_scores_median": float(np.median(scores)),
            "test_scores_max": float(np.max(scores)),
            "q_hat": self.q_hat,
        }


# ---------------------------------------------------------------------------
# Coverage curve + ECE
# ---------------------------------------------------------------------------

def domain_coverage_curve(
    decode_fn: Callable[[list[str]], list[str]],
    encode_pool_fn: Callable[[list[str]], torch.Tensor],
    calib_input_sents: list[str],
    calib_target_sents: list[str],
    test_input_sents: list[str],
    test_target_sents: list[str],
    alphas: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> dict[str, Any]:
    """Re-fit at multiple alpha levels; report coverage at each + ECE.

    ECE = mean |empirical_coverage(α) - (1-α)| over the α grid.
    Theoretical floor is 1 / n_test (granularity of empirical coverage).
    """
    per_alpha = []
    alpha_list = list(alphas)
    for a in alpha_list:
        cal = DomainConformalCalibrator(alpha=a)
        cal.fit(decode_fn, encode_pool_fn, calib_input_sents, calib_target_sents)
        per_alpha.append(cal.evaluate(
            decode_fn, encode_pool_fn, test_input_sents, test_target_sents,
        ))
    ece = float(np.mean([
        abs(r["empirical_coverage"] - r["nominal_coverage"]) for r in per_alpha
    ]))
    return {
        "alphas": alpha_list,
        "per_alpha": per_alpha,
        "ece": ece,
        "n_calib": len(calib_input_sents),
        "n_test": len(test_input_sents),
        "ece_floor": 1.0 / max(len(test_input_sents), 1),
    }
