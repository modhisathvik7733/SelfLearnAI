"""Compositional conformal prediction for operator chains.

Builds on `selflearnai.uncertainty.conformal.ConformalOperatorCalibrator`
to give chain-level prediction sets with formal coverage guarantees.

Two approaches are exposed; both are sound, with different
data-efficiency / tightness tradeoffs:

1. **End-to-end chain calibration** (`compose_operators` + reused
   ConformalOperatorCalibrator). Treat the composed function
   `op_K ∘ ... ∘ op_1` as a single operator and calibrate it on
   chain-level (input → final-target) pairs. Tighter sets when
   chain-level calibration data is available.

2. **Bonferroni composition** (`BonferroniChainCalibrator`). Calibrate
   each operator independently at level (1 − α/K), then propagate sets
   along the chain. Coverage ≥ (1 − α) by union bound. Conservative;
   useful when chain-level data is scarce but per-operator calibration
   data is plentiful. Reference: Compositional Conformal Prediction for
   Neuro-symbolic Programs, arXiv:2405.15912 (the broader paper covers
   tighter alternatives to Bonferroni; we start with the simplest valid
   form here).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .conformal import ConformalOperatorCalibrator


def compose_operators(ops: Sequence[Callable[[torch.Tensor], torch.Tensor]]):
    """Compose operators into a single callable applied left-to-right.

    Returns a callable `chain(z)` such that
        chain(z) = op_K(op_{K-1}( ... op_1(z) ... ))
    suitable for passing to `ConformalOperatorCalibrator.fit` for
    end-to-end chain calibration.
    """
    if not ops:
        raise ValueError("Need at least one operator to compose.")

    def chain(z: torch.Tensor) -> torch.Tensor:
        out = z
        for op in ops:
            out = op(out)
        return out

    return chain


class BonferroniChainCalibrator:
    """Chain prediction sets via Bonferroni union over per-operator calibrators.

    Given K operators with associated per-operator
    `ConformalOperatorCalibrator`s, each calibrated at level (1 − α / K),
    a chain prediction set is built by propagating intermediate
    candidate sets along the chain. The final set has coverage
    ≥ (1 − α) by union bound (a target only fails to be covered if at
    least one of the K conformal sets misses, each with probability
    ≤ α/K).

    Sound, valid finite-sample, but conservative — set sizes can be
    larger than tighter bounds (e.g. CCP for Neuro-symbolic Programs).
    Use as the default compositional method until tighter formulas are
    integrated.

    Notes:
      - This class assumes intermediate candidates are drawn from
        per-stage candidate pools (a `pools[i]` list of strings). For
        a chain of K operators, you supply K pools (one per stage).
      - For practical efficiency, set propagation is materialized as a
        union over stage-wise calibrated sets. With reasonable per-stage
        calibration tightness this stays bounded; pathological
        widenings can blow up the chain set, in which case end-to-end
        chain calibration is preferable.
    """

    def __init__(self, calibrators: Sequence[ConformalOperatorCalibrator]):
        if not calibrators:
            raise ValueError("Need at least one operator calibrator")
        for i, c in enumerate(calibrators):
            if c.q_hat is None:
                raise ValueError(
                    f"Calibrator {i} not fit yet. Call fit() on each before use."
                )
        self.calibrators: list[ConformalOperatorCalibrator] = list(calibrators)
        self.K: int = len(calibrators)

    @torch.no_grad()
    def predict_chain_set(
        self,
        ops: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        encode_fn: Callable[[list[str]], torch.Tensor],
        src: str,
        pools: Sequence[list[str]],
    ) -> list[str]:
        """Compute the chain prediction set for source `src`.

        Args:
            ops: K operators in application order.
            encode_fn: list[str] -> Tensor.
            src: source word.
            pools: K candidate pools (one per stage). pools[i] is the
                candidate set used at stage i; pools[-1] is the final
                target pool.

        Returns: chain prediction set as list[str] (subset of pools[-1]).
        """
        if len(ops) != self.K or len(pools) != self.K:
            raise ValueError(
                f"Number of ops ({len(ops)}), pools ({len(pools)}), and "
                f"calibrators ({self.K}) must match."
            )

        # Stage 0: the only "input candidate" is the source itself.
        current_candidates: list[str] = [src]

        for stage in range(self.K):
            op = ops[stage]
            cal = self.calibrators[stage]
            stage_pool = pools[stage]
            next_candidates_set: set[str] = set()

            # Each candidate from the previous stage seeds a stage-set.
            for cand in current_candidates:
                z_cand = encode_fn([cand])
                z_pred = op(z_cand)
                z_pool = encode_fn(stage_pool)
                cos = F.cosine_similarity(z_pred, z_pool, dim=-1)
                scores = (-cos).detach().cpu().numpy().astype(float).ravel()
                for w, s in zip(stage_pool, scores):
                    if s <= cal.q_hat:
                        next_candidates_set.add(w)

            current_candidates = list(next_candidates_set)
            # If the set ever empties, the chain set is empty.
            if not current_candidates:
                return []

        return current_candidates

    @torch.no_grad()
    def evaluate(
        self,
        ops: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        encode_fn: Callable[[list[str]], torch.Tensor],
        test_pairs: list[tuple[str, str]],
        pools: Sequence[list[str]],
    ) -> dict:
        """Empirical chain coverage + set-size statistics on test pairs.

        Each test pair is (chain_source, chain_target). Coverage is the
        fraction of test pairs whose target is in the chain set.
        """
        in_set_flags: list[int] = []
        set_sizes: list[int] = []
        for src, tgt in test_pairs:
            chain_set = self.predict_chain_set(ops, encode_fn, src, pools)
            in_set_flags.append(int(tgt in chain_set))
            set_sizes.append(len(chain_set))

        n_test = len(test_pairs)
        # Bonferroni nominal coverage: (1 - sum α_i). All same alpha here.
        # Caller can read each calibrator's alpha to derive nominal.
        bonferroni_alpha = sum(c.alpha for c in self.calibrators)
        nominal = max(0.0, 1.0 - bonferroni_alpha)

        return {
            "method": "bonferroni",
            "K": self.K,
            "per_op_alphas": [c.alpha for c in self.calibrators],
            "bonferroni_alpha": bonferroni_alpha,
            "nominal_coverage": nominal,
            "empirical_coverage": float(np.mean(in_set_flags)) if n_test else float("nan"),
            "n_test": n_test,
            "in_set_flags": in_set_flags,
            "set_sizes": set_sizes,
            "mean_set_size": float(np.mean(set_sizes)) if n_test else float("nan"),
            "median_set_size": float(np.median(set_sizes)) if n_test else float("nan"),
            "max_set_size": int(np.max(set_sizes)) if n_test else 0,
        }
