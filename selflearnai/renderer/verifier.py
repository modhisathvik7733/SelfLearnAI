"""Conformal-style verifier for LM-rendered output.

The brain's job is to decide what's true. The LM's job is to render
fluently. This verifier catches the case where the LM drifts from
what the brain decided — adds invented facts, contradicts the
retrieval, hallucinates, etc.

The verifier asks two questions about every LM output:

  1. GROUNDING: does the rendered text encode close to what the brain
     said was true (retrieved facts, operator output, raw target)?
     Score = cos(encode(rendered), ψ_brain_target).

  2. RELEVANCE: does the rendered text encode close to the user query
     (i.e. is it on-topic)?
     Score = cos(encode(rendered), ψ_user_query).

Both must clear thresholds. If either fails, the output is REJECTED
and the pipeline either regenerates or refuses.

The thresholds are NOT magic numbers — they should be calibrated per
deployment via a small held-out (query, expected_output) set,
following the pattern from `selflearnai/uncertainty/conformal.py`.
For v1 we use defaults derived from §20.0b's relative-gate calibration
lesson: thresholds are intentionally on the looser side, since LM
outputs are paraphrases of brain output and may be cos ≈ 0.7-0.85
rather than near-perfect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class VerificationResult:
    """One verification verdict — three booleans + the cosines."""
    grounded: bool
    relevant: bool
    accept: bool
    grounding_cos: float
    relevance_cos: float
    grounding_threshold: float
    relevance_threshold: float
    failure_reason: Optional[str] = None


class RenderingVerifier:
    """Two-axis verifier: grounding (vs brain target) + relevance (vs query).

    Construct once per session; call `verify` for each rendered output.

    `encode_fn`: list[str] -> Tensor of shape (n, D). Pass the same
    encoder the brain uses (E5 frozen).
    """

    def __init__(
        self,
        encode_fn: Callable[[list[str]], torch.Tensor],
        *,
        grounding_threshold: float = 0.70,
        relevance_threshold: float = 0.55,
    ) -> None:
        if not 0.0 < grounding_threshold < 1.0:
            raise ValueError(f"grounding_threshold out of (0,1): {grounding_threshold}")
        if not 0.0 < relevance_threshold < 1.0:
            raise ValueError(f"relevance_threshold out of (0,1): {relevance_threshold}")
        self.encode_fn = encode_fn
        self.grounding_threshold = grounding_threshold
        self.relevance_threshold = relevance_threshold

    @torch.no_grad()
    def verify(
        self,
        rendered_text: str,
        user_query: str,
        brain_target_texts: list[str],
    ) -> VerificationResult:
        """Run both checks and return a VerificationResult.

        `brain_target_texts`: the texts the brain said are true (retrieved
            facts, operator output, raw target). The verifier checks that
            the rendered text is close in ψ-space to these. Pass an empty
            list if there's no brain target (then grounding auto-passes).
        """
        if not rendered_text.strip():
            return VerificationResult(
                grounded=False, relevant=False, accept=False,
                grounding_cos=0.0, relevance_cos=0.0,
                grounding_threshold=self.grounding_threshold,
                relevance_threshold=self.relevance_threshold,
                failure_reason="rendered text is empty",
            )

        z_rendered = self._encode_one(rendered_text)
        z_query = self._encode_one(user_query)
        relevance_cos = float(F.cosine_similarity(
            z_rendered.unsqueeze(0), z_query.unsqueeze(0), dim=-1
        ).item())

        if brain_target_texts:
            z_targets = self.encode_fn(brain_target_texts)
            cos_per_target = F.cosine_similarity(
                z_rendered.unsqueeze(0), z_targets, dim=-1
            )
            grounding_cos = float(cos_per_target.max().item())
        else:
            # No brain target → grounding auto-passes (e.g. raw target rendering).
            grounding_cos = 1.0

        grounded = grounding_cos >= self.grounding_threshold
        relevant = relevance_cos >= self.relevance_threshold
        accept = grounded and relevant
        reason = None
        if not accept:
            parts = []
            if not grounded:
                parts.append(
                    f"grounding {grounding_cos:.3f} < {self.grounding_threshold}"
                )
            if not relevant:
                parts.append(
                    f"relevance {relevance_cos:.3f} < {self.relevance_threshold}"
                )
            reason = "; ".join(parts)

        return VerificationResult(
            grounded=grounded,
            relevant=relevant,
            accept=accept,
            grounding_cos=grounding_cos,
            relevance_cos=relevance_cos,
            grounding_threshold=self.grounding_threshold,
            relevance_threshold=self.relevance_threshold,
            failure_reason=reason,
        )

    def _encode_one(self, text: str) -> torch.Tensor:
        z = self.encode_fn([text])
        if z.dim() > 1:
            z = z.squeeze(0)
        return z.flatten()
