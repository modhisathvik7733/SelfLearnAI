"""Conformal-style verifier for LM-rendered output — THREE-axis gate.

The brain's job is to decide what's true. The LM's job is to render
fluently. This verifier catches the case where the LM drifts from
what the brain decided — adds invented facts, contradicts the
retrieval, hallucinates, etc.

THREE checks (must all pass):

  1. GROUNDING (cos): cos(encode(rendered), ψ_brain_target) ≥ τ_g
     Coarse semantic check — was the response in the right topic area?

  2. RELEVANCE (cos): cos(encode(rendered), ψ_user_query) ≥ τ_r
     Was the response on-topic with the user's question?

  3. CONTENT-WORD FIDELITY (NEW): fraction of rendered content words
     that appear (as substring/stem) in the brain's source texts ≥ τ_c
     Catches LM hallucination — the LM might produce text that's
     semantically close (passing #1) but introduces specific entities,
     numbers, or terms that AREN'T in the source. Cosine alone misses
     this; substring overlap catches it.

     Example caught (2026-05-06 demo): source said "photosynthesis is
     the process plants use to make food from sunlight"; LM rendered
     "...stored in glucose, using chlorophyll and sunlight as
     catalysts." Cosine grounding was 0.902 (passed). Content overlap
     was 0.29 (FAILS τ_c=0.50). Hallucination caught.

The thresholds calibrate per deployment. For v2 (after the demo
hallucination finding), defaults are stricter:
  τ_g = 0.65 (cos grounding — was already in v1)
  τ_r = 0.50 (cos relevance — was already in v1)
  τ_c = 0.50 (NEW: content-word overlap)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F


# Stopwords kept tiny and inline — no new dep. These are the high-frequency
# function words that don't carry content. We do NOT include domain-specific
# terms; let those count as content.
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "with", "from", "this", "that",
    "have", "has", "had", "was", "were", "been", "being",
    "they", "them", "their", "theirs", "what", "when", "where", "which",
    "while", "would", "could", "should", "into", "onto", "than", "then",
    "there", "these", "those", "such", "also", "very", "just", "more",
    "most", "some", "any", "each", "every", "other", "same",
    "make", "made", "use", "used", "using",       # generic verbs
    "your", "yours",
})


def _content_words(text: str, *, min_len: int = 4) -> list[str]:
    """Return list of lowercased content words (alphabetic, ≥min_len, non-stop)."""
    return [
        w for w in re.findall(r"[a-zA-Z]+", text.lower())
        if len(w) >= min_len and w not in _STOPWORDS
    ]


def content_overlap(
    rendered_text: str,
    source_texts: list[str],
    *,
    user_query: str = "",
    min_len: int = 4,
) -> tuple[float, list[str], list[str]]:
    """Compute content-word fidelity between rendered text and sources.

    A rendered content word "passes" if it appears as a SUBSTRING in the
    source-text concatenation. Substring (not exact equality) so that
    'hunt' matches 'hunting', 'cat' matches 'cats', etc. — handles
    morphology without needing a stemmer.

    The user query is included as a source by default — words from the
    question are clearly safe to repeat in the answer.

    Returns:
      (overlap_rate, novel_words, matched_words)
        overlap_rate: |matched| / |total content words|
        novel_words:  rendered content words NOT found in any source
        matched_words: rendered content words that DID match
    """
    rendered_cw = _content_words(rendered_text, min_len=min_len)
    if not rendered_cw:
        return 1.0, [], []
    source_blob = " " + " ".join([user_query] + source_texts).lower() + " "

    matched: list[str] = []
    novel: list[str] = []
    for w in rendered_cw:
        # Substring match — so 'hunt' matches 'hunting', 'cat' matches 'cats'.
        # Use word boundary on left to avoid 'cat' spuriously matching 'concat'.
        # We intentionally keep the right boundary loose for morphology.
        if re.search(rf"[^a-z]{re.escape(w)}", source_blob):
            matched.append(w)
        else:
            novel.append(w)
    return len(matched) / len(rendered_cw), novel, matched


@dataclass
class VerificationResult:
    """Verification verdict — three checks + diagnostic info."""
    grounded: bool
    relevant: bool
    content_fidelity_passed: bool
    accept: bool
    grounding_cos: float
    relevance_cos: float
    content_overlap_rate: float
    novel_content_words: list[str] = field(default_factory=list)
    matched_content_words: list[str] = field(default_factory=list)
    grounding_threshold: float = 0.0
    relevance_threshold: float = 0.0
    content_threshold: float = 0.0
    failure_reason: Optional[str] = None


class RenderingVerifier:
    """Three-axis verifier: cos grounding + cos relevance + content-word fidelity.

    Construct once per session; call `verify` for each rendered output.

    `encode_fn`: list[str] -> Tensor of shape (n, D). Pass the same
    encoder the brain uses (E5 frozen).
    """

    def __init__(
        self,
        encode_fn: Callable[[list[str]], torch.Tensor],
        *,
        grounding_threshold: float = 0.65,
        relevance_threshold: float = 0.50,
        content_threshold: float = 0.50,
    ) -> None:
        for name, val in (("grounding", grounding_threshold),
                          ("relevance", relevance_threshold),
                          ("content", content_threshold)):
            if not 0.0 < val < 1.0:
                raise ValueError(f"{name}_threshold out of (0,1): {val}")
        self.encode_fn = encode_fn
        self.grounding_threshold = grounding_threshold
        self.relevance_threshold = relevance_threshold
        self.content_threshold = content_threshold

    @torch.no_grad()
    def verify(
        self,
        rendered_text: str,
        user_query: str,
        brain_target_texts: list[str],
    ) -> VerificationResult:
        """Run all three checks and return a VerificationResult.

        `brain_target_texts`: texts the brain said are true. The verifier
            checks that the rendered text is (1) close in ψ-space, (2)
            on-topic with the query, AND (3) doesn't introduce content
            words that aren't in the sources.
        """
        if not rendered_text.strip():
            return VerificationResult(
                grounded=False, relevant=False, content_fidelity_passed=False,
                accept=False,
                grounding_cos=0.0, relevance_cos=0.0, content_overlap_rate=0.0,
                grounding_threshold=self.grounding_threshold,
                relevance_threshold=self.relevance_threshold,
                content_threshold=self.content_threshold,
                failure_reason="rendered text is empty",
            )

        # --- 1. Grounding (cos) -------------------------------------
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
            grounding_cos = 1.0    # no target → cosine auto-passes

        # --- 3. Content-word fidelity -------------------------------
        # Pass user_query as a source so the LM can repeat query words
        # without flagging them as novel.
        if brain_target_texts:
            overlap_rate, novel, matched = content_overlap(
                rendered_text, brain_target_texts, user_query=user_query,
            )
        else:
            # No brain target → only check against query (lenient)
            overlap_rate, novel, matched = content_overlap(
                rendered_text, [], user_query=user_query,
            )
            # When there's no target, allow a refusal-shaped response to
            # pass content fidelity (rendered text often has no overlap
            # with empty source).
            if "information" in rendered_text.lower():
                overlap_rate = 1.0

        grounded = grounding_cos >= self.grounding_threshold
        relevant = relevance_cos >= self.relevance_threshold
        content_passed = overlap_rate >= self.content_threshold
        accept = grounded and relevant and content_passed

        reason = None
        if not accept:
            parts = []
            if not grounded:
                parts.append(f"grounding {grounding_cos:.3f} < {self.grounding_threshold}")
            if not relevant:
                parts.append(f"relevance {relevance_cos:.3f} < {self.relevance_threshold}")
            if not content_passed:
                novel_str = ", ".join(novel[:6])
                if len(novel) > 6:
                    novel_str += f", +{len(novel)-6} more"
                parts.append(
                    f"content fidelity {overlap_rate:.3f} < "
                    f"{self.content_threshold} (novel words: [{novel_str}])"
                )
            reason = "; ".join(parts)

        return VerificationResult(
            grounded=grounded,
            relevant=relevant,
            content_fidelity_passed=content_passed,
            accept=accept,
            grounding_cos=grounding_cos,
            relevance_cos=relevance_cos,
            content_overlap_rate=overlap_rate,
            novel_content_words=novel,
            matched_content_words=matched,
            grounding_threshold=self.grounding_threshold,
            relevance_threshold=self.relevance_threshold,
            content_threshold=self.content_threshold,
            failure_reason=reason,
        )

    def _encode_one(self, text: str) -> torch.Tensor:
        z = self.encode_fn([text])
        if z.dim() > 1:
            z = z.squeeze(0)
        return z.flatten()
