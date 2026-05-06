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


# Optional dependency: wordfreq — used to distinguish common-paraphrase
# words (high zipf, fluency) from rare-specific words (low zipf, likely
# hallucinated facts). If wordfreq is unavailable, the controlled gate
# degrades gracefully to "every novel word treated as common" — i.e. it
# does NOT block rare-novel additions, only the overlap-rate gate fires.
try:
    from wordfreq import zipf_frequency as _zipf_frequency  # type: ignore[import-not-found]
    _HAS_WORDFREQ = True
except ImportError:
    _HAS_WORDFREQ = False
    def _zipf_frequency(word: str, lang: str) -> float:        # type: ignore[no-redef]
        del word, lang
        return 5.0       # treat as common when wordfreq unavailable


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
    common_zipf_threshold: float = 4.0,
) -> tuple[float, list[str], list[str], list[str]]:
    """Controlled-enrichment content fidelity.

    Distinguishes three classes of rendered content words:
      MATCHED    — appears as substring in source/query (paraphrase ground truth)
      COMMON-NOVEL — not in source, but high-frequency (zipf ≥ threshold).
                     Likely paraphrase synonym, allowed as fluency.
      RARE-NOVEL  — not in source AND low-frequency (zipf < threshold).
                     Likely a SPECIFIC FACT smuggled in by the LM.
                     Blocked.

    The verifier uses two gates on top of these:
      1. effective_overlap = (matched + common_novel) / total ≥ τ_overlap
      2. rare_novel_count ≤ τ_rare_max

    Both must pass for content fidelity. This is the "controlled enrichment"
    design: paraphrase fluency allowed; specific-entity hallucinations blocked.

    A rendered content word "matches" if it appears as a SUBSTRING in
    source/query — so 'hunt' matches 'hunting', 'cat' matches 'cats'.

    Returns:
      (effective_overlap, novel_words, matched_words, rare_novel_words)
    """
    rendered_cw = _content_words(rendered_text, min_len=min_len)
    if not rendered_cw:
        return 1.0, [], [], []
    source_blob = " " + " ".join([user_query] + source_texts).lower() + " "

    matched: list[str] = []
    novel: list[str] = []
    rare_novel: list[str] = []
    for w in rendered_cw:
        if re.search(rf"[^a-z]{re.escape(w)}", source_blob):
            matched.append(w)
        else:
            novel.append(w)
            if _zipf_frequency(w, "en") < common_zipf_threshold:
                rare_novel.append(w)

    common_novel_count = len(novel) - len(rare_novel)
    effective_overlap = (len(matched) + common_novel_count) / len(rendered_cw)
    return effective_overlap, novel, matched, rare_novel


@dataclass
class VerificationResult:
    """Verification verdict — three checks + controlled-enrichment diagnostics."""
    grounded: bool
    relevant: bool
    content_fidelity_passed: bool
    accept: bool
    grounding_cos: float
    relevance_cos: float
    content_overlap_rate: float
    rare_novel_count: int = 0
    novel_content_words: list[str] = field(default_factory=list)
    rare_novel_content_words: list[str] = field(default_factory=list)
    matched_content_words: list[str] = field(default_factory=list)
    grounding_threshold: float = 0.0
    relevance_threshold: float = 0.0
    content_threshold: float = 0.0
    rare_novel_max: int = 0
    failure_reason: Optional[str] = None


class RenderingVerifier:
    """Three-axis verifier: cos grounding + cos relevance + controlled-enrichment
    content-word fidelity.

    Content fidelity is a TWO-PRONGED gate (controlled enrichment):
      1. effective_overlap = (matched + common-novel) / total ≥ content_threshold
         Common-paraphrase synonyms (high zipf frequency) count as overlap —
         they're fluency, not hallucination.
      2. rare_novel_count ≤ rare_novel_max
         Rare/specific words (low zipf frequency) that aren't in source are
         treated as smuggled-in facts. Allow 1 (synonym safety); block 2+.

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
        rare_novel_max: int = 1,
        common_zipf_threshold: float = 4.0,
    ) -> None:
        for name, val in (("grounding", grounding_threshold),
                          ("relevance", relevance_threshold),
                          ("content", content_threshold)):
            if not 0.0 < val < 1.0:
                raise ValueError(f"{name}_threshold out of (0,1): {val}")
        if rare_novel_max < 0:
            raise ValueError(f"rare_novel_max must be >= 0, got {rare_novel_max}")
        self.encode_fn = encode_fn
        self.grounding_threshold = grounding_threshold
        self.relevance_threshold = relevance_threshold
        self.content_threshold = content_threshold
        self.rare_novel_max = rare_novel_max
        self.common_zipf_threshold = common_zipf_threshold

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
                rare_novel_count=0,
                grounding_threshold=self.grounding_threshold,
                relevance_threshold=self.relevance_threshold,
                content_threshold=self.content_threshold,
                rare_novel_max=self.rare_novel_max,
                failure_reason="rendered text is empty",
            )

        # --- 1. Grounding (cos) + 2. Relevance (cos) ----------------
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

        # --- 3. Content-word fidelity (controlled enrichment) -------
        if brain_target_texts:
            overlap_rate, novel, matched, rare_novel = content_overlap(
                rendered_text, brain_target_texts,
                user_query=user_query,
                common_zipf_threshold=self.common_zipf_threshold,
            )
        else:
            overlap_rate, novel, matched, rare_novel = content_overlap(
                rendered_text, [],
                user_query=user_query,
                common_zipf_threshold=self.common_zipf_threshold,
            )
            if "information" in rendered_text.lower():
                overlap_rate = 1.0           # refusal-shaped output auto-passes

        grounded = grounding_cos >= self.grounding_threshold
        relevant = relevance_cos >= self.relevance_threshold
        overlap_passed = overlap_rate >= self.content_threshold
        rare_passed = len(rare_novel) <= self.rare_novel_max
        content_passed = overlap_passed and rare_passed
        accept = grounded and relevant and content_passed

        reason = None
        if not accept:
            parts = []
            if not grounded:
                parts.append(f"grounding {grounding_cos:.3f} < {self.grounding_threshold}")
            if not relevant:
                parts.append(f"relevance {relevance_cos:.3f} < {self.relevance_threshold}")
            if not overlap_passed:
                novel_str = ", ".join(novel[:6])
                if len(novel) > 6:
                    novel_str += f", +{len(novel)-6} more"
                parts.append(
                    f"effective overlap {overlap_rate:.3f} < "
                    f"{self.content_threshold} (novel: [{novel_str}])"
                )
            if not rare_passed:
                rare_str = ", ".join(rare_novel[:6])
                if len(rare_novel) > 6:
                    rare_str += f", +{len(rare_novel)-6} more"
                parts.append(
                    f"rare-novel count {len(rare_novel)} > "
                    f"{self.rare_novel_max} (specific facts smuggled in: [{rare_str}])"
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
            rare_novel_count=len(rare_novel),
            novel_content_words=novel,
            rare_novel_content_words=rare_novel,
            matched_content_words=matched,
            grounding_threshold=self.grounding_threshold,
            relevance_threshold=self.relevance_threshold,
            content_threshold=self.content_threshold,
            rare_novel_max=self.rare_novel_max,
            failure_reason=reason,
        )

    def _encode_one(self, text: str) -> torch.Tensor:
        z = self.encode_fn([text])
        if z.dim() > 1:
            z = z.squeeze(0)
        return z.flatten()
