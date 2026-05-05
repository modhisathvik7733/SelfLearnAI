"""Intent module — natural-language question → structured Intent.

Three-tier hybrid (per plan §1, Gap 1):

  Tier 1 — Typed grammar (this file: `grammar.py`).
    Hand-authored regex patterns over the known concept library.
    Deterministic, fast, no encoder, no learned model. Either matches
    a known shape exactly OR returns None (refusal). Used as the first
    line of defense: if Tier 1 matches confidently, no further work.

  Tier 2 — Ψ-space classifier (Task 1.4, not yet built).
    Small MLP over the encoder embedding of the question, classifying
    into known concepts with calibrated probability. Catches paraphrases
    Tier 1 misses.

  Tier 3 — Program induction (Task 1.5, not yet built).
    Bounded beam search over operator chains that produce a state
    matching the question's encoded goal. Catches anything Tier 1 and
    Tier 2 can't decompose.

The router (Task 1.6) combines all three with conformal-gated
confidence — falling through tiers from cheapest to most expensive
and refusing explicitly when confidence stays below threshold.

This `__init__` exposes only the Tier-1 `Intent` and `TypedGrammarParser`
for now; later tasks add Tier-2/Tier-3 exports without changing the
public API.
"""
from .grammar import (
    Intent,
    TypedGrammarParser,
    DEFAULT_PATTERNS,
)
from .classifier import (
    IntentClassifier,
    IntentClassMapping,
    IntentPrediction,
)

__all__ = [
    # Tier 1
    "Intent",
    "TypedGrammarParser",
    "DEFAULT_PATTERNS",
    # Tier 2
    "IntentClassifier",
    "IntentClassMapping",
    "IntentPrediction",
]
