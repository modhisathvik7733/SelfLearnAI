"""Tier-1 intent parser — typed regex grammar over the known concept library.

Maps a natural-language question to a structured `Intent` if the question
matches one of the hand-authored patterns; returns `None` otherwise. The
None case is a deliberate REFUSAL — Tier 2 (learned classifier) and
Tier 3 (program induction) are responsible for the long tail of
phrasings; Tier 1 only commits when it is sure.

Design properties:
  - Deterministic: same question → same intent (or refusal). No
    randomness, no learned model.
  - Cheap: pure regex matching; microseconds per call.
  - Composable: callers can extend `DEFAULT_PATTERNS` or pass a custom
    pattern list to `TypedGrammarParser` to add concepts.
  - Honest: pattern_id field in the Intent records exactly which regex
    matched, so debugging / analytics is easy.

Concepts currently supported (matches `data/` library + `data/few_shot/`):
  plural, past_tense, comparative, superlative, opposite, agentive, young.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Intent:
    """Structured representation of a parsed user intent.

    `concept` matches a key in the concept library (e.g. "plural").
    `source` is the word/phrase the operator should be applied to.
    `pattern_id` records which Tier-1 pattern matched, for tracing.
    `confidence` is 1.0 for Tier-1 deterministic matches; later tiers
    will populate this with calibrated probabilities.
    `modifiers` is a free-form bag for future enrichments (e.g. tense,
    plurality, definiteness); empty in Tier 1.
    """
    concept: str
    source: str
    raw_question: str
    pattern_id: str
    confidence: float = 1.0
    modifiers: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Pattern:
    """One concept-pattern row.

    `regex` MUST contain a named group "source" capturing the operator's
    input word; nothing else is expected from the match.
    """
    concept: str
    pattern_id: str
    regex: re.Pattern


# ----- Pattern table --------------------------------------------------------
# Patterns are tried in order; the FIRST match wins. Order matters for
# disambiguation when patterns could overlap. More-specific patterns
# (longer surface, more keywords) come before more-general ones.
#
# Each pattern uses `re.match` semantics — anchored at start, free at end —
# so trailing punctuation / whitespace is allowed.

_OPT_LEAD = r"(?:(?:what(?:'s| is) the |(?:tell me )?the |give me the |find the )?)"
_OPT_TRAIL = r"\s*[?.!]?\s*$"

DEFAULT_PATTERNS: tuple[_Pattern, ...] = (
    # ------- plural -------
    _Pattern(
        "plural", "plural.canonical",
        re.compile(rf"^{_OPT_LEAD}plural (?:form )?of (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),
    _Pattern(
        "plural", "plural.makeit",
        re.compile(rf"^make (?P<source>\w+) plural{_OPT_TRAIL}", re.I),
    ),
    _Pattern(
        "plural", "plural.simple",
        re.compile(rf"^plurals? for (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),

    # ------- past_tense -------
    _Pattern(
        "past_tense", "past.canonical",
        re.compile(rf"^{_OPT_LEAD}past (?:tense )?(?:form )?of (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),
    _Pattern(
        "past_tense", "past.in_form",
        re.compile(rf"^(?P<source>\w+) in (?:the )?past(?: tense)?{_OPT_TRAIL}", re.I),
    ),

    # ------- superlative (BEFORE comparative because "most X" is more specific
    # than "more X" but they would never collide; ordered by concept clarity) -------
    _Pattern(
        "superlative", "sup.canonical",
        re.compile(rf"^{_OPT_LEAD}superlative (?:form )?of (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),
    _Pattern(
        "superlative", "sup.most",
        re.compile(rf"^most (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),

    # ------- comparative -------
    _Pattern(
        "comparative", "comp.canonical",
        re.compile(rf"^{_OPT_LEAD}comparative (?:form )?of (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),
    _Pattern(
        "comparative", "comp.more",
        re.compile(rf"^more (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),

    # ------- opposite / antonym -------
    _Pattern(
        "opposite", "opp.canonical",
        re.compile(rf"^{_OPT_LEAD}(?:opposite|antonym) of (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),

    # ------- agentive ("one who Xs", "agent of X") -------
    _Pattern(
        "agentive", "agent.canonical",
        re.compile(rf"^{_OPT_LEAD}agent(?:ive)? (?:form )?of (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),
    _Pattern(
        "agentive", "agent.one_who",
        re.compile(rf"^(?:a |an |one |someone |some(?:one|body) )?(?:person |someone )?who (?P<source>\w+)s{_OPT_TRAIL}", re.I),
    ),

    # ------- young animal -------
    _Pattern(
        "young", "young.canonical",
        re.compile(rf"^(?:baby|young) (?P<source>\w+){_OPT_TRAIL}", re.I),
    ),
    _Pattern(
        "young", "young.what_is",
        re.compile(rf"^what is a (?:young |baby )(?P<source>\w+)(?: called)?{_OPT_TRAIL}", re.I),
    ),
)


class TypedGrammarParser:
    """Tier-1 deterministic intent parser.

    Usage::

        parser = TypedGrammarParser()
        intent = parser("what's the plural of cat")
        if intent is not None:
            assert intent.concept == "plural"
            assert intent.source == "cat"
        else:
            # Tier 1 doesn't match; caller falls through to Tier 2/3.
            pass

    Custom pattern sets can be passed for testing or for plug-in concept
    extensions::

        parser = TypedGrammarParser(patterns=DEFAULT_PATTERNS + my_patterns)
    """

    def __init__(self, patterns: tuple[_Pattern, ...] | list[_Pattern] | None = None):
        self._patterns: tuple[_Pattern, ...] = (
            tuple(patterns) if patterns is not None else DEFAULT_PATTERNS
        )

    def __call__(self, question: str) -> Optional[Intent]:
        return self.parse(question)

    def parse(self, question: str) -> Optional[Intent]:
        q = question.strip()
        if not q:
            return None
        for pat in self._patterns:
            m = pat.regex.match(q)
            if m:
                source = m.group("source").strip()
                if not source:
                    continue
                return Intent(
                    concept=pat.concept,
                    source=source,
                    raw_question=question,
                    pattern_id=pat.pattern_id,
                )
        return None

    @property
    def supported_concepts(self) -> list[str]:
        return sorted({p.concept for p in self._patterns})

    @property
    def num_patterns(self) -> int:
        return len(self._patterns)
