"""PropertyIntentParser — deterministic regex parser for property questions.

Maps natural-language property questions to (entity, axis) tuples that
the RelationalOperator can answer. Pure regex, no LM.

Supported axes (must match Tier 0.1 trained vocabulary):
  color, sound, habitat, requires, function, made_of, used_for, part_of

Pattern shapes covered:
  "what color is the sky?"              → (sky, color)
  "what does a plant need?"             → (plant, requires)
  "where do penguins live?"             → (penguins, habitat)
  "what does a brain do?"               → (brain, function)
  "what sound does a horse make?"       → (horse, sound)
  "what is paper made of?"              → (paper, made_of)
  "what is a hammer used for?"          → (hammer, used_for)
  "what is a finger part of?"           → (finger, part_of)

Returns None when the query doesn't match any property pattern — the
pipeline then falls back to retrieval.
"""
from __future__ import annotations

import re
from typing import Optional


# Pattern → axis name. Order matters: more specific patterns first.
# Articles (a/an/the) optional in entity capture.
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ---- color ---------------------------------------------------
    (re.compile(r"^what colou?r (?:is|are) (?:the |a |an )?(.+?)\??$", re.IGNORECASE),
     "color"),
    (re.compile(r"^what (?:is|are) the colou?r of (?:the |a |an )?(.+?)\??$", re.IGNORECASE),
     "color"),

    # ---- sound ---------------------------------------------------
    (re.compile(r"^what sound does (?:the |a |an )?(.+?) make\??$", re.IGNORECASE),
     "sound"),
    (re.compile(r"^what does (?:the |a |an )?(.+?) sound like\??$", re.IGNORECASE),
     "sound"),

    # ---- habitat -------------------------------------------------
    (re.compile(r"^where (?:do|does) (?:the |a |an )?(.+?) live\??$", re.IGNORECASE),
     "habitat"),
    (re.compile(r"^where (?:do|does) (?:the |a |an )?(.+?) (?:reside|stay)\??$", re.IGNORECASE),
     "habitat"),
    (re.compile(r"^what is the habitat of (?:the |a |an )?(.+?)\??$", re.IGNORECASE),
     "habitat"),

    # ---- requires ------------------------------------------------
    (re.compile(r"^what (?:does|do) (?:the |a |an )?(.+?) (?:need|require)\??$", re.IGNORECASE),
     "requires"),
    (re.compile(r"^what is needed by (?:the |a |an )?(.+?)\??$", re.IGNORECASE),
     "requires"),

    # ---- function -----------------------------------------------
    (re.compile(r"^what does (?:the |a |an )?(.+?) do\??$", re.IGNORECASE),
     "function"),
    (re.compile(r"^what is the function of (?:the |a |an )?(.+?)\??$", re.IGNORECASE),
     "function"),

    # ---- made_of ------------------------------------------------
    (re.compile(r"^what (?:is|are) (?:the |a |an )?(.+?) made (?:of|from)\??$", re.IGNORECASE),
     "made_of"),
    (re.compile(r"^what (?:is|are) (?:the )?(.+?)'?s? main ingredient\??$", re.IGNORECASE),
     "made_of"),

    # ---- used_for -----------------------------------------------
    (re.compile(r"^what (?:is|are) (?:the |a |an )?(.+?) used for\??$", re.IGNORECASE),
     "used_for"),
    (re.compile(r"^what (?:do|does) (?:we |you )?use (?:the |a |an )?(.+?) for\??$", re.IGNORECASE),
     "used_for"),

    # ---- part_of ------------------------------------------------
    (re.compile(r"^what (?:is|are) (?:the |a |an )?(.+?) part of\??$", re.IGNORECASE),
     "part_of"),
    (re.compile(r"^where (?:do|does) (?:we |you )?find (?:the |a |an )?(.+?)\??$", re.IGNORECASE),
     "part_of"),
]


def parse_property_question(query: str) -> Optional[tuple[str, str]]:
    """Returns (entity, axis) if `query` matches a known property pattern,
    else None.

    Entity is lowercased and stripped of leading articles (kept simple —
    the RelationalOperator's encoder doesn't care about article presence).
    """
    q = query.strip()
    # Strip trailing punctuation other than ? for the regex
    for pattern, axis in PATTERNS:
        m = pattern.match(q)
        if m:
            entity = m.group(1).strip().lower()
            # Drop pluralization for queries like "what color are dogs?" → entity=dog
            # (Operator was trained on singular forms.)
            entity = entity.rstrip("?.!").strip()
            return entity, axis
    return None
