"""Encoder calibration — score frozen encoders against the diagnostic suite.

Reads triple test sets from `data/encoder_diagnostics/` (see Task 0.5.4)
and produces per-family geometric scores per encoder. The output drives
two architectural decisions, both honest in the reactive-adapter policy:

  1. **Per-family encoder selection.** When operating in a domain that
     stresses one geometric property (e.g., antonym separation),
     default to the encoder with the highest mean score on that
     family.

  2. **Adapter-watch flags.** When an encoder's mean score on a family
     falls below the watch threshold (default 0.05), flag the family
     for adapter consideration. Reactive — flagging does NOT auto-train
     anything; it surfaces the concern when downstream tasks in that
     family fail.

The calibration is **read-only** w.r.t. encoders (consistent with the
locked "frozen encoders, never fine-tune" policy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticTriple:
    """One row of an encoder diagnostic test set.

    The encoder is GOOD on this triple iff
        cos(anchor, positive) > cos(anchor, negative).
    """
    anchor: str
    positive: str
    negative: str


@dataclass
class DiagnosticFamily:
    """A test family — a directory under `data/encoder_diagnostics/`.

    Each family scores one geometric property (antonym separation,
    code structural similarity, paraphrase invariance, …).
    """
    name: str
    path: Path
    triples: list[DiagnosticTriple] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _strip(s: str) -> str:
    return s.strip()


def read_family(name: str, path: Path) -> DiagnosticFamily:
    """Load one diagnostic family from a triples.tsv file."""
    if not path.exists():
        raise FileNotFoundError(f"Diagnostic family file missing: {path}")
    triples: list[DiagnosticTriple] = []
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"{path}: row has < 3 tab columns: {line!r}"
                )
            triples.append(DiagnosticTriple(_strip(parts[0]), _strip(parts[1]), _strip(parts[2])))
    return DiagnosticFamily(name=name, path=path, triples=triples)


# Default suite layout: data/encoder_diagnostics/<name>/triples.tsv
DEFAULT_FAMILIES: tuple[str, ...] = ("antonym", "code_struct", "paraphrase")


def load_default_suite(root: Path | str = "data/encoder_diagnostics") -> list[DiagnosticFamily]:
    root = Path(root)
    return [
        read_family(name, root / name / "triples.tsv")
        for name in DEFAULT_FAMILIES
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_family(family: DiagnosticFamily, encode_fn: Callable) -> dict:
    """Score one diagnostic family with a given encoder.

    encode_fn: callable list[str] -> torch.Tensor of shape (n, d).
    Lazy-imports torch / numpy so module imports stay light.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if not family.triples:
        raise ValueError(f"Family {family.name!r} has no triples")

    anchors = [t.anchor for t in family.triples]
    positives = [t.positive for t in family.triples]
    negatives = [t.negative for t in family.triples]
    with torch.no_grad():
        z_a = encode_fn(anchors)
        z_p = encode_fn(positives)
        z_n = encode_fn(negatives)
        cos_p = F.cosine_similarity(z_a, z_p, dim=-1).cpu().numpy()
        cos_n = F.cosine_similarity(z_a, z_n, dim=-1).cpu().numpy()
    scores = cos_p - cos_n
    n_negative = int((scores < 0).sum())
    return {
        "name": family.name,
        "n_rows": len(family.triples),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "range": float(np.max(scores) - np.min(scores)),
        "std_score": float(np.std(scores)),
        "n_rows_negative": n_negative,
        "frac_rows_positive": (len(scores) - n_negative) / len(scores),
        "cos_positive_mean": float(np.mean(cos_p)),
        "cos_negative_mean": float(np.mean(cos_n)),
    }


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class EncoderCalibrator:
    """Score a registered set of frozen encoders against the diagnostic
    suite. Produces a (encoder × family) score table and a recommendation
    structure.

    Encoders are passed in as `(name, encode_fn)` pairs. The runner
    constructs encode_fns externally to keep this module free of HF
    dependencies.

    Watch threshold default 0.05 — a family with mean_score below this
    on the chosen encoder is flagged for adapter consideration.
    """

    def __init__(
        self,
        families: Iterable[DiagnosticFamily] | None = None,
        watch_threshold: float = 0.05,
    ):
        self.families: list[DiagnosticFamily] = (
            list(families) if families is not None else load_default_suite()
        )
        self.watch_threshold = watch_threshold

    def score(self, encoders: dict[str, Callable]) -> dict:
        """Score every (encoder, family) combination.

        Returns a structure:
          {
            "table": { encoder_name: { family_name: score_dict, ... }, ... },
            "watch_threshold": float,
            "recommendations": {  family_name: { "best_encoder", "best_score", "adapter_watch": bool }  },
            "encoders": [encoder_name, ...],
            "families": [family_name, ...],
          }
        """
        table: dict[str, dict[str, dict]] = {}
        for enc_name, encode_fn in encoders.items():
            table[enc_name] = {
                fam.name: score_family(fam, encode_fn)
                for fam in self.families
            }
        recs = encoder_recommendations(table, watch_threshold=self.watch_threshold)
        return {
            "table": table,
            "watch_threshold": self.watch_threshold,
            "recommendations": recs,
            "encoders": list(encoders.keys()),
            "families": [f.name for f in self.families],
        }


def encoder_recommendations(
    table: dict[str, dict[str, dict]],
    watch_threshold: float = 0.05,
) -> dict[str, dict]:
    """Per-family recommendations based on score table.

    For each family, identifies the best-scoring encoder and whether
    that best score still falls below the adapter-watch threshold.
    """
    if not table:
        return {}
    encoders = list(table.keys())
    families = list(table[encoders[0]].keys())

    recs: dict[str, dict] = {}
    for fam in families:
        scored = [
            (enc, table[enc][fam]["mean_score"])
            for enc in encoders
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_enc, best_score = scored[0]
        recs[fam] = {
            "best_encoder": best_enc,
            "best_score": best_score,
            "encoder_ranking": scored,
            "adapter_watch": best_score < watch_threshold,
            "watch_reason": (
                f"best mean_score {best_score:.4f} < threshold {watch_threshold:.4f}"
                if best_score < watch_threshold else None
            ),
        }
    return recs
