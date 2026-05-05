"""Read corpus TSVs produced by scripts/stage2a_1_corpus.py.

The corpus generator (sub-task 2a.1) writes:
  data/explanations_v2/train.tsv     concept\tsrc\ttgt\ttemplate_idx\tsentence
  data/explanations_v2/holdout.tsv   same schema
  data/explanations_v2/metadata.json corpus stats + audit results

This module provides the reader. 2a.3's training script reads via
read_corpus_tsv(); 2a.0e/2a.0f-era inline corpus generation is replaced.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CorpusEntry:
    """One row of train.tsv or holdout.tsv."""
    concept: str
    src: str
    tgt: str
    template_idx: int
    sentence: str


def read_corpus_tsv(path: str | Path) -> list[CorpusEntry]:
    """Read a TSV with header `concept\tsrc\ttgt\ttemplate_idx\tsentence`.

    Skips header. FATALs on malformed rows (defensive — corpus
    generator should always produce well-formed output).
    """
    rows: list[CorpusEntry] = []
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        expected = ["concept", "src", "tgt", "template_idx", "sentence"]
        if header != expected:
            raise ValueError(
                f"unexpected header in {p}: got {header}, expected {expected}"
            )
        for line_no, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise ValueError(
                    f"{p}:{line_no}: expected 5 tab-separated fields, got {len(parts)}"
                )
            concept, src, tgt, template_idx_str, sentence = parts
            try:
                template_idx = int(template_idx_str)
            except ValueError:
                raise ValueError(
                    f"{p}:{line_no}: template_idx not an int: {template_idx_str!r}"
                )
            rows.append(CorpusEntry(
                concept=concept, src=src, tgt=tgt,
                template_idx=template_idx, sentence=sentence,
            ))
    return rows


def sentences_only(rows: list[CorpusEntry]) -> list[str]:
    return [r.sentence for r in rows]


def pair_index(rows: list[CorpusEntry]) -> list[tuple[str, str, str]]:
    """Return [(concept, src, tgt), ...] aligned 1:1 with rows."""
    return [(r.concept, r.src, r.tgt) for r in rows]
