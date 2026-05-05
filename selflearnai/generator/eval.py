"""Evaluation gates for Phase 2a.

Three locked gates, all required per plan §19.14:

  - cos:            median cos(encode(generated), psi_target) ≥ 0.85
  - grammar:        ≥ 95% sentences pass grammar gate (LanguageTool
                    errors == 0; falls back to wordfreq proxy with
                    explicit caveat if Java not installed)
  - word-fidelity:  ≥ 70% sentences contain BOTH target src + tgt
                    words (the gate that 2a.0c lacked)

word_pair_fidelity and build_holdout_pair_index are extracted from
`scripts/stage2a_seq_conditioning.py` and `scripts/stage2a_pointer_novel.py`
(commits 2a5ce56, d554c6c) — the validated implementations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Soft import — quick_probe lives in scripts/, not the package, so
# importing it here would couple package to scripts. Instead we
# duplicate the small `grammar_grade` interface here. For the actual
# rule-based grading the user runs the scripts directly.
try:
    from wordfreq import zipf_frequency
    _HAS_WORDFREQ = True
except ImportError:
    _HAS_WORDFREQ = False
    def zipf_frequency(word: str, lang: str) -> float:        # type: ignore
        del word, lang
        return 0.0


_WORD_RE = re.compile(r"[a-zA-Z]+")


def _words_in_text(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def word_pair_fidelity(src: str, tgt: str, generated: str) -> tuple[bool, bool, bool]:
    """Returns (src_in_gen, tgt_in_gen, both_in_gen).

    A correct held-out generation should contain BOTH src and tgt
    words. 2a.0c had 0/168 exact matches and was producing valid
    templates with wrong word pairs — exactly what this metric
    catches.
    """
    gen_words = _words_in_text(generated)
    src_in = src.lower() in gen_words
    tgt_in = tgt.lower() in gen_words
    return src_in, tgt_in, (src_in and tgt_in)


def grammar_proxy(text: str) -> float:
    """Fast grammar proxy via wordfreq.zipf_frequency. Higher = more
    grammatical-looking. Range roughly [0, 1]; threshold 0.55 used as
    a fallback for the LanguageTool gate when Java isn't installed.

    NOTE: this is approximate; the gold-standard for the §9.4 grammar
    gate is LanguageTool (rule-based, no LLM, requires Java). The
    proxy is known to false-positive on word-salad with common words
    (see 2a.0c failure mode in §19.13). DO NOT rely on this proxy as
    a closing gate — install Java for production runs.
    """
    if not _HAS_WORDFREQ:
        return 0.0
    words = text.lower().split()
    if not words:
        return 0.0
    zipfs = [zipf_frequency(w, "en") for w in words]
    mean_zipf = sum(zipfs) / len(zipfs)
    common_score = mean_zipf / 6.0
    adjacent_dups = sum(
        1 for i in range(1, len(words)) if words[i] == words[i - 1]
    )
    rep_penalty = adjacent_dups / max(1, len(words) - 1)
    n_rare = sum(1 for z in zipfs if z < 2.0)
    rare_penalty = n_rare / len(words)
    return float(max(0.0, common_score - 0.3 * rep_penalty - 0.2 * rare_penalty))


def build_holdout_pair_index(
    holdout_corpus: Iterable[tuple[str, str, str, int, str]],
) -> list[tuple[str, str, str]]:
    """Given holdout corpus rows of form (concept, src, tgt, template_idx, sentence),
    return a list of (concept, src, tgt) — one per row, aligned 1:1
    with the order the rows arrive.

    This replaces the 2a.0f-era version that re-walked TEMPLATES ×
    truly-novel-pairs to recover the index. Now that 2a.1 writes
    train.tsv and holdout.tsv with concept/src/tgt columns, the index
    is just a projection.
    """
    return [(concept, src, tgt) for concept, src, tgt, _ti, _sent in holdout_corpus]


# ---------------------------------------------------------------------------
# Gate aggregation
# ---------------------------------------------------------------------------

@dataclass
class GenerationVerdict:
    """Single-record outcome — one held-out sentence's verdict."""
    target: str
    generated: str
    concept: str
    src_word: str
    tgt_word: str
    cos_recovered: float
    grammar_pass: bool
    grammar_n_errors: int
    grammar_proxy: float
    src_in_gen: bool
    tgt_in_gen: bool
    both_in_gen: bool
    exact_match: bool
    p_gen_mean: Optional[float] = None


@dataclass
class GeneralizationGates:
    """Roll-up of per-record verdicts into the §19.14 closing gates."""
    n_holdout: int
    median_cos: float
    n_cos_pass: int
    n_grammar_pass: int
    n_src_in: int
    n_tgt_in: int
    n_both_in: int
    n_exact_match: int
    cos_min: float
    grammar_pass_rate: float
    word_fidelity_min: float
    p_gen_overall: Optional[float] = None

    @property
    def cos_gate(self) -> bool:
        return self.median_cos >= self.cos_min

    @property
    def grammar_gate(self) -> bool:
        return self.n_grammar_pass >= int(self.grammar_pass_rate * self.n_holdout)

    @property
    def word_fidelity_gate(self) -> bool:
        return self.n_both_in >= int(self.word_fidelity_min * self.n_holdout)

    @property
    def all_pass(self) -> bool:
        return self.cos_gate and self.grammar_gate and self.word_fidelity_gate

    def summary_lines(self) -> list[str]:
        gt = int(self.grammar_pass_rate * self.n_holdout)
        wt = int(self.word_fidelity_min * self.n_holdout)
        cos_mark = "✓" if self.cos_gate else "✗"
        gr_mark = "✓" if self.grammar_gate else "✗"
        wf_mark = "✓" if self.word_fidelity_gate else "✗"
        out = [
            f"  {cos_mark} median cos:                    "
            f"{self.median_cos:.4f}  (target ≥ {self.cos_min})",
            f"      sentences passing cos:           {self.n_cos_pass}/{self.n_holdout}",
            f"  {gr_mark} sentences passing grammar:     "
            f"{self.n_grammar_pass}/{self.n_holdout}  (target ≥ {gt})",
            f"      sentences with src word:         {self.n_src_in}/{self.n_holdout}",
            f"      sentences with tgt word:         {self.n_tgt_in}/{self.n_holdout}",
            f"  {wf_mark} sentences with BOTH (gate):    "
            f"{self.n_both_in}/{self.n_holdout}  (target ≥ {wt})",
            f"      exact target matches:            {self.n_exact_match}/{self.n_holdout}",
        ]
        if self.p_gen_overall is not None:
            out.append(f"      mean p_gen across holdout:       {self.p_gen_overall:.3f}")
        return out


def roll_up_gates(
    verdicts: list[GenerationVerdict],
    *,
    cos_min: float = 0.85,
    grammar_pass_rate: float = 0.95,
    word_fidelity_min: float = 0.70,
) -> GeneralizationGates:
    """Aggregate per-record verdicts into the closing gates."""
    coses = sorted(v.cos_recovered for v in verdicts)
    median_cos = coses[len(coses) // 2] if coses else 0.0
    n = len(verdicts)
    p_gens = [v.p_gen_mean for v in verdicts if v.p_gen_mean is not None]
    p_gen_overall = (sum(p_gens) / len(p_gens)) if p_gens else None
    return GeneralizationGates(
        n_holdout=n,
        median_cos=median_cos,
        n_cos_pass=sum(1 for v in verdicts if v.cos_recovered >= cos_min),
        n_grammar_pass=sum(1 for v in verdicts if v.grammar_pass),
        n_src_in=sum(1 for v in verdicts if v.src_in_gen),
        n_tgt_in=sum(1 for v in verdicts if v.tgt_in_gen),
        n_both_in=sum(1 for v in verdicts if v.both_in_gen),
        n_exact_match=sum(1 for v in verdicts if v.exact_match),
        cos_min=cos_min,
        grammar_pass_rate=grammar_pass_rate,
        word_fidelity_min=word_fidelity_min,
        p_gen_overall=p_gen_overall,
    )
