"""selflearnai.generator — Phase 2a non-AR text generator (locked architecture).

Produces surface-form text (English explanations, code, JSON, ...) from
a Stage-1 PsiProgram. The architecture is the empirically-validated
recipe from plan §19.14:

  - Frozen E5-large-v2 encoder (preserved from Stages 0–1.5).
  - Sequence conditioning on the encoder's per-token activation
    sequence h ∈ R^{T_in × 1024} — NOT the pooled ψ vector.
  - Non-AR transformer decoder with bidirectional self-attention
    on output positions and cross-attention to the encoder's
    activations.
  - Pointer-Generator output head (See/Liu/Manning 2017): per-position
    mixture of vocab logits and copy-attention over encoder input
    tokens. The copy mechanism is what makes word-pair fidelity work
    on truly-novel inputs (validated on 2a.0f, 80% bit-exact match
    on words the model never saw during training).
  - Training loss: parallel position-wise NLL on the mixture +
    activation MSE + perturbation augmentation (Gaussian δ=0.7 OR
    30% token-mask) + feature dropout on conditioning tokens.

Public modules:
  - decoder:    PointerSeqCondDecoder
  - loss:       perturb_h, MseProjection, mixture_nll
  - sample:     decode_to_text, multi_candidate_sample
  - eval:       word_pair_fidelity, build_holdout_pair_index, gates
  - corpus:     read_corpus_tsv, CorpusEntry
"""
from .decoder import PointerSeqCondDecoder
from .loss import perturb_h, mixture_nll
from .sample import decode_to_text, multi_candidate_sample
from .eval import (
    word_pair_fidelity,
    build_holdout_pair_index,
    GeneralizationGates,
    GenerationVerdict,
)
from .corpus import read_corpus_tsv, CorpusEntry

__all__ = [
    "PointerSeqCondDecoder",
    "perturb_h",
    "mixture_nll",
    "decode_to_text",
    "multi_candidate_sample",
    "word_pair_fidelity",
    "build_holdout_pair_index",
    "GeneralizationGates",
    "GenerationVerdict",
    "read_corpus_tsv",
    "CorpusEntry",
]
