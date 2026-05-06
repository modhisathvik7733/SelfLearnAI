"""selflearnai.memory — semantic-memory layer (Plan §8).

The brain's "remember" component. Holds an indexed corpus of facts /
sentences / passages, encoded by E5 once and looked up by ψ-cosine
similarity at query time.

Used by Path B's pipeline to produce factual grounding for the LM
renderer: the brain decides which facts apply (via retrieval), the LM
renders them fluently, the verifier gates the output.

Public modules:
  - corpus:    CorpusRecord, Corpus
  - retriever: Retriever (top-K cosine retrieval)

Storage layout (when persisted):
  <root>/
    corpus.pt       float tensor (N × D) of pooled ψs
    corpus.json     N records of {"text": ..., "metadata": {...}}
"""
from .corpus import Corpus, CorpusRecord
from .retriever import Retriever

__all__ = [
    "Corpus",
    "CorpusRecord",
    "Retriever",
]
