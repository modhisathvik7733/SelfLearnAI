"""Retriever — top-K cosine similarity over an indexed Corpus.

For Path B's pipeline. Brain's "what's relevant?" decision happens here.
The Retriever does not decide whether to USE the retrieval — it just
returns ranked candidates with cosine scores. The pipeline (or the
admit threshold) decides whether to admit them.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .corpus import Corpus


class Retriever:
    """Top-K cosine retrieval over a Corpus.

    Construction:
        retriever = Retriever(corpus, device="cuda")  # moves ψ index to device

    Usage:
        results = retriever.retrieve(z_query, k=5)
        # results: list of (text, score, metadata) sorted by score desc
    """

    def __init__(self, corpus: Corpus, *, device: str = "cuda") -> None:
        if len(corpus) == 0:
            raise ValueError("Cannot build Retriever on empty corpus")
        self.corpus = corpus
        self.device = device
        # Pre-compute the L2-normalized index for fast cosine.
        psis = corpus.stacked_psis().to(device).float()
        self._psis_norm = F.normalize(psis, p=2, dim=-1)
        self._device = device

    @torch.no_grad()
    def retrieve(
        self,
        z_query: torch.Tensor,
        *,
        k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Returns top-k (text, score, metadata) by cosine similarity.

        `z_query` may be 1-D (D,) or 2-D (1, D). Anything else is rejected.
        """
        if z_query.dim() == 1:
            zq = z_query.unsqueeze(0)
        elif z_query.dim() == 2 and z_query.size(0) == 1:
            zq = z_query
        else:
            raise ValueError(
                f"z_query must be (D,) or (1,D); got shape {tuple(z_query.shape)}"
            )
        zq = F.normalize(zq.to(self._device).float(), p=2, dim=-1)
        # Cosine = (zq · psis_norm.T) since both unit-norm.
        scores = (zq @ self._psis_norm.T).squeeze(0)            # (N,)
        k = min(k, scores.numel())
        top_scores, top_idx = torch.topk(scores, k)
        results: list[tuple[str, float, dict[str, Any]]] = []
        for s, i in zip(top_scores.tolist(), top_idx.tolist()):
            rec = self.corpus.records[i]
            results.append((rec.text, float(s), dict(rec.metadata)))
        return results

    def __len__(self) -> int:
        return len(self.corpus)
