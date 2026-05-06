"""Corpus indexing — encode a batch of texts to ψ once, persist for reuse.

Storage: flat tensor + JSON metadata. Designed so the corpus can be
extended (append more facts) without retraining anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch


@dataclass
class CorpusRecord:
    """One indexed fact. `psi` is filled at indexing time."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    psi: Optional[torch.Tensor] = None      # (D,)

    def to_serializable(self) -> dict[str, Any]:
        return {"text": self.text, "metadata": dict(self.metadata)}


class Corpus:
    """Indexed corpus of (text, ψ, metadata) records.

    Construct via:
        corpus = Corpus.build(texts, encode_fn, metadata=...)
        corpus.save(root)
        corpus2 = Corpus.load(root)

    For runtime retrieval, pass `corpus` to a `Retriever`.
    """

    def __init__(self, records: list[CorpusRecord]) -> None:
        self.records: list[CorpusRecord] = list(records)

    # ---- Build / save / load ----------------------------------------

    @classmethod
    def build(
        cls,
        texts: list[str],
        encode_fn: Callable[[list[str]], torch.Tensor],
        *,
        metadata: Optional[list[dict[str, Any]]] = None,
        batch_size: int = 64,
    ) -> "Corpus":
        """Encode all texts in batches; return an in-memory Corpus."""
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata list must match texts length")
        records: list[CorpusRecord] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            with torch.no_grad():
                z = encode_fn(batch).detach().cpu()
            for i, text in enumerate(batch):
                meta = metadata[start + i] if metadata else {}
                records.append(CorpusRecord(
                    text=text, metadata=meta, psi=z[i].clone(),
                ))
        return cls(records)

    def save(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        # Stack ψs into a single tensor.
        psis = torch.stack([r.psi for r in self.records], dim=0)
        torch.save(psis, root / "corpus.pt")
        with open(root / "corpus.json", "w") as f:
            json.dump(
                [r.to_serializable() for r in self.records],
                f, indent=2,
            )

    @classmethod
    def load(cls, root: str | Path) -> "Corpus":
        root = Path(root)
        psis = torch.load(root / "corpus.pt", map_location="cpu", weights_only=True)
        with open(root / "corpus.json") as f:
            meta_records = json.load(f)
        if len(meta_records) != int(psis.size(0)):
            raise ValueError(
                f"corpus.json has {len(meta_records)} records but "
                f"corpus.pt has {int(psis.size(0))} ψs"
            )
        records = [
            CorpusRecord(
                text=m["text"],
                metadata=dict(m.get("metadata", {})),
                psi=psis[i].clone(),
            )
            for i, m in enumerate(meta_records)
        ]
        return cls(records)

    # ---- Read API ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def texts(self) -> list[str]:
        return [r.text for r in self.records]

    def stacked_psis(self) -> torch.Tensor:
        """All ψs as one tensor of shape (N, D)."""
        return torch.stack([r.psi for r in self.records], dim=0)
