"""PropertyOperatorPackage — bundles a trained property RelationalOperator
with everything needed to query it.

A trained property operator alone is useless for inference: you also
need the AxisVocabulary (axis name → index), the pool_mean (used at
training to debias encoder anisotropy), and the candidate value pool
(strings + their encoded ψs) to do nearest-neighbor retrieval.

This class persists all four to disk under a single root and provides
one-call query semantics:

    pkg = PropertyOperatorPackage.load(root, encode_fn, device="cuda")
    topk = pkg.query("plant", "requires", k=5)
    # → [("sunlight", 0.89), ("water", 0.85), ("soil", 0.83), ...]

On-disk layout (under <root>/):
    operator.pt       RelationalOperator state_dict + arch metadata
    vocab.json        AxisVocabulary (axis names list)
    pool_mean.pt      (1, D) tensor or absent if centering was OFF
    value_pool.json   list[str] of candidate values (training-time pool)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .relational_operator import AxisVocabulary, RelationalOperator


@dataclass
class PropertyTopK:
    """One query result: top-K candidate values with cosine scores."""
    entity: str
    axis: str
    candidates: list[tuple[str, float]]   # [(value, score), ...] descending

    @property
    def top_value(self) -> str:
        return self.candidates[0][0] if self.candidates else ""

    @property
    def top_score(self) -> float:
        return self.candidates[0][1] if self.candidates else 0.0

    @property
    def margin(self) -> float:
        """Top-1 score minus top-2 score. Higher = more confident."""
        if len(self.candidates) < 2:
            return 0.0
        return self.candidates[0][1] - self.candidates[1][1]


class PropertyOperatorPackage:
    """Trained property operator + everything needed to query it.

    Construct via PropertyOperatorPackage.load() AFTER training.
    The training script (curriculum_tier0_property.py) writes the
    package via .save_from_training(...).
    """

    def __init__(
        self,
        op: RelationalOperator,
        vocab: AxisVocabulary,
        pool_mean: Optional[torch.Tensor],     # (1, D) or None
        value_pool: list[str],
        value_pool_psi: torch.Tensor,           # (P, D) — already centered if pool_mean
        device: str = "cuda",
    ) -> None:
        self.op = op
        self.vocab = vocab
        self.pool_mean = pool_mean
        self.value_pool = list(value_pool)
        # Pre-normalize for fast cosine retrieval
        self.value_pool_psi = value_pool_psi
        self._value_pool_psi_norm = F.normalize(value_pool_psi, p=2, dim=-1)
        self.device = device

    # ---- Save / load ------------------------------------------------

    @staticmethod
    def save_from_training(
        root: str | Path,
        op: RelationalOperator,
        vocab: AxisVocabulary,
        pool_mean: Optional[torch.Tensor],
        value_pool: list[str],
    ) -> None:
        """Persist a freshly-trained operator package to <root>/.

        `value_pool` is the list of candidate value STRINGS at training
        time (ψs are encoded fresh on load — no need to persist them).
        `pool_mean` is the centering vector fitted on training values
        (None if centering was off).
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        op.save(root / "operator.pt")
        vocab.save(root / "vocab.json")
        if pool_mean is not None:
            torch.save(pool_mean.detach().cpu(), root / "pool_mean.pt")
        with open(root / "value_pool.json", "w") as f:
            json.dump(list(value_pool), f, indent=2)

    @classmethod
    def load(
        cls,
        root: str | Path,
        encode_fn: Callable[[list[str]], torch.Tensor],
        device: str = "cuda",
    ) -> "PropertyOperatorPackage":
        """Restore the full package from disk + re-encode the value pool."""
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"property package root not found: {root}")
        op = RelationalOperator.load(root / "operator.pt").to(device)
        vocab = AxisVocabulary.load(root / "vocab.json")
        pool_mean: Optional[torch.Tensor] = None
        if (root / "pool_mean.pt").exists():
            pool_mean = torch.load(
                str(root / "pool_mean.pt"), map_location=device, weights_only=True,
            )
        with open(root / "value_pool.json") as f:
            value_pool = json.load(f)
        # Re-encode value pool with current encoder (consistent with training).
        z_pool = encode_fn(value_pool)
        if pool_mean is not None:
            z_pool = z_pool - pool_mean
        return cls(
            op=op, vocab=vocab,
            pool_mean=pool_mean,
            value_pool=value_pool,
            value_pool_psi=z_pool,
            device=device,
        )

    # ---- Query ------------------------------------------------------

    def has_axis(self, axis: str) -> bool:
        return axis in self.vocab

    def axis_names(self) -> list[str]:
        return self.vocab.axis_names()

    @torch.no_grad()
    def query(
        self,
        entity: str,
        axis: str,
        encode_fn: Callable[[list[str]], torch.Tensor],
        *,
        k: int = 5,
    ) -> PropertyTopK:
        """(entity, axis) → PropertyTopK with k candidate values + scores."""
        if axis not in self.vocab:
            raise KeyError(
                f"axis {axis!r} unknown; trained axes: {self.vocab.axis_names()}"
            )
        z_entity = encode_fn([entity]).squeeze(0).flatten()
        if self.pool_mean is not None:
            z_entity = z_entity - self.pool_mean.squeeze(0)
        axis_idx = self.vocab[axis]
        z_pred = self.op(z_entity, axis_idx).flatten()
        z_pred_norm = F.normalize(z_pred.unsqueeze(0), p=2, dim=-1).squeeze(0)
        scores = self._value_pool_psi_norm @ z_pred_norm           # (P,)
        topk = torch.topk(scores, k=min(k, scores.numel()))
        candidates = [
            (self.value_pool[idx], float(score))
            for idx, score in zip(topk.indices.tolist(), topk.values.tolist())
        ]
        return PropertyTopK(entity=entity, axis=axis, candidates=candidates)
