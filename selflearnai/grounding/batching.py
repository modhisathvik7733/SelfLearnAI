"""Batch construction — locked rules from plan section 1.5.

Three rules, all prerequisites (not nice-to-haves):
  1. Hard negatives, not random   — rolling memory bank, top-K most similar.
  2. Cross-category coverage      — stratified batching across semantic groups.
  3. Count variation in vision    — stratified batching across object counts.

A working architecture with random batches will silently fail.
A slightly weaker architecture with the right batches will succeed.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Memory bank — rolling buffer of recent embeddings for hard-negative mining
# ---------------------------------------------------------------------------
class EmbeddingMemoryBank:
    """Fixed-size FIFO of embeddings + their identifiers.

    Used for hard-negative mining: at each training step, after computing
    InfoNCE on the current batch's positives, we ALSO push the batch's
    embeddings into the bank and pull the top-K most-similar non-matching
    embeddings from the bank as additional negatives.
    """

    def __init__(self, dim: int, capacity: int = 4096, device: str = "cpu"):
        self.dim = dim
        self.capacity = capacity
        self.device = device
        self.buffer = torch.zeros(capacity, dim, device=device)
        self.ids: list[str] = [""] * capacity
        self.size = 0
        self.head = 0  # next write position

    @torch.no_grad()
    def push(self, embeddings: torch.Tensor, ids: Iterable[str]) -> None:
        """Append (embeddings, ids) into the FIFO."""
        ids = list(ids)
        assert embeddings.dim() == 2 and embeddings.size(1) == self.dim
        assert len(ids) == embeddings.size(0)
        embeddings = embeddings.detach().to(self.device)
        for i, eid in enumerate(ids):
            self.buffer[self.head] = embeddings[i]
            self.ids[self.head] = eid
            self.head = (self.head + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    @torch.no_grad()
    def hard_negatives(
        self,
        query: torch.Tensor,           # (B, D)
        exclude_ids: Iterable[str],
        k: int = 64,
    ) -> torch.Tensor:
        """Return the top-K most-similar bank entries that are NOT in exclude_ids.
        Returns (M, D) where M ≤ K. May return an empty tensor if bank is empty.
        """
        if self.size == 0:
            return torch.zeros(0, self.dim, device=self.device)
        active = self.buffer[: self.size]                                # (S, D)
        active_ids = self.ids[: self.size]
        q_n = F.normalize(query, dim=-1)
        a_n = F.normalize(active, dim=-1)
        # (B, S) similarity of each query to each bank entry, then aggregate
        # over the batch by max — pulls examples that are hard for *any* query.
        sims = (q_n @ a_n.T).max(dim=0).values
        # Mask out excluded ids.
        excl = set(exclude_ids)
        for i, eid in enumerate(active_ids):
            if eid in excl:
                sims[i] = -float("inf")
        k_eff = min(k, self.size)
        top_idx = sims.topk(k_eff).indices
        return active[top_idx]


# ---------------------------------------------------------------------------
# Stratified batch builder — enforces category and count coverage per batch
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    """One training datum: a (text, image_id, metadata) triple.

    text     : caption string (or short description).
    image_id : opaque identifier; loader resolves it to a PIL.Image lazily.
    category : top-level semantic group (e.g., 'animal', 'object', 'place').
    count    : number of countable entities in the visual scene (1, 2, 3, ...).
    """
    text: str
    image_id: str
    category: str
    count: int


class StratifiedBatchBuilder:
    """Yields batches that satisfy the three locked rules:

      • Each batch has at least `min_categories` distinct categories.
      • Each batch has at least `min_count_levels` distinct count levels.
      • Sampling is otherwise uniform.

    The builder does NOT load images. It returns lists of `Sample` objects;
    a separate dataset/loader resolves image_id → PIL.Image and calls the
    foundation encoders.
    """

    def __init__(
        self,
        samples: list[Sample],
        batch_size: int = 64,
        min_categories: int = 4,
        min_count_levels: int = 3,
        seed: int = 0,
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.min_categories = min_categories
        self.min_count_levels = min_count_levels
        self.rng = random.Random(seed)

        # Indices grouped by (category, count) for fast stratified pulls.
        self._by_category: dict[str, list[int]] = defaultdict(list)
        self._by_count: dict[int, list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            self._by_category[s.category].append(i)
            self._by_count[s.count].append(i)

        if len(self._by_category) < min_categories:
            raise ValueError(
                f"Dataset has only {len(self._by_category)} categories; need ≥ {min_categories}."
            )
        if len(self._by_count) < min_count_levels:
            raise ValueError(
                f"Dataset has only {len(self._by_count)} count levels; need ≥ {min_count_levels}."
            )

    def __iter__(self):
        return self

    def __next__(self) -> list[Sample]:
        return self._sample_batch()

    def _sample_batch(self) -> list[Sample]:
        chosen: list[int] = []
        used: set[int] = set()

        # Stage 1: ensure category coverage.
        cats = self.rng.sample(list(self._by_category.keys()), k=self.min_categories)
        for c in cats:
            i = self.rng.choice(self._by_category[c])
            if i not in used:
                chosen.append(i); used.add(i)

        # Stage 2: ensure count coverage.
        counts = self.rng.sample(list(self._by_count.keys()), k=self.min_count_levels)
        for c in counts:
            i = self.rng.choice(self._by_count[c])
            if i not in used:
                chosen.append(i); used.add(i)

        # Stage 3: fill the rest uniformly at random.
        remaining = self.batch_size - len(chosen)
        if remaining > 0:
            pool = [i for i in range(len(self.samples)) if i not in used]
            extra = self.rng.sample(pool, k=min(remaining, len(pool)))
            chosen.extend(extra)

        return [self.samples[i] for i in chosen]
