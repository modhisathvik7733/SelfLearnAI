"""Versioned concept registry with growth control (Task 1.5.5).

Persistent storage substrate where validated concepts (from
discovery.validate) and promoted macros (from discovery.refactor)
actually land. Without this, every sleep cycle re-discovers the same
concepts because there's nowhere to write "this concept is now in
the live library."

Per safeguard #5 (plan §19.9, memory feedback_stage1_5_safeguards),
the registry has explicit growth control:

  - `max_active_concepts` cap on simultaneous active concepts.
  - When full, prune by LRU (`last_used_at`). Future extension:
    utility-decay (rank by usage_count / time, not just recency).
  - Pruned entries are RETIRED — moved to an archive subdirectory
    so we never lose the work, but they don't load into the active
    operator library.

Versioning + rollback per safeguard #5: each concept can have
multiple versions; the current version is what the planner sees.
A new fit (e.g., from sleep refining the operator on a larger
cluster) bumps the version; rollback sets a previous version as
current without deleting either.

On-disk layout:

  <root>/
    concepts/<concept_id>.json         active concept metadata
    archive/<concept_id>.json           retired concept metadata
    state_dicts/<concept_id>__v<N>.pt   operator weights (per-version)

Each <concept_id>.json contains the full ConceptEntry.to_dict() —
version list, current_version, last_used_at, support counts. State
dicts are loaded lazily on `get_op` to keep registry boot cheap.

Note: the registry assumes operators share a single architecture
(`ConceptOperator(dim=...)`). Macros (compound ops) cannot live in
the registry directly because they're function compositions, not
nn.Modules. Macros are persisted by their CHAIN — the registry
stores the chain string + the chain's component concept IDs, and
macros are reconstructed at planner-init time via `make_macro_op`
on the (concept, op) lookup. (Implemented in a later sub-task.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from selflearnai.concepts.operator import ConceptOperator


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConceptVersion:
    """One version of a concept's operator.

    Every fit (initial register / bump_version) creates a new version.
    Old versions are kept on disk for rollback; only the
    `current_version` is exposed by `get_op`.
    """
    version: int
    state_dict_path: str           # relative to registry root
    provenance: dict[str, Any] = field(default_factory=dict)
    support_count: int = 0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConceptVersion":
        return cls(**d)


@dataclass
class ConceptEntry:
    """A single concept in the registry. May have multiple versions."""
    concept_id: str
    versions: list[ConceptVersion]
    current_version: int
    last_used_at: str

    def current(self) -> ConceptVersion:
        for v in self.versions:
            if v.version == self.current_version:
                return v
        raise KeyError(
            f"concept {self.concept_id!r}: current_version={self.current_version} "
            f"not found in versions"
        )

    def get_version(self, version: int) -> ConceptVersion:
        for v in self.versions:
            if v.version == version:
                return v
        raise KeyError(
            f"concept {self.concept_id!r}: version {version} not found"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "versions": [v.to_dict() for v in self.versions],
            "current_version": self.current_version,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConceptEntry":
        return cls(
            concept_id=d["concept_id"],
            versions=[ConceptVersion.from_dict(v) for v in d["versions"]],
            current_version=d["current_version"],
            last_used_at=d["last_used_at"],
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ConceptRegistry:
    """Versioned, on-disk concept registry with LRU growth control.

    Typical lifecycle (driven by sleep cycle):

        registry = ConceptRegistry(root, dim=1024, max_active=50)

        # New concept proposed by clustering + validated by validate.py:
        registry.register(
            concept_id="plurality_v3",
            op=trained_concept_op,
            provenance={"source": "sleep_cycle_42",
                        "cluster_silhouette": 0.55,
                        "validation": validation_result.to_dict()},
            support_count=12,
        )

        # Planner asks for the operator:
        op = registry.get_op("plurality_v3")    # bumps last_used_at

        # A later sleep cycle refines the operator on more data:
        registry.bump_version(
            concept_id="plurality_v3",
            op=better_op,
            provenance={"source": "sleep_cycle_53",
                        "support_count_after": 38},
            support_count=38,
        )

        # Regression caught — rollback:
        registry.rollback("plurality_v3", target_version=1)
    """

    def __init__(
        self,
        root: str | Path,
        *,
        dim: int,
        max_active: int = 50,
        operator_factory=None,
    ) -> None:
        self.root = Path(root)
        self.dim = dim
        self.max_active = max_active
        # Allow callers to inject an alternate ConceptOperator-shaped class
        # (e.g. multi-head variants). Default is the canonical ConceptOperator.
        self.operator_factory = operator_factory or (lambda: ConceptOperator(dim=dim))
        self._entries: dict[str, ConceptEntry] = {}
        # Initialize on-disk layout.
        (self.root / "concepts").mkdir(parents=True, exist_ok=True)
        (self.root / "archive").mkdir(parents=True, exist_ok=True)
        (self.root / "state_dicts").mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    # ---- Persistence ------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Read all <root>/concepts/<id>.json into memory."""
        self._entries.clear()
        for path in sorted((self.root / "concepts").glob("*.json")):
            with open(path) as f:
                entry = ConceptEntry.from_dict(json.load(f))
            self._entries[entry.concept_id] = entry

    def _save_entry(self, entry: ConceptEntry) -> None:
        path = self.root / "concepts" / f"{entry.concept_id}.json"
        with open(path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

    def _archive_entry(self, entry: ConceptEntry) -> None:
        """Move active entry's metadata to archive/ (state_dicts kept
        in place — rollback / re-activation could come back to it)."""
        active_path = self.root / "concepts" / f"{entry.concept_id}.json"
        archive_path = self.root / "archive" / f"{entry.concept_id}.json"
        if active_path.exists():
            active_path.rename(archive_path)
        elif not archive_path.exists():
            # Shouldn't normally happen — write archive directly.
            with open(archive_path, "w") as f:
                json.dump(entry.to_dict(), f, indent=2)

    # ---- Read API ---------------------------------------------------------

    def list_active(self) -> list[ConceptEntry]:
        """All currently-active entries, sorted by last_used_at descending."""
        return sorted(
            self._entries.values(),
            key=lambda e: e.last_used_at,
            reverse=True,
        )

    def has(self, concept_id: str) -> bool:
        return concept_id in self._entries

    def get_entry(self, concept_id: str) -> ConceptEntry:
        if concept_id not in self._entries:
            raise KeyError(f"concept {concept_id!r} not active in registry")
        return self._entries[concept_id]

    def get_op(self, concept_id: str) -> ConceptOperator:
        """Load the current version's operator. Touches last_used_at."""
        entry = self.get_entry(concept_id)
        version = entry.current()
        sd_path = self.root / version.state_dict_path
        if not sd_path.exists():
            raise FileNotFoundError(
                f"concept {concept_id!r} v{version.version}: "
                f"state_dict file missing at {sd_path}"
            )
        op = self.operator_factory()
        op.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
        op.eval()
        for p in op.parameters():
            p.requires_grad_(False)
        self._touch(concept_id)
        return op

    def _touch(self, concept_id: str) -> None:
        entry = self._entries[concept_id]
        entry.last_used_at = _now_iso()
        self._save_entry(entry)

    # ---- Write API --------------------------------------------------------

    def register(
        self,
        concept_id: str,
        op: ConceptOperator,
        *,
        provenance: Optional[dict[str, Any]] = None,
        support_count: int = 0,
    ) -> ConceptEntry:
        """Register a new concept. If the id already exists, this is a
        bump_version (semantic: "register the latest fit")."""
        if concept_id in self._entries:
            return self.bump_version(
                concept_id, op, provenance=provenance, support_count=support_count,
            )
        version = 1
        sd_rel = f"state_dicts/{concept_id}__v{version}.pt"
        torch.save(op.state_dict(), self.root / sd_rel)
        cv = ConceptVersion(
            version=version,
            state_dict_path=sd_rel,
            provenance=dict(provenance or {}),
            support_count=support_count,
            created_at=_now_iso(),
        )
        entry = ConceptEntry(
            concept_id=concept_id,
            versions=[cv],
            current_version=version,
            last_used_at=_now_iso(),
        )
        self._entries[concept_id] = entry
        self._save_entry(entry)
        self._prune_if_full()
        return entry

    def bump_version(
        self,
        concept_id: str,
        op: ConceptOperator,
        *,
        provenance: Optional[dict[str, Any]] = None,
        support_count: int = 0,
    ) -> ConceptEntry:
        """Add a new version to an existing concept and set it as current."""
        entry = self.get_entry(concept_id)
        new_version = max(v.version for v in entry.versions) + 1
        sd_rel = f"state_dicts/{concept_id}__v{new_version}.pt"
        torch.save(op.state_dict(), self.root / sd_rel)
        cv = ConceptVersion(
            version=new_version,
            state_dict_path=sd_rel,
            provenance=dict(provenance or {}),
            support_count=support_count,
            created_at=_now_iso(),
        )
        entry.versions.append(cv)
        entry.current_version = new_version
        entry.last_used_at = _now_iso()
        self._save_entry(entry)
        return entry

    def rollback(self, concept_id: str, target_version: int) -> ConceptEntry:
        """Set a previous version as the current. Both old and new
        version state_dicts are retained; this is reversible."""
        entry = self.get_entry(concept_id)
        # Validate target exists.
        entry.get_version(target_version)
        entry.current_version = target_version
        entry.last_used_at = _now_iso()
        self._save_entry(entry)
        return entry

    def retire(self, concept_id: str) -> None:
        """Remove from active set; metadata moves to archive/."""
        if concept_id not in self._entries:
            return
        entry = self._entries.pop(concept_id)
        self._archive_entry(entry)

    # ---- Growth control ---------------------------------------------------

    def _prune_if_full(self) -> list[str]:
        """If len(active) > max_active, retire the LRU entries until under cap.
        Returns the list of concept_ids retired in this prune."""
        if len(self._entries) <= self.max_active:
            return []
        # Sort ascending by last_used_at; retire from the front.
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.last_used_at,
        )
        n_to_retire = len(self._entries) - self.max_active
        retired: list[str] = []
        for entry in sorted_entries[:n_to_retire]:
            self.retire(entry.concept_id)
            retired.append(entry.concept_id)
        return retired


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Microsecond-precision UTC ISO timestamp.

    LRU pruning relies on `last_used_at` being a strict total order — if
    two `_touch` calls fall in the same second, an LRU sort by string
    timestamp at second-resolution gets a tie, and Python's sort is
    stable but the prune behavior becomes input-order-dependent.
    Microsecond precision keeps successive calls strictly ordered.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
