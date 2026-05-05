"""Versioned domain registry for Stage 3 (sub-task 3.3).

Mirrors Stage 1.5's ConceptRegistry (selflearnai/discovery/registry.py,
commit 5216c74 — all 5 round-trip sub-cases passed) but stores DOMAINS
instead of CONCEPTS. Each domain entry references multiple per-version
artifacts: decoder checkpoint (Phase 2a recipe), energy-model checkpoint
(sub-task 3.2), and an optional conformal-calibration record (sub-task
3.6 will populate).

On-disk layout:

  <root>/
    domains/<domain_id>.json        active domain metadata
    archive/<domain_id>.json         retired domain metadata
    artifacts/<domain_id>__v<N>/    per-version artifact subdir:
      decoder.pt                    PointerSeqCondDecoder state_dict
      energy_<kind>.pt              per-domain energy model
      provenance.json               training/eval stats

Each <domain_id>.json contains the full DomainEntry.to_dict():
versions list, current_version, last_used_at, etc.

Same growth-control + rollback discipline as Stage 1.5: cap on
max_active_domains, LRU prune (move to archive/), versioned with
explicit rollback.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DomainVersion:
    """One version of a domain — decoder + energy + provenance."""
    version: int
    decoder_path: str           # relative to registry root
    energy_path: str            # relative to registry root
    energy_kind: str            # "GaussianEnergy" | "MLPEnergyModel"
    conformal_path: Optional[str] = None    # set by 3.6
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DomainVersion":
        return cls(**d)


@dataclass
class DomainEntry:
    """A single domain in the registry. May have multiple versions."""
    domain_id: str
    versions: list[DomainVersion]
    current_version: int
    last_used_at: str

    def current(self) -> DomainVersion:
        for v in self.versions:
            if v.version == self.current_version:
                return v
        raise KeyError(
            f"domain {self.domain_id!r}: current_version="
            f"{self.current_version} not found"
        )

    def get_version(self, version: int) -> DomainVersion:
        for v in self.versions:
            if v.version == version:
                return v
        raise KeyError(
            f"domain {self.domain_id!r}: version {version} not found"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "versions": [v.to_dict() for v in self.versions],
            "current_version": self.current_version,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DomainEntry":
        return cls(
            domain_id=d["domain_id"],
            versions=[DomainVersion.from_dict(v) for v in d["versions"]],
            current_version=d["current_version"],
            last_used_at=d["last_used_at"],
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class DomainRegistry:
    """Versioned, on-disk domain registry with LRU growth control.

    Typical lifecycle (driven by sub-task 3.4 orchestrator):

        reg = DomainRegistry(root, max_active=20)

        # Register a freshly-ingested domain (after train/eval pass):
        reg.register(
            domain_id="definitional",
            decoder_src_path="data/explanations_v2/checkpoints/decoder_3a1.pt",
            energy_src_path="data/explanations_v2/checkpoints/energy_def_mlp.pt",
            energy_kind="MLPEnergyModel",
            provenance={
                "n_train_sents": 500,
                "n_holdout_sents": 120,
                "decoder_eval": {...},
                "energy_eval": {"roc_auc": 1.0000, "model": "MLPEnergyModel"},
            },
        )

        # Look up a domain at planner-init time:
        entry = reg.get_entry("definitional")
        v = entry.current()
        decoder_ckpt = reg.root / v.decoder_path
        energy_ckpt = reg.root / v.energy_path

        # A later improvement (better hyperparams, new conformal calibration):
        reg.bump_version(...)

        # Regression caught — rollback:
        reg.rollback("definitional", target_version=1)
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_active: int = 20,
    ) -> None:
        self.root = Path(root)
        self.max_active = max_active
        self._entries: dict[str, DomainEntry] = {}
        (self.root / "domains").mkdir(parents=True, exist_ok=True)
        (self.root / "archive").mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    # ---- Persistence ------------------------------------------------------

    def _load_from_disk(self) -> None:
        self._entries.clear()
        for path in sorted((self.root / "domains").glob("*.json")):
            with open(path) as f:
                entry = DomainEntry.from_dict(json.load(f))
            self._entries[entry.domain_id] = entry

    def _save_entry(self, entry: DomainEntry) -> None:
        path = self.root / "domains" / f"{entry.domain_id}.json"
        with open(path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

    def _archive_entry(self, entry: DomainEntry) -> None:
        active = self.root / "domains" / f"{entry.domain_id}.json"
        archived = self.root / "archive" / f"{entry.domain_id}.json"
        if active.exists():
            active.rename(archived)
        elif not archived.exists():
            with open(archived, "w") as f:
                json.dump(entry.to_dict(), f, indent=2)

    # ---- Read API ---------------------------------------------------------

    def list_active(self) -> list[DomainEntry]:
        return sorted(
            self._entries.values(),
            key=lambda e: e.last_used_at,
            reverse=True,
        )

    def has(self, domain_id: str) -> bool:
        return domain_id in self._entries

    def get_entry(self, domain_id: str) -> DomainEntry:
        if domain_id not in self._entries:
            raise KeyError(f"domain {domain_id!r} not active in registry")
        self._touch(domain_id)
        return self._entries[domain_id]

    def _touch(self, domain_id: str) -> None:
        entry = self._entries[domain_id]
        entry.last_used_at = _now_iso()
        self._save_entry(entry)

    # ---- Write API --------------------------------------------------------

    def register(
        self,
        domain_id: str,
        *,
        decoder_src_path: str | Path,
        energy_src_path: str | Path,
        energy_kind: str,
        conformal_src_path: Optional[str | Path] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> DomainEntry:
        """Register a new domain or bump the version of an existing one.

        Copies the source artifact files into the registry's
        artifacts/<domain_id>__v<N>/ subdir so the registry owns the
        canonical paths.
        """
        if domain_id in self._entries:
            return self.bump_version(
                domain_id,
                decoder_src_path=decoder_src_path,
                energy_src_path=energy_src_path,
                energy_kind=energy_kind,
                conformal_src_path=conformal_src_path,
                provenance=provenance,
            )
        version = 1
        cv = self._stash_artifacts(
            domain_id, version,
            decoder_src_path=decoder_src_path,
            energy_src_path=energy_src_path,
            energy_kind=energy_kind,
            conformal_src_path=conformal_src_path,
            provenance=provenance,
        )
        entry = DomainEntry(
            domain_id=domain_id,
            versions=[cv],
            current_version=version,
            last_used_at=_now_iso(),
        )
        self._entries[domain_id] = entry
        self._save_entry(entry)
        self._prune_if_full()
        return entry

    def bump_version(
        self,
        domain_id: str,
        *,
        decoder_src_path: str | Path,
        energy_src_path: str | Path,
        energy_kind: str,
        conformal_src_path: Optional[str | Path] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> DomainEntry:
        entry = self._entries[domain_id]
        new_version = max(v.version for v in entry.versions) + 1
        cv = self._stash_artifacts(
            domain_id, new_version,
            decoder_src_path=decoder_src_path,
            energy_src_path=energy_src_path,
            energy_kind=energy_kind,
            conformal_src_path=conformal_src_path,
            provenance=provenance,
        )
        entry.versions.append(cv)
        entry.current_version = new_version
        entry.last_used_at = _now_iso()
        self._save_entry(entry)
        return entry

    def rollback(self, domain_id: str, target_version: int) -> DomainEntry:
        entry = self._entries[domain_id]
        entry.get_version(target_version)        # validates existence
        entry.current_version = target_version
        entry.last_used_at = _now_iso()
        self._save_entry(entry)
        return entry

    def retire(self, domain_id: str) -> None:
        if domain_id in self._entries:
            entry = self._entries.pop(domain_id)
            self._archive_entry(entry)

    # ---- Growth control ---------------------------------------------------

    def _prune_if_full(self) -> list[str]:
        if len(self._entries) <= self.max_active:
            return []
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.last_used_at,
        )
        n_to_retire = len(self._entries) - self.max_active
        retired: list[str] = []
        for entry in sorted_entries[:n_to_retire]:
            self.retire(entry.domain_id)
            retired.append(entry.domain_id)
        return retired

    # ---- Internals --------------------------------------------------------

    def _stash_artifacts(
        self,
        domain_id: str, version: int,
        *,
        decoder_src_path: str | Path,
        energy_src_path: str | Path,
        energy_kind: str,
        conformal_src_path: Optional[str | Path] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> DomainVersion:
        """Copy artifact files into artifacts/<domain_id>__v<N>/ and
        return the DomainVersion record."""
        import shutil
        artifact_dir = self.root / "artifacts" / f"{domain_id}__v{version}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        # Decoder
        decoder_dst = artifact_dir / "decoder.pt"
        shutil.copyfile(decoder_src_path, decoder_dst)
        # Energy
        energy_filename = f"energy_{energy_kind.lower()}.pt"
        energy_dst = artifact_dir / energy_filename
        shutil.copyfile(energy_src_path, energy_dst)
        # Conformal (optional)
        conformal_rel: Optional[str] = None
        if conformal_src_path is not None:
            conformal_dst = artifact_dir / "conformal.json"
            shutil.copyfile(conformal_src_path, conformal_dst)
            conformal_rel = str(conformal_dst.relative_to(self.root))
        # Provenance
        prov_dict = dict(provenance or {})
        prov_path = artifact_dir / "provenance.json"
        with open(prov_path, "w") as f:
            json.dump(prov_dict, f, indent=2)
        return DomainVersion(
            version=version,
            decoder_path=str(decoder_dst.relative_to(self.root)),
            energy_path=str(energy_dst.relative_to(self.root)),
            energy_kind=energy_kind,
            conformal_path=conformal_rel,
            provenance=prov_dict,
            created_at=_now_iso(),
        )


def _now_iso() -> str:
    """Microsecond-precision UTC ISO timestamp.

    Same rationale as Stage 1.5's ConceptRegistry: LRU pruning relies
    on last_used_at being a strict total order, so second-resolution
    ties make prune behavior input-order-dependent. Microsecond
    precision keeps successive calls strictly ordered.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
