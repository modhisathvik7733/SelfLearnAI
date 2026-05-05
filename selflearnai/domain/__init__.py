"""selflearnai.domain — Stage 3 universal domain ingestion machinery.

Per plan §19.17, Stage 3 builds the per-domain pieces that let the
system absorb new domains from docs + examples while preserving the
"reason in Ψ-space + express via Phase 2a decoder" architecture.

Module layout:

  energy.py       Per-domain energy/density model E_D(ψ) — distinguishes
                   on-manifold from off-manifold inputs at inference. Used
                   to refuse out-of-domain queries instead of hallucinating.
                   (Sub-task 3.2)
  registry.py     Versioned domain registry — extends Stage 1.5's
                   ConceptRegistry pattern to whole domains. Stores
                   (domain_id, version, decoder_ckpt, energy_ckpt,
                    conformal_calibration). (Sub-task 3.3)
  ingest.py       Universal ingestion orchestrator: docs + examples →
                   trained per-domain decoder + energy model + registry
                   entry. (Sub-task 3.4)
"""
from .energy import (
    EnergyModelBase,
    GaussianEnergy,
    MLPEnergyModel,
    roc_auc,
)

__all__ = [
    "EnergyModelBase",
    "GaussianEnergy",
    "MLPEnergyModel",
    "roc_auc",
]
