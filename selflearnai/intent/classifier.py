"""Tier-2 Ψ-space intent classifier.

Small frozen-encoder + tiny MLP that maps a NATURAL-LANGUAGE QUESTION to
one of the known concepts (or an explicit "unknown" class). Catches the
paraphrase tail Tier-1's regex grammar deliberately doesn't reach
(see plan §1, Gap 1; Task 1.3 measured Tier-1 paraphrase recall at ~37%).

Architectural commitments (per §0a / §1 of the plan):
  - The frozen encoder (GTE-base or E5-large-v2) is the SAME one
    operators use; we never fine-tune it.
  - The classifier MLP is small (~50–100K params); it learns intent
    boundaries in encoder space, not new representations.
  - Output includes an explicit "unknown" class so out-of-library
    questions are routed to refusal, not silently misclassified.
  - Calibrated confidence comes from Task 1.5 (conformal prediction
    on classifier scores); this module just outputs raw softmax.

Source-word extraction is NOT done here — it stays with Tier 1's regex
groups for matched questions; for paraphrases that bypass Tier 1, the
router (Task 1.6) wires up a separate source extractor before
forwarding to a concept operator. This module is concept-classification
only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class IntentClassMapping:
    """Maps integer class ids to concept names + the explicit unknown class.

    `unknown_label` MUST be one of `classes`. Convention: place it last.
    """
    classes: tuple[str, ...]
    unknown_label: str = "unknown"

    def __post_init__(self):
        if self.unknown_label not in self.classes:
            raise ValueError(
                f"unknown_label {self.unknown_label!r} must appear in classes "
                f"{self.classes}"
            )

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def unknown_index(self) -> int:
        return self.classes.index(self.unknown_label)

    def index(self, name: str) -> int:
        if name not in self.classes:
            raise ValueError(f"Unknown class name {name!r}; have {self.classes}")
        return self.classes.index(name)

    def name(self, idx: int) -> str:
        return self.classes[idx]


@dataclass
class IntentPrediction:
    """One prediction from the classifier."""
    concept: str           # the predicted class name (could be "unknown")
    confidence: float      # softmax probability of the predicted class
    is_unknown: bool       # True iff concept == mapping.unknown_label
    probs: dict            # full distribution {class_name: probability}
    raw_logits: list       # raw logit values (unnormalized), per class


class IntentClassifier(nn.Module):
    """Tiny MLP over a frozen encoder embedding → distribution over
    {known concepts, unknown}.

    Architecture: Linear(D_enc → hidden) → GELU → Linear(hidden → C).
    For D_enc = 768 (GTE-base), C = 8, hidden = 128 the model has
    768·128 + 128 + 128·8 + 8 ≈ 99K params.

    Independent of any specific encoder — caller passes an embedding
    of shape (B, D_enc); this module produces logits of shape (B, C).
    """

    def __init__(
        self,
        encoder_dim: int,
        mapping: IntentClassMapping,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if encoder_dim <= 0:
            raise ValueError(f"encoder_dim must be positive, got {encoder_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.mapping = mapping
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mapping.num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, encoder_dim) — frozen encoder output. Returns raw
        logits (B, num_classes)."""
        return self.net(z)

    @torch.no_grad()
    def predict_one(self, z: torch.Tensor) -> IntentPrediction:
        """Single-example prediction. z must be shape (encoder_dim,)
        or (1, encoder_dim)."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] != 1:
            raise ValueError(f"predict_one expects a single example; got batch {z.shape[0]}")
        logits = self(z).squeeze(0)
        probs = F.softmax(logits, dim=-1)
        argmax = int(probs.argmax().item())
        concept = self.mapping.name(argmax)
        confidence = float(probs[argmax].item())
        probs_dict = {
            self.mapping.name(i): float(probs[i].item())
            for i in range(self.mapping.num_classes)
        }
        return IntentPrediction(
            concept=concept,
            confidence=confidence,
            is_unknown=(concept == self.mapping.unknown_label),
            probs=probs_dict,
            raw_logits=[float(x) for x in logits.tolist()],
        )

    @torch.no_grad()
    def predict_batch(self, z: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """Batched prediction. Returns (predicted concept names, full
        probability matrix shape (B, C))."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        logits = self(z)
        probs = F.softmax(logits, dim=-1)
        argmax = probs.argmax(dim=-1).cpu().tolist()
        names = [self.mapping.name(i) for i in argmax]
        return names, probs

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
