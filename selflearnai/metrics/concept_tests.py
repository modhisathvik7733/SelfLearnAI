"""Concept-emergence tests — does the operator learn a real concept,
or memorize per-noun mappings?

Locked metrics from plan section 6b:
  • Intra-direction coherence > 0.7  (per-pair shifts cluster around mean).
  • Held-out transfer ≥ 70% accuracy (operator generalizes beyond training).
  • Inversibility > 0.7 cosine on held-out singulars.
  • Pure-translation reduction (how much is just `z + v_const`).
  • Cross-modal direction cosine > 0.5 (text shift ≈ vision shift).

Plus the deepest test: held-out SEMANTIC CATEGORY transfer. Train on animals,
test on objects. If the operator only works within its training distribution,
it's pattern memorization, not concept learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from selflearnai.adapters import AdapterBundle
from selflearnai.concepts import ConceptOperator, InverseConceptOperator


@torch.no_grad()
def intra_direction_coherence(
    z_sing: torch.Tensor,         # (N, D)
    z_plur: torch.Tensor,         # (N, D)
) -> float:
    """Mean cosine between each individual (plur - sing) and the mean shift.
    Threshold per plan: > 0.7."""
    diffs = z_plur - z_sing
    v_mean = diffs.mean(dim=0, keepdim=True)
    return F.cosine_similarity(diffs, v_mean.expand_as(diffs), dim=-1).mean().item()


@torch.no_grad()
def held_out_text_transfer(
    fwd: ConceptOperator,
    bundle: AdapterBundle,
    gte,
    held_out_pairs: list[tuple[str, str]],     # (sing_text, plur_text)
    candidate_pool: list[str],                  # at least the held-out plurals + many distractors
) -> float:
    """For each held-out singular: forward(emb(sing)) → nearest candidate.
    Threshold per plan: ≥ 70%.
    """
    sing_texts = [p[0] for p in held_out_pairs]
    plur_texts = [p[1] for p in held_out_pairs]
    z_sing = bundle.adapter_t(gte.encode(sing_texts))
    z_pred = fwd(z_sing)

    z_pool = bundle.adapter_t(gte.encode(candidate_pool))
    sims = F.cosine_similarity(
        z_pred.unsqueeze(1).expand(-1, len(candidate_pool), -1),
        z_pool.unsqueeze(0).expand(len(sing_texts), -1, -1),
        dim=-1,
    )
    best_idx = sims.argmax(dim=-1)
    best_words = [candidate_pool[i] for i in best_idx.tolist()]
    correct = sum(b == p for b, p in zip(best_words, plur_texts))
    return correct / len(held_out_pairs)


@torch.no_grad()
def inversibility(
    fwd: ConceptOperator,
    inv: InverseConceptOperator,
    bundle: AdapterBundle,
    gte,
    held_out_singulars: list[str],
) -> float:
    """cos(z, inverse(forward(z))) averaged over held-out singulars.
    Threshold per plan: > 0.7.
    """
    z = bundle.adapter_t(gte.encode(held_out_singulars))
    z_round_trip = inv(fwd(z))
    return F.cosine_similarity(z, z_round_trip, dim=-1).mean().item()


@torch.no_grad()
def pure_translation_score(
    bundle: AdapterBundle,
    gte,
    train_pairs: list[tuple[str, str]],
    held_out_pairs: list[tuple[str, str]],
    candidate_pool: list[str],
) -> float:
    """How well does `z + v_const` (no MLP) recover plurals?

    v_const = mean of (plur - sing) over training pairs.
    Apply z_sing + v_const for held-out singulars, lookup nearest candidate.
    Reports the linear-fraction of plurality structure.
    """
    train_sing = bundle.adapter_t(gte.encode([p[0] for p in train_pairs]))
    train_plur = bundle.adapter_t(gte.encode([p[1] for p in train_pairs]))
    v_const = (train_plur - train_sing).mean(dim=0)

    held_sing = bundle.adapter_t(gte.encode([p[0] for p in held_out_pairs]))
    held_plur_target = [p[1] for p in held_out_pairs]
    pred = held_sing + v_const

    z_pool = bundle.adapter_t(gte.encode(candidate_pool))
    sims = F.cosine_similarity(
        pred.unsqueeze(1).expand(-1, len(candidate_pool), -1),
        z_pool.unsqueeze(0).expand(len(held_out_pairs), -1, -1),
        dim=-1,
    )
    best_idx = sims.argmax(dim=-1)
    best_words = [candidate_pool[i] for i in best_idx.tolist()]
    correct = sum(b == p for b, p in zip(best_words, held_plur_target))
    return correct / len(held_out_pairs)


@torch.no_grad()
def cross_modal_direction_cosine(
    bundle: AdapterBundle,
    gte, vjepa,
    text_pairs: list[tuple[str, str]],          # (sing_text, plur_text)
    image_pairs: list[tuple[Path, Path]],       # (one_img_path, many_img_path)
) -> float:
    """cos(v_text, v_visual) — the operational definition of grounding.
    Threshold per plan: > 0.5.
    """
    z_sing = bundle.adapter_t(gte.encode([p[0] for p in text_pairs]))
    z_plur = bundle.adapter_t(gte.encode([p[1] for p in text_pairs]))
    v_text = (z_plur - z_sing).mean(dim=0)

    one_imgs  = [Image.open(p[0]).convert("RGB") for p in image_pairs]
    many_imgs = [Image.open(p[1]).convert("RGB") for p in image_pairs]
    z_one  = bundle.adapter_v(vjepa.encode_patches(one_imgs))
    z_many = bundle.adapter_v(vjepa.encode_patches(many_imgs))
    v_vis = (z_many - z_one).mean(dim=0)

    return F.cosine_similarity(v_text.unsqueeze(0), v_vis.unsqueeze(0)).item()


@torch.no_grad()
def cross_category_held_out(
    fwd: ConceptOperator,
    bundle: AdapterBundle,
    gte,
    pairs_in_category: list[tuple[str, str]],     # e.g. animals
    pairs_other_category: list[tuple[str, str]],  # e.g. objects
    candidate_pool: list[str],
) -> dict:
    """The deepest concept test. Train on one category, test on another.
    If accuracy collapses on the other category, the 'concept' was a
    per-category memorization, not a general operator.
    """
    in_cat_score = held_out_text_transfer(
        fwd, bundle, gte, pairs_in_category, candidate_pool,
    )
    other_cat_score = held_out_text_transfer(
        fwd, bundle, gte, pairs_other_category, candidate_pool,
    )
    return {
        "in_category":     in_cat_score,
        "out_of_category": other_cat_score,
        "transfer_gap":    in_cat_score - other_cat_score,
    }


@dataclass
class ConceptReport:
    """Summary of one concept's metrics. Print or assert against thresholds."""
    name: str
    coherence: float
    held_out_transfer: float
    inversibility: float
    pure_translation: float
    cross_modal_cosine: float
    cross_category_gap: float | None = None


def report_concept(
    name: str,
    fwd: ConceptOperator,
    inv: InverseConceptOperator,
    bundle: AdapterBundle,
    gte, vjepa,
    train_pairs: list[tuple[str, str]],
    held_out_pairs: list[tuple[str, str]],
    candidate_pool: list[str],
    image_pairs: list[tuple[Path, Path]] | None = None,
    cross_category_pairs: list[tuple[str, str]] | None = None,
) -> ConceptReport:
    """Run the metric battery on a single concept; return a ConceptReport."""
    z_train_sing = bundle.adapter_t(gte.encode([p[0] for p in train_pairs]))
    z_train_plur = bundle.adapter_t(gte.encode([p[1] for p in train_pairs]))

    coherence = intra_direction_coherence(z_train_sing, z_train_plur)
    held_out = held_out_text_transfer(fwd, bundle, gte, held_out_pairs, candidate_pool)
    inv_score = inversibility(
        fwd, inv, bundle, gte, [p[0] for p in held_out_pairs],
    )
    pure_trans = pure_translation_score(
        bundle, gte, train_pairs, held_out_pairs, candidate_pool,
    )

    if image_pairs is not None and len(image_pairs) > 0:
        xmodal = cross_modal_direction_cosine(
            bundle, gte, vjepa, train_pairs, image_pairs,
        )
    else:
        xmodal = float("nan")

    cross_cat_gap = None
    if cross_category_pairs is not None:
        gap = cross_category_held_out(
            fwd, bundle, gte, held_out_pairs, cross_category_pairs, candidate_pool,
        )
        cross_cat_gap = gap["transfer_gap"]

    return ConceptReport(
        name=name,
        coherence=coherence,
        held_out_transfer=held_out,
        inversibility=inv_score,
        pure_translation=pure_trans,
        cross_modal_cosine=xmodal,
        cross_category_gap=cross_cat_gap,
    )
