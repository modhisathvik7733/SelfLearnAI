# SelfLearnAI — Project Progress Log

One-page status. For the research-paper write-up see [RESULTS.md](RESULTS.md);
for what to run see [RUNBOOK.md](RUNBOOK.md). This file is a context-survival
log — what we built, what we learned, where we are, what's next.

## What this project is

A concept-learning system that operates **entirely in embedding space**, with
**no next-token prediction at any layer**. Concepts are modeled as
direction-vectors in a frozen text encoder's latent space (GTE-base / E5-large).
A concept operator is `forward(z) = z + α·v + residual_MLP([z; v])` —
~150K parameters per concept, trained on as few as N=3 pairs. Inference is
nearest-neighbor lookup over a candidate vocabulary.

## What we validated (headline numbers)

| Claim | Result |
|---|---|
| Concepts learnable from N=3 examples | **1.000 held-out transfer** on 5/6 concept types (plural, past, comparative, agentive, superlative) |
| Independently-trained operators compose | **6/6** chained accuracy (agentive ∘ plural) in FAIR pool |
| Operators are nearly-linear | `cos(linear-chain, MLP-chain) = 0.983` |
| Operators are individually invertible | `cos(inverse(forward(z)), z) = 0.990` |
| Standard NLP analogy benchmark | **13/15 (0.867)** across 5 families with N=3 each |
| End-to-end runtime | ~25 min on one RTX 5090, ~$0.20 GPU |

## Architectural pivots learned the hard way

1. **Two-pathway architecture.** Stage-1 alignment (V-JEPA-2 + GTE → 384-dim
   shared space) helps grounded vision concepts but *hurts* text-only concepts
   (past tense 0.833 → 1.000 when we removed Stage 1). Conclusion: route each
   concept through the pathway where its grounding lives. Image-grounded =
   Stage 1; pure-text = raw encoder.
2. **The 0.500 ceiling on cross-category-preserving concepts** (young-animal:
   horse→foal vs cow→calf) was *not* an architectural limit. It was a
   compression artifact (Stage-1 384-dim) + GTE capacity ceiling combined.
   Raw E5-large (no Stage 1) recovers 5/6.
3. **HARSH vs FAIR pool design** matters for compositionality. Composition
   targets (`girls`, `sisters`) sit at cos 0.86–0.91 to the truth embedding,
   but lose retrieval to single-operator distractors (`girl`, `sister`) by
   tiny margins. Composition works in embedding space — retrieval design
   determines whether you see it.
4. **Star topology + freeze CLIP anchor** prevented the multi-loss collapse
   that destroys most multimodal training runs.
5. **Precompute & cache foundation features** to disk (fp16). Stage-1 step
   time went from ~3s (PIL bottlenecked) to ~50ms.

## The one architectural floor we can still see

**Multi-axial concepts** — antonyms (big↔small, hot↔cold, true↔false, …) —
score ~0% with a single-head operator. A single direction-vector cannot span
disjoint semantic axes. This is the *one* failure case in the entire concept
library, and the reason Option 3 below exists.

## Current state of the repo

- `selflearn.py` — frozen toy reference (Ladder 2 baseline).
- `selflearnai/` — production package: foundations (frozen GTE / CLIP /
  V-JEPA-2 wrappers), adapters (Stage-1 trainer with star-topology +
  hard-negative bank), concepts (single-head + multi-head operator,
  ConceptLibrary).
- `data/{plurality, past_tense, comparative, opposite}/` — full-data concepts.
- `data/few_shot/{agentive, superlative, young}/` — N=3 few-shot concepts.
- `scripts/` — experiments, including run_text_only_concepts.py
  (concept × encoder matrix), test_compositionality.py (FAIR/HARSH pools),
  analogy_demo.py (5 standard analogy families), multi_head_young.py
  (architectural sweep).
- `checkpoints/stage1/` — trained adapters (GPU-side; not in repo).

## Right now — Option 3: break the multi-axial limit

Hypothesis: **K direction vectors + a router lets a single operator cover
disjoint antonym axes**. Each head specializes (size, temperature, truth,
emotion, …); the router sends each input to the right head.

Two variants to test:
- `MultiHeadConceptOperator` — K direction-vectors, K alphas, **shared**
  residual MLP. Cheap; already implemented.
- `MultiHeadConceptOperatorPerHead` — K direction-vectors, K alphas,
  **per-head** residual MLPs. More capacity for axis-specific non-linearities.
  *Add this next.*

Sweep on `data/opposite/`:
- K ∈ {1, 2, 3, 4} × {shared-residual, per-head-residual} × 3 seeds.
- Held-out 6 antonym pairs across spatial / temperature / quality /
  emotion / abstract / epistemic axes.

If multi-head clears 0.5: routing solves multi-axiality, and the system
covers every concept type tested. If it stays at ~0: the floor is in the
encoder's antonym geometry, and the next move is encoder-side (E5 / a
contrastive antonym-aware embedding).

## What comes after Option 3

Per RESULTS.md "Future Work":
- Cross-modal grounding for the image-grounded pathway end-to-end.
- Latent-space planning (operator chains as search actions).
- A Layer-2 expression module bolted on top — explicitly out of scope until
  Layer 1 is fully solid.
