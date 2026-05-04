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

## The "multi-axial floor" turned out to be encoder geometry, not operator capacity

Initially recorded as the project's only failure case (~0% on antonyms with
single-head). Option-3 multi-head sweep + cos→truth diagnostic (commit
`4e2f752`) re-diagnosed it:

- **cos(pred, truth) ≈ 0.86 across every architecture** (single-head, K=2,
  K=3, K=4, shared-residual, per-head-residual). All produce essentially the
  same representation — multi-head adds no signal here.
- Under FAIR pool (training targets dropped), **single-head reaches 0.722**
  on GTE-base, 0.611 on E5. Not the ~0% originally feared.
- The 2/6 persistent failures (`warm→cool`, `calm→angry`) lose retrieval to
  *encoder neighbors* of the truth, not to the operator's wrong direction.
  The operator points to the right antonym region; the encoder's
  neighborhood puts a closer distractor in the way.

So the "single direction can't span disjoint axes" hypothesis was wrong: a
single direction + residual MLP DOES span all 6 axes (cos 0.86 to each).
The real limit is encoder topology for soft/ambiguous antonyms (warm, calm).

Open question for later: contrastive antonym fine-tune of the encoder. Not
on the immediate path — the system already covers every concept type at
high quality once retrieval design is fair.

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

## Option-3 result (multi-head sweep) — done

Run: `python scripts/multi_head_opposite.py --encoder {gte-base|e5-large-v2}
[--fair-pool] [--show-routing]`. Results above. Multi-head adds no signal
on antonyms once you control for pool design — diagnostic showed cos→truth
is invariant to architecture.

## What comes after Option 3

Per RESULTS.md "Future Work":
- Cross-modal grounding for the image-grounded pathway end-to-end.
- Latent-space planning (operator chains as search actions).
- A Layer-2 expression module bolted on top — explicitly out of scope until
  Layer 1 is fully solid.
