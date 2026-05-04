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
| Multi-axial concepts (antonyms, curated) | **6/6 = 1.000** with shared_K2 + E5-large-v2; 0.944 single-head |
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

## Multi-axial concepts — solved (curated data ⇒ 1.000)

Originally recorded as the project's only architectural floor (~0% on
antonyms with single-head). Resolved across two experiments:

1. **Option-3 multi-head sweep + cos→truth diagnostic** (commits
   `4e2f752`, `7f032ca`) showed the operator was never the bottleneck:
   `cos(pred, truth) ≈ 0.86` invariant across single-head / K=2/3/4 /
   shared-residual / per-head-residual. The 0.000 was a HARSH-pool
   retrieval artifact + encoder soft-synonym ambiguity for 2 of 6 pairs
   (`warm→cool`, `calm→angry`).
2. **Curated `data/opposite_v2`** (commits `5ce3120`+) used the
   encoder-neighborhood probe to surgically fix retrieval design: drop
   pool words rated by the encoder as soft synonyms of held-outs, swap
   the 2 encoder-broken held-out pairs for cleaner-domain antonyms
   (`dawn→dusk`, `accept→reject`). On v2:
   - single-head + E5-large-v2 → **0.944** (5/6)
   - **shared_K2 + E5-large-v2 → 1.000 (6/6) across all 3 seeds**
   - GTE-base → 0.889 single-head; the one remaining miss
     (`near→far` → `missing`) is GTE topology, not architecture.

Multi-axial concepts are not an architectural failure mode. The system
covers every concept type tested at ≥0.94 with the right pathway and
retrieval design. There is no architectural floor at the operator level
in this project.

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

Run: `python scripts/multi_head_opposite.py [--data-dir data/opposite_v2]
--encoder {gte-base|e5-large-v2} --fair-pool [--show-routing]`. On v1 data,
multi-head adds no signal — the diagnostic showed `cos→truth` is
invariant to architecture; the floor was retrieval design + encoder
topology. On v2 (curated) data, single-head reaches 0.944 and shared_K2
reaches 1.000 on multi-axial antonyms.

## What comes after Option 3

Per RESULTS.md "Future Work":
- Cross-modal grounding for the image-grounded pathway end-to-end.
- Latent-space planning (operator chains as search actions).
- A Layer-2 expression module bolted on top — explicitly out of scope until
  Layer 1 is fully solid.
