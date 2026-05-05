# Stage 3 — Universal Domain Ingestion: Closing Report

**Status: CLOSED.** All seven sub-tasks PASS on `e5-large-v2`. Closing-regression
runner (`scripts/stage3_regression.py`) green. Universal-pipeline thesis
empirically validated; cross-domain operator composition validated *with the
research-backed factored output refinement*.

## Headline Result

| Sub-task | Commit | Result |
|---|---|---|
| 3.1 cross-domain validation (definitional) | `c8c75cc` | PASS — median cos 1.0, grammar 100%, word-fidelity 90% (108/120) |
| 3.2 per-domain energy model              | `d45c40a` | PASS — both Gaussian + MLP at ROC-AUC 1.0000 |
| 3.3 versioned domain registry            | `28a57df` | PASS — 5/5 sub-cases (register, LRU, version-bump, rollback, persistence) |
| 3.4 universal ingestion orchestrator     | `253f7cb` | PASS — 5/5 plumbing checks; bonus quality on temporal (median cos 1.0, 30/30 grammar) |
| **3.5f** factored output composition     | **`e565efd`** | **PASS — 6/8 (75%) on truly-novel after FOUR previous 0/8 attempts** |
| 3.6 per-domain conformal calibration     | `5a68d9b` | PASS — ECE 0.0694 < 0.07 |
| 3.7 closing regression runner            | (this commit) | PASS — quick mode green, chained Stage 1.5 green |

## What Stage 3 Buys for Everything That Follows

- **Universal pipeline validated**: the single function `ingest_domain(...)`
  takes a domain spec (train + holdout sentences, src/tgt word pairs) and
  produces (decoder, energy model, conformal calibration, registry entry)
  in one round-trip call. Empirically demonstrated on TWO new domains
  beyond Phase 2a's morphological four (definitional in 3.1, temporal in
  3.4). Same architecture, same recipe, both hit Phase 2a-class numbers.
- **Domain-level out-of-domain refusal**: `E_domain(ψ)` energy models
  cleanly distinguish on-manifold from off-manifold inputs (ROC-AUC 1.000
  on definitional vs Phase 2a's 432 morphological holdouts). This is the
  refusal mechanism the planner can use at inference instead of
  hand-tuned cosine thresholds.
- **Cross-domain composition WITH refinement**: the naive
  δ-broadcast-through-vocab-head approach for `definitional ∘ plural`
  fails empirically (4 attempts, all 0/8). The *factored output* fix
  (research-recommended) works: stem decoder + tiny δ-classifier +
  per-domain rule-based morphology hits 6/8 on truly-novel subjects.
  Universal-pipeline thesis survives WITH the architectural lesson
  filed as memory for all future Stage 3 domains.
- **Per-domain conformal calibration**: each registered domain now has
  a calibrated coverage set (ECE < 0.07). The decoder's "is this output
  fidelity-OK?" decision is principled, not threshold-tuned.

Cost: ~10 GPU-hours of wall-clock on the Stage 3 build (well within the
plan §3 envelope, which budgeted 2-3 months). The rapid pace was bought
by Phase 2a having already validated the decoder architecture.

## The 3.5 Empirical Journey (where the real work happened)

Cross-domain operator transfer was the architecturally-loaded sub-task.
Six attempts, three architectural lessons:

| Attempt | Approach | Result | Lesson |
|---|---|---|---|
| 3.5  | Word-pair operator + δ-broadcast | 0/8 | **Operators must be trained in their destination encoder manifold.** Word-ψ and sentence-ψ are different geometries even on the same encoder. |
| 3.5b | Sentence-pair operator + δ-broadcast | 0/8 (ψ-lift now positive!) | ψ-space lift fixed (+0.0137); decoder doesn't follow the ψ-shift to surface form. |
| 3.5c | + decoder co-trained on (h_perturbed → plur_target) | 0/8 (stream C nll → 0; decoder MEMORIZES) | Decoder memorizes specific training transforms; doesn't generalize the rule. |
| 3.5d | Diagnostic — oracle δ on novel subjects | 0/8 even with oracle δ | **The decoder genuinely did not generalize the morphological rule** — even ground-truth δ on novel inputs produces no plural surface form. |
| 3.5e | Scale to 150 subjects (5×) | 0/8 (still!) | **Data-starved hypothesis FALSIFIED.** Architectural ceiling is real, not a corpus-size issue. |
| **3.5f** | **Factored output (research recommendation)** | **6/8 PASS** | **Factor WHAT (stem) from HOW (transformation).** Pointer-Generator copies novel subjects fine at the stem level; tag classifier learns the trivial `δ → {NONE, PL}` problem; rule-based morphology applies. |

### What the failure pattern actually was

Pointer-Generator with `p_gen ≈ 0.91` (vocab-branch dominant) trained on
identity reconstruction memorizes specific input-token → output-token
mappings (apple→apples, banana→bananas). It does NOT learn the
morphological rule "apply +s to whatever subject token is in the noun
position." For novel subjects (mango, peach, helicopter), the vocab head
has no learned plural-token mapping → it emits SEP/PAD or stays
singular. The pointer can't help because suffix tokens (`##s`, `##es`)
aren't in the encoder input.

Research (Patel & Bhattamishra ACL 2022, SIGMORPHON 2022, Subramani et
al. ACL Findings 2022) documents this as a known failure mode of
non-AR decoders trained on identity reconstruction. The standard fix is
factored output: separate the stem (which the existing architecture is
good at) from the transformation (which requires a different mechanism
than ψ-shift broadcast through a vocab head).

### Why the factored fix preserves the architecture

The fix does NOT abandon the universal-pipeline claim. It makes the
claim more precise:

- **Stem** (the "what to say"): produced by the per-domain decoder via
  pointer-copy, which already handles novel subjects fine.
- **Tag** (the "in what mode"): predicted by a tiny ~65K-param
  classifier on the operator's δ vector. The classifier learns a binary
  problem in δ-space — much smaller than the vocab head's
  150-distinct-plural-token problem.
- **Morphology** (the "how to inflect"): a deterministic per-domain
  rule function (regex + suffix tables, ~20 LOC for English plurals).
  This IS the per-domain "structural backbone" that plan §13.9
  explicitly anticipated for syntactically-strict transformations.

Cost: NO retraining of the 34M-param decoder. Tag classifier trains in
seconds. Morphology rule is hand-authored once per (domain,
transformation) pair.

The architectural memory is filed at
`feedback_factored_output_for_cross_domain.md` for all future Stage 3+
domains.

## What Stage 3 Did NOT Solve (deferred to v2/CSIL)

- **Multi-subword copy** (`bookcase` → `[book, ##case]`, only first
  piece copied). Documented limitation from Phase 2a §19.14, surfaced
  again in 3.5f's "bookcase" failure. Span-copy mechanism is the fix;
  v2 work.
- **Decoder hallucination on edge inputs** ("cabbage" produced "cabbage
  is cabbage vegetable" in 3.5f). Pre-existing 3.1 decoder issue,
  independent of the factored architecture. More training data per
  domain or per-position attention regularization would help.
- **Energy model AUC at smaller-domain scales**: 3.4's temporal domain
  hit ROC-AUC 0.9924 on 144 train ψ — close to but below the 0.95
  target. Acceptable at this scale; might need more ψ at < 100 train.
- **Cross-conformal (CV+) for distribution-shift mitigation**: 3.6's
  ECE 0.0694 sits 1% under the 0.07 gate but the score distribution is
  very peaked (q_hat saturates at -1.0 for α≥0.20). Same issue as the
  deferred 0.5.2.5 task. CV+ is the fix when needed.

## Decisions Carried Forward

- **Macro persistence** (Stage 1.5 follow-up): still untriggered.
- **Cross-conformal for `comparative` and `past_tense`** (0.5.2.5):
  still untriggered.
- **Value function `V(ψ, ψ_goal)`** (Task 1.9): still deferred.
- **Span-copy for multi-subword** (Phase 2a §19.14): still deferred,
  now with two pieces of evidence (comparative in 2a, bookcase in 3.5f).

## Recurring Architectural Patterns Observed

Three patterns seen across Stages 0.5 / 1 / 1.5 / 3 are now memory-filed:

1. **Encoder-space gates are RELATIVE, not absolute** — feedback_relative_gates.
   Saw this AGAIN in 3.5b (cos-direction gate with absolute thresholds)
   and 3.5d (the operated-vs-baseline "lift" measure replaced
   "operated > 0.85" absolute).
2. **ΔΨ clusters: operator-consistency is the HARD gate, not silhouette**
   — feedback_psi_residuals_use_operator_consistency. Stage 3 did not
   re-encounter this directly but the principle applies to Stage 3.4's
   energy-model probe selection.
3. **Cross-domain composition: factored output, not δ-broadcast** —
   NEW in 3.5f, the freshest entry.

## What Plan §19 Looks Like After Stage 3

The plan's locked architecture for v1 (Stages 0–4) survives in full.
Stage 3 closed cleanly; **Stage 4 (PsiNet-Refactor-v1 benchmark) is the
remaining v1 deliverable**.

Stage 4 anticipated work: ingest Python via the Stage 3 universal pipeline
(no tree-sitter), train per-task operators (rename, extract function,
loop→comprehension, etc.) on small corpora, run benchmark §10. The
factored-output lesson from 3.5f says: the per-task transformation
operators will use the same factor-WHAT-from-HOW pattern. Stem is
copied/generated by the Python decoder; transformation tag fires the
deterministic AST-aware rule. This re-imports a *little* per-language
scaffolding for transformation rules, but the universal *pipeline*
(encode → ψ → operator → decoder → energy → registry) stays universal.
That's the plan §13.9 fallback, applied surgically per transformation
rather than per language.

## How to Reproduce

```bash
# Closing regression (quick mode, ~20-30 min):
python scripts/stage3_regression.py

# Full retraining regression (~6-7 hr — for major milestones):
python scripts/stage3_regression.py --full

# Individual sub-tasks (in execution order):
python scripts/stage3_1_definitional.py        # 3.1 — ~4 hr
python scripts/stage3_2_energy_model.py        # 3.2 — ~1 hr
python scripts/stage3_3_registry_smoke.py      # 3.3 — ~5 sec
python scripts/stage3_4_orchestrator_smoke.py  # 3.4 — ~30-45 min
python scripts/stage3_5f_factored_output.py    # 3.5f — ~5 min
python scripts/stage3_6_conformal.py           # 3.6 — ~5-10 min
```

## Sources / Research Anchored

The research agent's `2026-05` literature pull anchored the 3.5 architecture
decision (commits 60bd34c through e565efd):

- Patel & Bhattamishra, *Revisiting Compositional Generalization Abilities*
  (ACL 2022) — the ~300-primitives threshold for vocab-vs-rule learning.
- SIGMORPHON 2022 inflection shared task baselines + Wu/Cotterell/Hulden
  — character-level / factored output as the SOTA for unseen-lemma
  morphology generalization.
- Subramani et al., *Extracting Latent Steering Vectors from Pretrained
  LMs* (ACL Findings 2022) — vector-arithmetic precedent in latent
  decoder conditioning.
- "Don't Copy That!" (arXiv 2403.10963, 2024) — confirms PGN failure on
  morphology, NOT a balance/tuning problem.

The full ranked list with URLs is in the conversation log; the
architectural lesson is filed at
`feedback_factored_output_for_cross_domain.md`.
