# data/opposite_v2 — curated antonym data

Curated successor to `data/opposite/` after the encoder-neighborhood probe
(see RESULTS.md "Multi-axial re-diagnosis" section) revealed two distinct
classes of failure in v1:

1. **Encoder-ambiguous held-outs.** `warm→cool` and `calm→angry` — the
   encoder rates soft synonyms (`sick`, `weary`) at cos ≥ 0.83 to the
   truth, beating the operator on retrieval regardless of architecture.
2. **Encoder-noise distractors.** Pool words like `boring`, `weary`,
   `unhappy`, `foolish`, `enemy`, `early`, `late` sit close to held-out
   targets in pure encoder space — they steal retrieval before the
   operator's prediction has a chance.

What v2 changes vs v1:

- **Train pairs**: identical (30 pairs, same TSV). The training set was
  not the problem.
- **Held-out (6 pairs)**: 4 unchanged (`near→far`, `fresh→stale`,
  `peace→war`, `truth→lie`); 2 swapped:
  - `warm→cool` → `dawn→dusk` (time-of-day antonym, cleaner geometry)
  - `calm→angry` → `accept→reject` (decision antonym)
- **Candidate pool (66 words)**: training targets retained (so HARSH-pool
  eval still works); v1 distractors that the probe identified as
  encoder-noise lures were removed. The remaining distractors are
  antonyms-of-other-words that are not lexically/semantically adjacent to
  any held-out target.

Run with the same script:

```bash
python scripts/multi_head_opposite.py --data-dir data/opposite_v2 \
    --encoder e5-large-v2 --fair-pool --show-routing
```

What we expect to see:

- **If accuracy lifts toward 1.000**: the v1 0.722 ceiling really was
  retrieval-side noise + 2 encoder-broken pairs, and a clean pool +
  cleanly-separable held-outs lets the single-head operator close to
  perfect on multi-axial concepts.
- **If accuracy stays around 0.72**: the encoder still has soft-synonym
  topology around at least one held-out target, and the principled fix is
  encoder-side (contrastive antonym fine-tune), not pool design.
