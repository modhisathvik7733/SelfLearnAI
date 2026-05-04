# SelfLearnAI — Runbook (MVP)

This is the practical "how do I actually train it on my GPU box" guide.
Architectural rationale lives in `/Users/chintu/.claude/plans/you-are-a-senior-jazzy-shannon.md`.

The MVP target: aligned 384-dim shared space across V-JEPA-2 + GTE + CLIP, plus the plurality concept end-to-end with the full metric battery.

---

## 1. GPU box setup (one-time)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download COCO 2017 (val + annotations is enough for MVP — ~6GB)
mkdir -p /data/coco/{images,annotations}
cd /data/coco/images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
cd /data/coco/annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

Adjust the `coco_root` field in `configs/stage1_alignment.yaml` to match.

---

## 2. Stage 0 — verify foundations load

```bash
# Light: GTE + CLIP only (~1GB download)
python3 scripts/test_foundations.py --device cuda

# Full: also pull V-JEPA-2 ViT-L (~1.5GB download, requires reasonable RAM)
python3 scripts/test_foundations.py --device cuda --vjepa
```

Pass criteria: each enabled encoder produces non-NaN, distinguishable embeddings of the documented dim.

---

## 3. Stage 1 — adapter alignment (~24-48 GPU-hours on 1 A100)

```bash
python3 scripts/stage1_train.py --config configs/stage1_alignment.yaml
```

Watch the log for:

- `loss` decreasing.
- `L_text` and `L_vis` (InfoNCE) decreasing toward zero.
- `L_vic_*` settling around 1.0–2.0 (anti-collapse pressure active but not blowing up).
- `anchor_cos` staying > 0.85 — if it drops, the star topology is breaking down.

Checkpoints save to `checkpoints/stage1/step_*.pt`. Final at `checkpoints/stage1/final.pt`.

### Stage 1 exit gates (plan section 2)

After Stage 1, run grounding-only metrics:

```bash
# (You'll need an eval-pairs TSV with caption \t image_path lines.)
python3 scripts/run_metrics.py \
    --stage1-ckpt checkpoints/stage1/final.pt \
    --stage2-ckpt /dev/null \
    --eval-pairs eval/coco_eval_subset.tsv \
    --device cuda
```

Must pass:
- `cross_modal_cosine` ≥ 0.5
- `retrieval_recall@5` ≥ random (i.e., `5 / N_eval_pairs`)
- `visual_perturbation` ≥ 0.05 (vision is responsive)
- `min_std[*]` ≥ 0.3 across all four adapters

If any fail: don't move to Stage 2. Investigate (see plan section 0b — known fragilities).

---

## 4. Optional Stage 1.5 — JEPA span loss (gated)

Only run if Stage 1 metrics pass. Out of scope for this MVP scaffold; sketch is in plan section 2.

---

## 5. Build Stage-2 image pairs from COCO

```bash
python3 scripts/build_image_pairs_coco.py \
    --coco-root /data/coco \
    --split val2017 \
    --out data/plurality/image_pairs.tsv \
    --pairs-per-noun 10
```

This writes ~200 (one_image, many_image, noun) triples for nouns that exist in COCO categories.

---

## 6. Stage 2 — plurality operator (~6-12 GPU-hours)

```bash
python3 scripts/stage2_train.py \
    --config configs/stage2_plurality.yaml \
    --data-dir data/plurality
```

Watch:

- `L_fwd` and `L_inv` (text-side MSE) decreasing toward zero.
- `L_xmodal` (visual-shift ↔ text-shift consistency) decreasing toward zero.

Final checkpoint: `checkpoints/stage2_plurality/final.pt`.

---

## 7. Full metric battery

```bash
python3 scripts/run_metrics.py \
    --stage1-ckpt checkpoints/stage1/final.pt \
    --stage2-ckpt checkpoints/stage2_plurality/final.pt \
    --concept plural \
    --data-dir data/plurality \
    --eval-pairs eval/coco_eval_subset.tsv \
    --device cuda
```

MVP success requires all `✓` on:

| Metric | Threshold |
|---|---|
| `cross_modal_cosine` | ≥ 0.5 |
| `min_std[*]` | ≥ 0.3 |
| `intra_direction_coherence` | ≥ 0.7 |
| `held_out_transfer` | ≥ 0.7 |
| `inversibility` | ≥ 0.7 |
| `cross_modal_direction` | ≥ 0.5 |

---

## 8. What to do if metrics fail

See plan section 0b (known fragilities). Quick guide:

| Symptom | Likely cause | First fix |
|---|---|---|
| `cross_modal_cosine` low but `loss` is low | "good cosine, bad semantics"; InfoNCE found a degenerate alignment | Increase hard-negative pool, recheck batch construction |
| `min_std` low | Embedding collapse | Increase VICReg weight; check for NaN gradients |
| `anchor_cos` drifts down | Star topology broken | Lower LR for `adapter_c` only, or freeze it after step 200 |
| `held_out_transfer` low | Operator memorized training distribution | Add cross-category training pairs |
| `cross_modal_direction` low | Text and vision operators decoupled | Increase `w_xmodal` in Stage 2 config |
| `visual_perturbation` low | Vision being ignored (modality shortcut) | Add stronger vision-side augmentation, increase batch's count variation |

Plan section 6c has the full diagnostic list.

---

## File-by-file map

```
selflearnai/
├── foundations/        Frozen encoder wrappers   (Stage 0)
├── adapters/           Trainable projection heads (Stage 1)
├── concepts/           Operator library          (Stage 2+)
├── grounding/          Cross-modal data + batching
├── reasoning/          Post-MVP scaffold (empty)
├── metrics/            Diagnostic battery
└── losses.py           Shared loss functions
```

The toy reference (`selflearn.py` at repo root) is **read-only** — it's the validated baseline that Stage 1+2 should preserve in spirit.
