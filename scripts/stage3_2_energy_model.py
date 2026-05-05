"""Stage 3 / Sub-task 3.2 — per-domain energy model.

Trains energy models for the definitional domain (Stage 3.1) and
evaluates ROC-AUC on a held-out in/out-of-domain probe set.

In-domain (positive class for "is in-domain"):
  Stage 3.1's 120 truly-novel held-out sentences. The model never
  saw these during energy-model training but they ARE definitional.

Out-of-domain (the test the energy model must reject):
  Phase 2a's 432 truly-novel held-out sentences (plural / past_tense /
  comparative / opposite). The energy model has NEVER seen these or
  even the Phase 2a domain. If the model truly learned the
  definitional manifold, these should score higher (more anomalous).

Energy-model training data: Stage 3.1's 500 training sentences only.

We train and compare TWO models:
  1. GaussianEnergy — Mahalanobis distance to in-domain mean/cov.
     Zero training, closed-form. Strong baseline.
  2. MLPEnergyModel — small MLP (~263K params) trained with NCE-style
     contrastive loss (positives = in-domain ψ, negatives = Gaussian
     noise matched to in-domain stats).

Acceptance gate (HARD per plan §19.17):
  At least one model achieves ROC-AUC ≥ 0.95.

If ROC-AUC ≥ 0.95: 3.2 closes; proceed to 3.3 (domain registry) and
3.4 (orchestrator) which use the trained energy models.

If ROC-AUC < 0.95: investigate. Possibilities: (a) probe set is too
similar to training (Phase 2a + definitional both English templated),
(b) MLP needs different negative sampling, (c) Mahalanobis isn't
expressive enough for this geometry. Diagnose before scaling.

Run on the GPU box (~1 hr — encoder forward dominates):
  python scripts/stage3_2_energy_model.py
  python scripts/stage3_2_energy_model.py --mlp-steps 5000  # longer MLP train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from selflearnai.domain import (
    GaussianEnergy,
    MLPEnergyModel,
    roc_auc,
)
from selflearnai.generator import read_corpus_tsv

from scripts.stage1_planner_beam_smoke import ENCODERS


@torch.no_grad()
def encode_pooled(sents, tok, mdl, device, max_length=64):
    inputs = tok(sents, padding=True, truncation=True, max_length=max_length,
                 return_tensors="pt").to(device)
    out = mdl(**inputs).last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).float()
    return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def build_definitional_corpora() -> tuple[list[str], list[str]]:
    """Reconstruct Stage 3.1's training and held-out sentence lists.

    Mirrors scripts/stage3_1_definitional.py:TRAIN_PAIRS,
    TRULY_NOVEL_PAIRS, and TEMPLATES. Kept inline (rather than imported)
    so 3.2 doesn't depend on the script's global state.
    """
    train_pairs = [
        ("apple", "fruit"),    ("banana", "fruit"),    ("orange", "fruit"),
        ("grape", "fruit"),    ("lemon", "fruit"),
        ("carrot", "vegetable"),  ("spinach", "vegetable"),
        ("potato", "vegetable"),  ("broccoli", "vegetable"),
        ("lettuce", "vegetable"),
        ("tiger", "animal"),  ("dolphin", "animal"),  ("eagle", "animal"),
        ("rabbit", "animal"), ("snake", "animal"),
        ("chair", "furniture"),  ("table", "furniture"),  ("sofa", "furniture"),
        ("desk", "furniture"),   ("bed", "furniture"),
        ("piano", "instrument"), ("guitar", "instrument"), ("drum", "instrument"),
        ("violin", "instrument"), ("flute", "instrument"),
        ("red", "color"),  ("blue", "color"),  ("green", "color"),
        ("yellow", "color"),  ("purple", "color"),
        ("car", "vehicle"),  ("bus", "vehicle"),  ("plane", "vehicle"),
        ("train", "vehicle"),  ("bicycle", "vehicle"),
        ("house", "building"),    ("school", "building"),  ("hospital", "building"),
        ("library", "building"),  ("museum", "building"),
        ("tennis", "sport"),  ("soccer", "sport"),  ("basketball", "sport"),
        ("swimming", "sport"),  ("running", "sport"),
        ("rain", "weather"),  ("snow", "weather"),  ("wind", "weather"),
        ("sunshine", "weather"),  ("fog", "weather"),
    ]
    truly_novel = [
        ("mango", "fruit"),  ("peach", "fruit"),
        ("cabbage", "vegetable"),
        ("penguin", "animal"),
        ("bookcase", "furniture"),
        ("trumpet", "instrument"),
        ("white", "color"),  ("black", "color"),
        ("helicopter", "vehicle"),
        ("church", "building"),
        ("cricket", "sport"),
        ("thunder", "weather"),
    ]
    templates = [
        "{subject} is a {category}",
        "a {subject} is a {category}",
        "{subject} is a type of {category}",
        "{subject} is a kind of {category}",
        "we call {subject} a {category}",
        "the {subject} is a {category}",
        "{subject} belongs to {category}",
        "{subject} is an example of {category}",
        "every {subject} is a {category}",
        "{subject} is part of {category}",
    ]
    train_sents = [
        tpl.format(subject=s, category=c)
        for s, c in train_pairs for tpl in templates
    ]
    holdout_sents = [
        tpl.format(subject=s, category=c)
        for s, c in truly_novel for tpl in templates
    ]
    return train_sents, holdout_sents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # MLP energy training
    parser.add_argument("--mlp-steps", type=int, default=3000)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-margin", type=float, default=1.0)
    parser.add_argument("--mlp-hidden1", type=int, default=256)
    parser.add_argument("--mlp-hidden2", type=int, default=64)
    # Acceptance
    parser.add_argument("--auc-min", type=float, default=0.95)
    # Phase 2a holdout (out-of-domain probes)
    parser.add_argument("--phase2a-holdout",
                        default="data/explanations_v2/holdout.tsv",
                        help="Phase 2a's truly-novel holdout TSV — used as out-of-domain probes.")
    # Output
    parser.add_argument("--out", default="results/stage3/energy_definitional.json")
    parser.add_argument("--ckpt-gauss",
                        default="data/explanations_v2/checkpoints/energy_def_gauss.pt")
    parser.add_argument("--ckpt-mlp",
                        default="data/explanations_v2/checkpoints/energy_def_mlp.pt")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Stage 3 / Sub-task 3.2 — per-domain energy model (definitional)")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]

    # ---- Build corpora + encode --------------------------------------
    print("\n[1] Building probe sets")
    print("-" * 78)
    train_def, holdout_def = build_definitional_corpora()
    print(f"  in-domain train (definitional):    {len(train_def)} sentences")
    print(f"  in-domain holdout (definitional):  {len(holdout_def)} sentences")

    # Out-of-domain: Phase 2a's holdout
    phase2a_path = Path(args.phase2a_holdout)
    if not phase2a_path.exists():
        raise SystemExit(f"FATAL: Phase 2a holdout not found at {phase2a_path}")
    phase2a_rows = read_corpus_tsv(phase2a_path)
    phase2a_sents = [r.sentence for r in phase2a_rows]
    print(f"  out-of-domain probes (Phase 2a):   {len(phase2a_sents)} sentences")

    print("\n[2] Encoding all probe sets via E5")
    print("-" * 78)
    psi_train_in = encode_pooled(train_def, tok, mdl, args.device).cpu()
    psi_holdout_in = encode_pooled(holdout_def, tok, mdl, args.device).cpu()
    psi_holdout_out = encode_pooled(phase2a_sents, tok, mdl, args.device).cpu()
    print(f"  ψ_train_in:    {tuple(psi_train_in.shape)}")
    print(f"  ψ_holdout_in:  {tuple(psi_holdout_in.shape)}")
    print(f"  ψ_holdout_out: {tuple(psi_holdout_out.shape)}")

    # Build the unified probe set: 120 in-domain (label 0) + 432 out (label 1)
    n_in = psi_holdout_in.size(0)
    n_out = psi_holdout_out.size(0)
    psi_probe = torch.cat([psi_holdout_in, psi_holdout_out], dim=0)
    y_probe = torch.cat([
        torch.zeros(n_in, dtype=torch.long),       # in-domain
        torch.ones(n_out, dtype=torch.long),       # out-of-domain (positive class)
    ])
    print(f"  unified probe: {n_in + n_out} samples ({n_in} in-domain, {n_out} out-of-domain)")

    results = {}

    # ---- Gaussian baseline -------------------------------------------
    print("\n[3] GaussianEnergy (closed-form fit)")
    print("-" * 78)
    gauss = GaussianEnergy(dim=DIM)
    gauss_stats = gauss.fit(psi_train_in)
    print(f"  fit: n_train={gauss_stats['n_train']}, "
          f"diag_mean={gauss_stats['diag_mean']:.4f}, "
          f"ridge_used={gauss_stats['ridge_used']:.4e}")
    e_in_g = gauss.energy(psi_holdout_in)
    e_out_g = gauss.energy(psi_holdout_out)
    auc_g = roc_auc(y_probe, torch.cat([e_in_g, e_out_g]))
    print(f"  in-domain energy:   mean={float(e_in_g.mean()):.2f}  "
          f"std={float(e_in_g.std()):.2f}")
    print(f"  out-of-domain:      mean={float(e_out_g.mean()):.2f}  "
          f"std={float(e_out_g.std()):.2f}")
    print(f"  ROC-AUC: {auc_g:.4f}  (target ≥ {args.auc_min:.2f})")
    Path(args.ckpt_gauss).parent.mkdir(parents=True, exist_ok=True)
    gauss.save(args.ckpt_gauss)
    print(f"  saved → {args.ckpt_gauss}")
    results["gauss"] = {
        "auc": auc_g,
        "e_in_mean": float(e_in_g.mean().item()),
        "e_in_std": float(e_in_g.std().item()),
        "e_out_mean": float(e_out_g.mean().item()),
        "e_out_std": float(e_out_g.std().item()),
        "fit_stats": gauss_stats,
    }

    # ---- MLP energy model --------------------------------------------
    print("\n[4] MLPEnergyModel (NCE-style contrastive training)")
    print("-" * 78)
    mlp = MLPEnergyModel(
        dim=DIM, hidden1=args.mlp_hidden1, hidden2=args.mlp_hidden2,
    )
    mlp_stats = mlp.fit(
        psi_train_in,
        steps=args.mlp_steps,
        batch_size=args.mlp_batch_size,
        lr=args.mlp_lr,
        margin=args.mlp_margin,
        device=args.device,
        seed=args.seed,
    )
    print(f"  fit: {mlp_stats['n_params']/1e3:.1f}K params, "
          f"{mlp_stats['steps']} steps")
    print(f"  loss history (every 500 steps):")
    for h in mlp_stats["loss_history"]:
        print(f"    step {h['step']:>4}  loss={h['loss']:.4f}  "
              f"E_pos={h['E_pos_mean']:+.3f}  E_neg={h['E_neg_mean']:+.3f}")
    mlp.eval()
    mlp.cpu()
    with torch.no_grad():
        e_in_m = mlp.energy(psi_holdout_in)
        e_out_m = mlp.energy(psi_holdout_out)
    auc_m = roc_auc(y_probe, torch.cat([e_in_m, e_out_m]))
    print(f"  in-domain energy:   mean={float(e_in_m.mean()):+.3f}  "
          f"std={float(e_in_m.std()):.3f}")
    print(f"  out-of-domain:      mean={float(e_out_m.mean()):+.3f}  "
          f"std={float(e_out_m.std()):.3f}")
    print(f"  ROC-AUC: {auc_m:.4f}  (target ≥ {args.auc_min:.2f})")
    Path(args.ckpt_mlp).parent.mkdir(parents=True, exist_ok=True)
    mlp.save(args.ckpt_mlp)
    print(f"  saved → {args.ckpt_mlp}")
    results["mlp"] = {
        "auc": auc_m,
        "e_in_mean": float(e_in_m.mean().item()),
        "e_in_std": float(e_in_m.std().item()),
        "e_out_mean": float(e_out_m.mean().item()),
        "e_out_std": float(e_out_m.std().item()),
        "fit_stats": {k: v for k, v in mlp_stats.items() if k != "loss_history"},
    }

    # ---- Verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  GaussianEnergy   ROC-AUC: {auc_g:.4f}  "
          f"{'PASS' if auc_g >= args.auc_min else 'FAIL'}")
    print(f"  MLPEnergyModel   ROC-AUC: {auc_m:.4f}  "
          f"{'PASS' if auc_m >= args.auc_min else 'FAIL'}")

    best_auc = max(auc_g, auc_m)
    best_name = "MLPEnergyModel" if auc_m >= auc_g else "GaussianEnergy"

    if best_auc >= args.auc_min:
        verdict = "STAGE_3_2_PASS"
        message = (
            f"At least one energy model meets the ≥{args.auc_min} ROC-AUC "
            f"target. {best_name} wins at {best_auc:.4f}. The definitional-"
            f"domain energy model can reject Phase 2a out-of-domain inputs. "
            f"3.2 closes; proceed to 3.3 (domain registry) which will "
            f"persist this checkpoint alongside Stage 3.1's decoder."
        )
    elif best_auc >= 0.85:
        verdict = "STAGE_3_2_BELOW_TARGET"
        message = (
            f"Best ROC-AUC ({best_auc:.4f}) below the {args.auc_min} target "
            f"but well above chance (0.5). The definitional and Phase 2a "
            f"manifolds are partially separable but not cleanly so — "
            f"likely because both are short English sentences and share "
            f"surface statistics. Try (a) longer MLP training, "
            f"(b) hard-negative mining from Phase 2a during MLP fit, "
            f"(c) accept lower threshold for refusal at inference."
        )
    else:
        verdict = "STAGE_3_2_INSUFFICIENT"
        message = (
            f"Best ROC-AUC ({best_auc:.4f}) is too close to chance. The "
            f"energy models can't distinguish definitional from morphological "
            f"in encoder space at this scale. Investigate: maybe the encoder "
            f"itself doesn't separate these domains, in which case we'd "
            f"need a different signal (per-domain conformal sets, or "
            f"explicit domain classifier)."
        )

    print(f"\n  best: {best_name} @ ROC-AUC {best_auc:.4f}")
    print(f"\n→ {verdict}")
    print(f"\n{message}")

    payload = {
        "task": "3.2",
        "domain": "definitional",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "n_train_sents": len(train_def),
        "n_in_domain_holdout": n_in,
        "n_out_of_domain_probes": n_out,
        "thresholds": {"auc_min": args.auc_min},
        "results": results,
        "best_auc": best_auc,
        "best_model": best_name,
        "verdict": verdict,
        "message": message,
        "checkpoints": {
            "gaussian": args.ckpt_gauss,
            "mlp": args.ckpt_mlp,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if best_auc >= args.auc_min else 1)


if __name__ == "__main__":
    main()
