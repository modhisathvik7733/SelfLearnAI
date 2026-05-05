"""Stage 3 / Sub-task 3.4 — universal ingestion orchestrator smoke.

Round-trip test: feed a fresh domain (TEMPORAL — 'monday is a weekday',
'january is a month', etc.) through `ingest_domain()` and verify:

  1. Decoder + energy artifacts land in registry.root/artifacts/<id>__v1/
  2. Registry persists the new domain (round-trip via fresh instance)
  3. Decoder checkpoint reloads + does a forward pass
  4. Energy checkpoint reloads + scores in/out-of-domain ψ
  5. Provenance is JSON-roundtripped

Per plan §19.17 sub-task 3.4 acceptance: successful round-trip on a
fresh domain producing all expected artifacts.

Note: the GATES (cos / grammar / word-fidelity) are tracked but are
NOT the acceptance for 3.4. 3.4 tests pipeline plumbing, not absolute
quality. Quality on a 6-template tiny corpus with reduced training
steps (3000 vs 3.1's 10000) won't necessarily hit Stage 3.1's bar —
that's expected. The hard acceptance is "round-trip works".

CPU-only impossible (PointerSeqCondDecoder forward pass needs GPU
for any reasonable speed). ~30-45 min on a single GPU.

Run on the GPU box:
  python scripts/stage3_4_orchestrator_smoke.py
  python scripts/stage3_4_orchestrator_smoke.py --steps 5000  # closer to 3.1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from selflearnai.domain import (
    DomainRegistry,
    GaussianEnergy,
    IngestionConfig,
    MLPEnergyModel,
    ingest_domain,
)
from selflearnai.generator import PointerSeqCondDecoder, read_corpus_tsv

from scripts.stage1_planner_beam_smoke import ENCODERS


# ---------------------------------------------------------------------------
# Temporal domain — totally new from Phase 2a + Stage 3.1
# ---------------------------------------------------------------------------

TRAIN_PAIRS: list[tuple[str, str]] = [
    # weekday
    ("monday",    "weekday"),
    ("tuesday",   "weekday"),
    ("wednesday", "weekday"),
    ("thursday",  "weekday"),
    ("friday",    "weekday"),
    # weekend
    ("saturday", "weekend"),
    ("sunday",   "weekend"),
    # month
    ("january",  "month"),
    ("february", "month"),
    ("march",    "month"),
    ("april",    "month"),
    ("may",      "month"),
    ("june",     "month"),
    ("july",     "month"),
    ("august",   "month"),
    # season
    ("spring", "season"),
    ("summer", "season"),
    ("autumn", "season"),
    ("winter", "season"),
    # time-of-day
    ("morning",   "time"),
    ("noon",      "time"),
    ("afternoon", "time"),
    ("evening",   "time"),
    ("midnight",  "time"),
]

# Truly novel — same categories, different items, never in train.
TRULY_NOVEL_PAIRS: list[tuple[str, str]] = [
    ("october",  "month"),
    ("november", "month"),
    ("december", "month"),
    ("dawn",     "time"),
    ("dusk",     "time"),
]

TEMPLATES: list[str] = [
    "{subject} is a {category}",
    "a {subject} is a {category}",
    "{subject} is a kind of {category}",
    "we call {subject} a {category}",
    "the {subject} is a {category}",
    "every {subject} is a {category}",
]


def audit_no_overlap() -> None:
    train_set = set(TRAIN_PAIRS)
    leaks = [p for p in TRULY_NOVEL_PAIRS if p in train_set]
    if leaks:
        print(f"FATAL: TRULY_NOVEL pairs leak into train: {leaks}")
        raise SystemExit(1)
    if len(set(TRAIN_PAIRS)) != len(TRAIN_PAIRS):
        print(f"FATAL: train pairs contain duplicates")
        raise SystemExit(1)


def render(pairs: list[tuple[str, str]]) -> tuple[list[str], list[tuple[str, str]]]:
    sents, words = [], []
    for subj, cat in pairs:
        for tpl in TEMPLATES:
            sents.append(tpl.format(subject=subj, category=cat))
            words.append((subj, cat))
    return sents, words


# ---------------------------------------------------------------------------
# Main smoke
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000,
                        help="decoder steps (3.4 is plumbing-test scale; "
                             "use 5000-10000 for quality-comparable to 3.1)")
    parser.add_argument("--mlp-steps", type=int, default=1500)
    parser.add_argument("--energy-kind", default="MLPEnergyModel",
                        choices=["GaussianEnergy", "MLPEnergyModel"])
    parser.add_argument("--registry-root", default=None,
                        help="if omitted, uses a tempdir that is cleaned up")
    parser.add_argument("--phase2a-holdout",
                        default="data/explanations_v2/holdout.tsv",
                        help="optional out-of-domain probes for energy ROC-AUC")
    parser.add_argument("--out", default="results/stage3/orchestrator_smoke.json")
    args = parser.parse_args()

    audit_no_overlap()

    enc_cfg = ENCODERS[args.encoder]
    print("Stage 3 / Sub-task 3.4 — orchestrator smoke (temporal domain)")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    # ---- Encoder load (shared across calls) ---------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = enc_cfg["dim"]

    # ---- Build temporal corpus ----------------------------------------
    train_sents, train_words = render(TRAIN_PAIRS)
    holdout_sents, holdout_words = render(TRULY_NOVEL_PAIRS)
    print(f"\n[1] temporal corpus")
    print("-" * 78)
    print(f"  train:    {len(train_sents)} sentences ({len(TRAIN_PAIRS)} pairs × {len(TEMPLATES)} templates)")
    print(f"  holdout:  {len(holdout_sents)} sentences ({len(TRULY_NOVEL_PAIRS)} truly-novel × {len(TEMPLATES)})")
    print(f"  samples (train):    {train_sents[:2]!r}")
    print(f"  samples (holdout):  {holdout_sents[:2]!r}")

    # ---- Optional out-of-domain ψ for energy AUC ----------------------
    out_of_domain_psi = None
    phase2a_path = Path(args.phase2a_holdout)
    if phase2a_path.exists():
        print(f"\n[2] encoding out-of-domain probes from {phase2a_path}")
        print("-" * 78)
        rows = read_corpus_tsv(phase2a_path)
        ood_sents = [r.sentence for r in rows]
        from selflearnai.domain.ingest import _encode_pooled
        out_of_domain_psi = _encode_pooled(
            ood_sents, tok, mdl, args.device, max_length=32,
        ).cpu()
        print(f"  out-of-domain ψ: {tuple(out_of_domain_psi.shape)}")
    else:
        print(f"\n[2] no out-of-domain probes ({phase2a_path} missing) — skipping energy AUC eval")

    # ---- Registry root (tempdir if not specified) --------------------
    if args.registry_root is None:
        registry_root = Path(tempfile.mkdtemp(prefix="domain_orchestrator_smoke_"))
        cleanup = True
    else:
        registry_root = Path(args.registry_root)
        cleanup = False
    print(f"\n[3] registry root: {registry_root}")
    print("-" * 78)

    workdir = registry_root / "_workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        # ---- Round-trip via ingest_domain ----------------------------
        cfg = IngestionConfig(
            domain_id="temporal",
            decoder_steps=args.steps,
            mlp_steps=args.mlp_steps,
            energy_kind=args.energy_kind,
            seed=args.seed,
            log_every=max(args.steps // 8, 100),
        )
        registry = DomainRegistry(registry_root, max_active=10)
        print(f"\n[4] calling ingest_domain(...)")
        print("-" * 78)
        result = ingest_domain(
            cfg,
            train_sents=train_sents, train_pair_words=train_words,
            holdout_sents=holdout_sents, holdout_pair_words=holdout_words,
            encoder_tokenizer=tok, encoder_model=mdl,
            encoder_dim=DIM, device=args.device,
            registry=registry, workdir=workdir,
            out_of_domain_psi=out_of_domain_psi,
            verbose=True,
        )
        print(f"\n[ingest_domain returned] verdict={result.verdict}")
        print(f"  decoder_eval: {result.decoder_eval}")
        print(f"  gates: {result.gates}")
        print(f"  energy_auc: {result.energy_auc}")
        print(f"  registered as version: {result.registry_entry.current_version}")

        # ---- ACCEPTANCE CHECKS ---------------------------------------
        print("\n" + "=" * 78)
        print("ACCEPTANCE CHECKS (pipeline plumbing)")
        print("=" * 78)
        sub = {}
        all_pass = True

        # 1. Artifacts land in registry/artifacts/<id>__v1/
        artifact_dir = registry_root / "artifacts" / "temporal__v1"
        decoder_in_reg = artifact_dir / "decoder.pt"
        energy_filename = f"energy_{cfg.energy_kind.lower()}.pt"
        energy_in_reg = artifact_dir / energy_filename
        prov_in_reg = artifact_dir / "provenance.json"
        ok1 = decoder_in_reg.exists() and energy_in_reg.exists() and prov_in_reg.exists()
        sub["1_artifacts_in_registry"] = ok1
        all_pass = all_pass and ok1
        print(f"  [1] artifacts under {artifact_dir.name}/  "
              f"{'PASS' if ok1 else 'FAIL'}")
        if not ok1:
            print(f"      decoder.pt: {decoder_in_reg.exists()}")
            print(f"      {energy_filename}: {energy_in_reg.exists()}")
            print(f"      provenance.json: {prov_in_reg.exists()}")

        # 2. Persistence: fresh registry instance reloads
        fresh = DomainRegistry(registry_root, max_active=10)
        ok2 = fresh.has("temporal")
        if ok2:
            entry = fresh.get_entry("temporal")
            ok2 = (entry.current_version == 1
                   and len(entry.versions) == 1
                   and entry.current().energy_kind == cfg.energy_kind)
        sub["2_fresh_registry_persistence"] = ok2
        all_pass = all_pass and ok2
        print(f"  [2] fresh registry reload                          "
              f"{'PASS' if ok2 else 'FAIL'}")

        # 3. Decoder checkpoint reloads + forward pass
        try:
            decoder_path = registry_root / fresh.get_entry("temporal").current().decoder_path
            new_decoder = PointerSeqCondDecoder(
                encoder_dim=DIM, hidden_dim=cfg.hidden_dim, t_max=cfg.t_max,
                vocab_size=tok.vocab_size, n_layers=cfg.n_layers,
                n_heads=cfg.n_heads, ffn_mult=cfg.ffn_mult,
                feat_dropout=cfg.feat_dropout, attn_dropout=cfg.attn_dropout,
            ).to(args.device)
            sd = torch.load(str(decoder_path), map_location=args.device)
            new_decoder.load_state_dict(sd)
            new_decoder.eval()
            from selflearnai.domain.ingest import _encode_activations
            h_smoke, h_mask_smoke = _encode_activations(
                holdout_sents[:2], tok, mdl, args.device, t_max=cfg.t_max,
            )
            ids_smoke = tok(holdout_sents[:2], padding="max_length",
                             truncation=True, max_length=cfg.t_max,
                             return_tensors="pt").to(args.device).input_ids
            with torch.no_grad():
                lp, _, pg = new_decoder(h_smoke, h_mask_smoke, ids_smoke)
            ok3 = (lp.shape == (2, cfg.t_max, tok.vocab_size)
                   and pg.shape == (2, cfg.t_max, 1)
                   and torch.isfinite(lp).all().item())
        except Exception as e:
            ok3 = False
            print(f"      reload error: {e}")
        sub["3_decoder_reload_forward"] = ok3
        all_pass = all_pass and ok3
        print(f"  [3] decoder reload + forward pass                  "
              f"{'PASS' if ok3 else 'FAIL'}")

        # 4. Energy reload + scoring
        try:
            energy_path = registry_root / fresh.get_entry("temporal").current().energy_path
            if cfg.energy_kind == "GaussianEnergy":
                em = GaussianEnergy.load(energy_path)
            else:
                em = MLPEnergyModel.load(energy_path)
                em.eval()
            from selflearnai.domain.ingest import _encode_pooled as _ep
            psi_in = _ep(
                holdout_sents[:3], tok, mdl, args.device, max_length=cfg.t_max,
            ).cpu()
            with torch.no_grad():
                e_in_reload = em.energy(psi_in)
            ok4 = (e_in_reload.shape == (3,)
                   and torch.isfinite(e_in_reload).all().item())
        except Exception as e:
            ok4 = False
            print(f"      reload error: {e}")
        sub["4_energy_reload_scoring"] = ok4
        all_pass = all_pass and ok4
        print(f"  [4] energy reload + scoring                        "
              f"{'PASS' if ok4 else 'FAIL'}")

        # 5. Provenance JSON roundtrip
        try:
            with open(prov_in_reg) as f:
                prov_disk = json.load(f)
            ok5 = (prov_disk.get("n_train_sents") == len(train_sents)
                   and prov_disk.get("n_holdout_sents") == len(holdout_sents)
                   and "decoder_eval" in prov_disk
                   and "energy_eval" in prov_disk
                   and prov_disk["energy_eval"]["kind"] == cfg.energy_kind)
        except Exception as e:
            ok5 = False
            print(f"      provenance error: {e}")
        sub["5_provenance_roundtrip"] = ok5
        all_pass = all_pass and ok5
        print(f"  [5] provenance JSON roundtrip                      "
              f"{'PASS' if ok5 else 'FAIL'}")

        # ---- Verdict --------------------------------------------------
        verdict = "STAGE_3_4_PASS" if all_pass else "STAGE_3_4_FAIL"
        print()
        print(f"→ Stage 3.4: {'PASS' if all_pass else 'FAIL'}")
        if all_pass:
            print(
                f"  Universal ingestion orchestrator round-trips a fresh\n"
                f"  domain (temporal) end-to-end: train decoder → train energy\n"
                f"  → register → reload-from-disk → score. All 5 plumbing\n"
                f"  checks pass. The (decoder, energy, registry) trio is the\n"
                f"  abstraction that lets future Stage 3 sub-tasks add new\n"
                f"  domains in one call."
            )

        # ---- Save JSON ----------------------------------------------
        payload = {
            "task": "3.4",
            "verdict": verdict,
            "all_pass": all_pass,
            "sub_acceptance": sub,
            "encoder": args.encoder,
            "encoder_dim": DIM,
            "domain": {
                "domain_id": "temporal",
                "n_train_pairs": len(TRAIN_PAIRS),
                "n_truly_novel_pairs": len(TRULY_NOVEL_PAIRS),
                "n_templates": len(TEMPLATES),
                "n_train_sents": len(train_sents),
                "n_holdout_sents": len(holdout_sents),
            },
            "ingest_result": {
                "decoder_n_params": result.decoder_n_params,
                "decoder_ckpt_path": result.decoder_ckpt_path,
                "energy_kind": result.energy_kind,
                "energy_ckpt_path": result.energy_ckpt_path,
                "energy_auc": result.energy_auc,
                "decoder_eval": result.decoder_eval,
                "gates": result.gates,
                "verdict": result.verdict,
                "message": result.message,
            },
            "registry_root": str(registry_root),
            "config": {
                "decoder_steps": cfg.decoder_steps,
                "mlp_steps": cfg.mlp_steps,
                "energy_kind": cfg.energy_kind,
            },
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n→ saved JSON to {out_path}")

    finally:
        if cleanup:
            shutil.rmtree(registry_root, ignore_errors=True)
            print(f"→ cleaned up {registry_root}")

    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
