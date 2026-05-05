"""Task 1.11 — Ψ-program trace serialization round-trip smoke.

For each of the 6 chain test cases (paint → painters, etc.):

  1. Train operators + fit step calibrators (same as Task 1.10).
  2. Run planner.execute_hint with chain ['agentive', 'plural'].
  3. Run verify_chain to get per-step gate results.
  4. Build a PsiProgram from the planner + verifier output.
  5. Serialize to JSON, write to disk.
  6. Read the JSON back as a PsiProgram.
  7. Verify field-level equality on the round-trip (no information lost).
  8. Replay the loaded program (re-encode source + re-execute chain
     against fresh operators) and confirm reproducibility.

Acceptance gate (Task 1.11):
  - All 6 traces serialize, deserialize, and round-trip with
    final_cos >= 0.999 (replay reproduces the saved final ψ).
  - All 6 replays produce the same top-1 over the FAIR pool as the
    saved trace.
  - JSON files are non-empty and parseable.

Output: per-case verdict line, JSON files in
results/stage1/traces/<verb>__<plural_agent>.json, and a summary
JSON in results/stage1/planner_trace_smoke.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.planner import (
    AxiomGate,
    BeamSearchPlanner,
    ConformalGate,
    DEFAULT_TYPE_SIGNATURES,
    DriftGate,
    PsiProgram,
    StepCalibrator,
    TypeGate,
    build_psi_program,
    replay,
    verify_chain,
)

# Reuse data + helpers from earlier Stage-1 scripts.
from scripts.stage1_planner_beam_smoke import (
    CHAIN_TRIPLES,
    POOL_FAIR,
    ENCODERS,
    read_pairs,
    make_encode_fn,
)
from scripts.stage1_planner_prior_train import (
    CONCEPTS_DATA,
    build_concept_operators,
)
from scripts.stage1_planner_verifier_smoke import build_reference_vocab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="gte-base", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operator-epochs", type=int, default=2000)
    parser.add_argument("--axiom-threshold", type=float, default=0.99)
    parser.add_argument("--drift-threshold", type=float, default=0.5)
    parser.add_argument("--conformal-alpha", type=float, default=0.10)
    parser.add_argument(
        "--final-cos-min", type=float, default=0.999,
        help="Replay reproducibility threshold for the final ψ.",
    )
    parser.add_argument(
        "--per-step-cos-min", type=float, default=0.99,
        help="Replay reproducibility threshold for intermediate states.",
    )
    parser.add_argument(
        "--traces-dir", default="results/stage1/traces",
        help="Directory to write per-case Ψ-program JSON files.",
    )
    parser.add_argument(
        "--out", default="results/stage1/planner_trace_smoke.json",
        help="Summary JSON output path.",
    )
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Task 1.11 — Ψ-program trace serialization round-trip smoke")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Reproducibility thresholds: final_cos>={args.final_cos_min} "
          f"per_step_cos>={args.per_step_cos_min}")

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Reference vocab + operators + calibrators (identical to Task 1.10) ----
    print("\nBuilding DriftGate reference vocabulary ...")
    ref_words = build_reference_vocab(CONCEPTS_DATA)
    ref_emb = encode(ref_words)
    print(f"  reference vocab: {len(ref_words)} words")

    print("\nTraining all 7 ConceptOperators ...")
    operators = build_concept_operators(
        encode, CONCEPTS_DATA, dim=enc_cfg["dim"], device=args.device,
        seed=args.seed, epochs=args.operator_epochs,
    )

    print("\nFitting per-operator step calibrators ...")
    step_calibrators: dict[str, StepCalibrator] = {}
    for concept, ddir in CONCEPTS_DATA:
        train_pairs = read_pairs(Path(ddir) / "text_pairs_train.tsv")
        sources = [p[0] for p in train_pairs]
        if not sources:
            continue
        step_calibrators[concept] = StepCalibrator.fit(
            op_name=concept, op=operators[concept],
            z_src=encode(sources), alpha=args.conformal_alpha,
        )

    # ---- Gate stack ----
    gates = [
        TypeGate(signatures=DEFAULT_TYPE_SIGNATURES, source_type="Verb"),
        AxiomGate(threshold=args.axiom_threshold),
        DriftGate(reference_embeddings=ref_emb, threshold=args.drift_threshold),
        ConformalGate(calibrators=step_calibrators),
    ]

    # ---- Planner (used as chain executor) ----
    planner = BeamSearchPlanner(
        operators=operators, beam_width=4, max_depth=3, step_bonus=0.015,
    )

    # ---- Encode FAIR pool once for final argmax ----
    z_pool = encode(POOL_FAIR)
    pool_n = F.normalize(z_pool, dim=-1)

    traces_dir = Path(args.traces_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)

    chain_hint = ["agentive", "plural"]
    print("\n" + "=" * 78)
    print(f"PER-CASE TRACE BUILD + ROUND-TRIP  (chain: {chain_hint})")
    print("=" * 78)

    case_records: list[dict] = []
    n_round_trip_pass = 0
    n_replay_pass = 0
    n_top1_match = 0

    for verb, _, plural_agent in CHAIN_TRIPLES:
        # Encode source + goal.
        psi_start = encode([verb]).squeeze(0)
        psi_goal = encode([plural_agent]).squeeze(0)

        # Run planner.
        plan = planner.execute_hint(psi_start, psi_goal, chain_hint)

        # Re-execute step-by-step to collect per-step intermediates,
        # because plan only carries the final psi.
        intermediate_psis: list[torch.Tensor] = []
        psi_t = psi_start
        with torch.no_grad():
            for op_name in plan.chain:
                psi_t = operators[op_name](psi_t.unsqueeze(0)).squeeze(0)
                intermediate_psis.append(psi_t)

        # Verify chain to populate the gate results we'll save.
        verif = verify_chain(plan.chain, operators, psi_start, gates)

        # Final pool argmax.
        pred_n = F.normalize(plan.psi.unsqueeze(0), dim=-1)
        sims = pred_n @ pool_n.T
        top1_idx = int(sims.argmax(dim=-1).item())
        top1_word = POOL_FAIR[top1_idx]

        # Build the Ψ-program.
        program = build_psi_program(
            source=verb,
            target=plural_agent,
            encoder_name=args.encoder,
            encoder_dim=enc_cfg["dim"],
            psi_initial=psi_start,
            psi_goal=psi_goal,
            chain=plan.chain,
            intermediate_psis=intermediate_psis,
            verification_steps=verif.steps,
            final_top1_word=top1_word,
            final_cos_to_goal=plan.cos_to_goal,
            metadata={
                "task": "1.11",
                "planner": {
                    "beam_width": planner.beam_width,
                    "max_depth": planner.max_depth,
                    "step_bonus": planner.step_bonus,
                    "execute_hint_chain": chain_hint,
                },
                "operator_epochs": args.operator_epochs,
                "seed": args.seed,
            },
        )

        # Serialize → file → load → field-level round-trip equality.
        trace_path = traces_dir / f"{verb}__{plural_agent}.json"
        program.to_file(trace_path)
        program_loaded = PsiProgram.from_file(trace_path)
        round_trip_ok = (
            program_loaded.source == program.source
            and program_loaded.target == program.target
            and program_loaded.encoder_name == program.encoder_name
            and program_loaded.encoder_dim == program.encoder_dim
            and len(program_loaded.steps) == len(program.steps)
            and tuple(program_loaded.chain) == tuple(program.chain)
            and program_loaded.final_top1_word == program.final_top1_word
            and (
                len(program_loaded.psi_initial) == len(program.psi_initial)
            )
        )
        if round_trip_ok:
            n_round_trip_pass += 1

        # Replay against the loaded program.
        result = replay(
            program_loaded,
            encoder_encode_fn=encode,
            operators=operators,
            candidate_pool=POOL_FAIR,
            final_psi_threshold=args.final_cos_min,
            per_step_threshold=args.per_step_cos_min,
        )
        if result.reproducible:
            n_replay_pass += 1
        if result.final_top1_match:
            n_top1_match += 1

        rt_mark = "✓" if round_trip_ok else "✗"
        rp_mark = "✓" if result.reproducible else "✗"
        m_mark = "✓" if result.final_top1_match else "✗"
        print(
            f"  {verb:<6}→ {plural_agent:<10}  "
            f"{rt_mark} round-trip  "
            f"{rp_mark} replay (final_cos={result.final_cos_to_saved:.4f})  "
            f"{m_mark} top-1: replay={result.final_top1_word_replay}, "
            f"saved={result.final_top1_word_saved}"
        )

        case_records.append({
            "verb": verb,
            "expected_target": plural_agent,
            "trace_path": str(trace_path),
            "round_trip_ok": round_trip_ok,
            "replay": {
                "reproducible": result.reproducible,
                "final_cos_to_saved": result.final_cos_to_saved,
                "final_top1_word_saved": result.final_top1_word_saved,
                "final_top1_word_replay": result.final_top1_word_replay,
                "final_top1_match": result.final_top1_match,
                "per_step_cos_to_saved": result.per_step_cos_to_saved,
                "notes": result.notes,
            },
        })

    # ---- Acceptance ----
    n = len(CHAIN_TRIPLES)
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.11)")
    print("=" * 78)
    rt_pass = n_round_trip_pass == n
    rp_pass = n_replay_pass == n
    m_pass = n_top1_match == n
    print(
        f"  Round-trip integrity: {n_round_trip_pass}/{n}  "
        f"(HARD gate: all)  → {'PASS' if rt_pass else 'FAIL'}"
    )
    print(
        f"  Replay reproducibility (final_cos >= {args.final_cos_min}): "
        f"{n_replay_pass}/{n}  → {'PASS' if rp_pass else 'FAIL'}"
    )
    print(
        f"  Top-1 word match (replay == saved): "
        f"{n_top1_match}/{n}  → {'PASS' if m_pass else 'FAIL'}"
    )
    overall = rt_pass and rp_pass and m_pass
    print(f"\n→ Task 1.11: {'PASS' if overall else 'FAIL'}")

    # ---- Save summary JSON ----
    payload = {
        "task": "1.11",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "thresholds": {
            "final_cos_min": args.final_cos_min,
            "per_step_cos_min": args.per_step_cos_min,
        },
        "n_cases": n,
        "n_round_trip_pass": n_round_trip_pass,
        "n_replay_pass": n_replay_pass,
        "n_top1_match": n_top1_match,
        "traces_dir": str(traces_dir),
        "cases": case_records,
        "pass": overall,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved per-case traces to {traces_dir}/")
    print(f"→ saved summary JSON to {out_path}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
