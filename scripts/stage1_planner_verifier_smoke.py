"""Task 1.10 — verification gates smoke test (Type + Axiom + Drift + Conformal).

Runs all four structural verification gates from
selflearnai/planner/verifier.py on the agentive ∘ plural chain test
cases (paint → painters, drive → drivers, ..., help → helpers).

For each test case (verb, plural_agent):
  - Encode verb → ψ_start
  - Train operator library {agentive, plural, ...}
  - Fit a StepCalibrator per operator on its training-pair sources
  - Run planner.execute_hint(ψ_start, ψ_goal, ['agentive', 'plural'])
  - Run verify_chain on the result with all four gates
  - Confirm every gate passes at every step

Plus a CONTROL test: explicitly try a type-incorrect chain (plural
applied to a Verb input) and confirm TypeGate REJECTS it. This
validates that gates are actually doing their job, not just rubber-
stamping every chain.

Acceptance gate (Task 1.10):
  - Correct chain:   all 4 gates pass on every step of all 6 cases
                     (4 gates × 2 steps × 6 cases = 48 checks all green).
  - Type-incorrect chain: TypeGate REJECTS the first step (the
                          source-type mismatch), validating it's a
                          real check rather than a tautology.

Pure CPU after the encoder loads — gates are tiny operations (cos
similarity + lookup). The whole smoke runs in seconds once operators
are trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selflearnai.planner import (
    BeamSearchPlanner,
    DEFAULT_TYPE_SIGNATURES,
    AxiomGate,
    ConformalGate,
    DriftGate,
    StepCalibrator,
    TypeGate,
    verify_chain,
)

# Reuse data + helpers from earlier Stage-1 scripts to stay consistent.
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


# Reference vocabulary for the DriftGate. Drawn from each concept's
# train + held-out source/target pairs so any "valid" intermediate
# has a known nearby word.
def build_reference_vocab(concepts_data) -> list[str]:
    vocab: set[str] = set()
    for _concept, ddir in concepts_data:
        for fname in ("text_pairs_train.tsv", "text_pairs_held_out.tsv"):
            path = Path(ddir) / fname
            if not path.exists():
                continue
            for src, tgt in read_pairs(path):
                vocab.add(src.lower())
                vocab.add(tgt.lower())
    # Add the FAIR pool too — those are typical chain-test target words.
    for w in POOL_FAIR:
        vocab.add(w.lower())
    return sorted(vocab)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="gte-base", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operator-epochs", type=int, default=2000)
    parser.add_argument("--axiom-threshold", type=float, default=0.99)
    parser.add_argument("--drift-threshold", type=float, default=0.5)
    parser.add_argument(
        "--conformal-alpha", type=float, default=0.10,
        help="Per-operator step-magnitude calibration α. Lower = tighter "
             "lower bound (more rejections of borderline behavior).",
    )
    parser.add_argument("--out", default="results/stage1/planner_verifier_smoke.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Task 1.10 — verification gates smoke (Type / Axiom / Drift / Conformal)")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Axiom non-identity threshold: {args.axiom_threshold}")
    print(f"Drift on-manifold threshold:  {args.drift_threshold}")
    print(f"Conformal calibration α:      {args.conformal_alpha}")

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Reference vocab for DriftGate ----
    print("\nBuilding reference vocabulary for DriftGate ...")
    ref_words = build_reference_vocab(CONCEPTS_DATA)
    print(f"  reference vocab: {len(ref_words)} words")
    ref_emb = encode(ref_words)

    # ---- Operators ----
    print("\nTraining all 7 ConceptOperators ...")
    operators = build_concept_operators(
        encode, CONCEPTS_DATA, dim=enc_cfg["dim"], device=args.device,
        seed=args.seed, epochs=args.operator_epochs,
    )

    # ---- Per-operator step calibrators (for ConformalGate) ----
    print("\nFitting per-operator step calibrators ...")
    step_calibrators: dict[str, StepCalibrator] = {}
    for concept, ddir in CONCEPTS_DATA:
        train_pairs = read_pairs(Path(ddir) / "text_pairs_train.tsv")
        sources = [p[0] for p in train_pairs]
        if not sources:
            print(f"  {concept:<14s}  (no train pairs; skipped)")
            continue
        z_src = encode(sources)
        cal = StepCalibrator.fit(
            op_name=concept,
            op=operators[concept],
            z_src=z_src,
            alpha=args.conformal_alpha,
        )
        step_calibrators[concept] = cal
        cos_arr = cal.raw_cos_values
        print(
            f"  {concept:<14s}  n={cal.n_calib:<3d}  "
            f"cos_low(α={args.conformal_alpha})={cal.cos_low:+.4f}  "
            f"[min={min(cos_arr):+.4f}, max={max(cos_arr):+.4f}]"
        )

    # ---- Planner (used as the chain executor only) ----
    planner = BeamSearchPlanner(
        operators=operators, beam_width=4, max_depth=3, step_bonus=0.015,
    )

    # ---- Build gate stack ----
    type_gate_with_verb_source = TypeGate(
        signatures=DEFAULT_TYPE_SIGNATURES, source_type="Verb",
    )
    axiom_gate = AxiomGate(threshold=args.axiom_threshold)
    drift_gate = DriftGate(reference_embeddings=ref_emb, threshold=args.drift_threshold)
    conformal_gate = ConformalGate(calibrators=step_calibrators)
    gates = [type_gate_with_verb_source, axiom_gate, drift_gate, conformal_gate]

    # ============================================================
    # MAIN TEST: correct chain (agentive ∘ plural) on each verb.
    # All gates must pass.
    # ============================================================
    print("\n" + "=" * 78)
    print("MAIN TEST: correct chain agentive ∘ plural (source_type=Verb)")
    print("=" * 78)
    n_cases = len(CHAIN_TRIPLES)
    n_gate_total = 0
    n_gate_passed = 0
    case_records: list[dict] = []
    for verb, _, plural_agent in CHAIN_TRIPLES:
        psi_start = encode([verb]).squeeze(0)
        psi_goal = encode([plural_agent]).squeeze(0)
        plan = planner.execute_hint(psi_start, psi_goal, ["agentive", "plural"])
        verif = verify_chain(plan.chain, operators, psi_start, gates)

        print(f"\n  {verb} → {plural_agent}  chain: {' ∘ '.join(plan.chain)}")
        for step in verif.steps:
            print(f"    step {step.step_index} ({step.op_name}):")
            for gate_name, gr in step.gate_results.items():
                tag = "✓" if gr.passed else "✗"
                print(f"      {tag} {gate_name:<20s}  {gr.reason}")
                n_gate_total += 1
                if gr.passed:
                    n_gate_passed += 1

        case_records.append({
            "verb": verb,
            "expected_target": plural_agent,
            "chain": list(plan.chain),
            "all_gates_passed": verif.all_passed,
            "step_summary": verif.step_summary(),
            "per_step": [
                {
                    "step": s.step_index,
                    "op": s.op_name,
                    "all_passed": s.all_passed,
                    "gates": {
                        gn: {
                            "passed": gr.passed,
                            "metric": gr.metric,
                            "threshold": gr.threshold,
                            "reason": gr.reason,
                        }
                        for gn, gr in s.gate_results.items()
                    },
                }
                for s in verif.steps
            ],
        })

    print(
        f"\n  Main test summary: {n_gate_passed}/{n_gate_total} gate checks passed "
        f"across {n_cases} cases × {len(gates)} gates × 2 steps."
    )

    # ============================================================
    # CONTROL TEST: type-incorrect chain (plural applied to Verb).
    # TypeGate must reject the first step.
    # ============================================================
    print("\n" + "=" * 78)
    print("CONTROL TEST: type-incorrect chain plural ∘ plural (source_type=Verb)")
    print("=" * 78)
    print("  Expectation: TypeGate fails on step 0 because plural's input "
          "type is Noun, not Verb.")

    # Use a fresh planner just to call execute_hint — chain forced.
    bad_chain = ["plural", "plural"]
    verb = CHAIN_TRIPLES[0][0]
    plural_agent = CHAIN_TRIPLES[0][2]
    psi_start = encode([verb]).squeeze(0)
    psi_goal = encode([plural_agent]).squeeze(0)
    bad_plan = planner.execute_hint(psi_start, psi_goal, bad_chain)
    bad_verif = verify_chain(bad_plan.chain, operators, psi_start, gates)
    type_gate_step0 = bad_verif.steps[0].gate_results["type"]
    type_rejection_works = not type_gate_step0.passed

    print(f"\n  bad chain: {' ∘ '.join(bad_plan.chain)} (source: '{verb}', Verb)")
    for step in bad_verif.steps:
        print(f"    step {step.step_index} ({step.op_name}):")
        for gate_name, gr in step.gate_results.items():
            tag = "✓" if gr.passed else "✗"
            print(f"      {tag} {gate_name:<20s}  {gr.reason}")
    print(
        f"\n  Type-rejection on step 0: "
        f"{'CORRECTLY REJECTED' if type_rejection_works else 'FAILED TO REJECT'}"
    )

    # ============================================================
    # ACCEPTANCE
    # ============================================================
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.10)")
    print("=" * 78)
    main_pass = n_gate_passed == n_gate_total
    print(
        f"  Main test (correct chain): {n_gate_passed}/{n_gate_total} gates "
        f"passed (HARD gate: all)  → {'PASS' if main_pass else 'FAIL'}"
    )
    print(
        f"  Control test (TypeGate rejects bad chain): "
        f"→ {'PASS' if type_rejection_works else 'FAIL'}"
    )
    overall = main_pass and type_rejection_works
    print(f"\n→ Task 1.10: {'PASS' if overall else 'FAIL'}")

    # ---- Save JSON ----
    payload = {
        "task": "1.10",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "thresholds": {
            "axiom_non_identity": args.axiom_threshold,
            "drift": args.drift_threshold,
            "conformal_alpha": args.conformal_alpha,
        },
        "step_calibrators": {
            name: {
                "cos_low": cal.cos_low,
                "alpha": cal.alpha,
                "n_calib": cal.n_calib,
            }
            for name, cal in step_calibrators.items()
        },
        "ref_vocab_size": len(ref_words),
        "main_test": {
            "n_cases": n_cases,
            "n_gates_per_step": len(gates),
            "n_steps_per_chain": 2,
            "n_gate_total": n_gate_total,
            "n_gate_passed": n_gate_passed,
            "cases": case_records,
        },
        "control_test": {
            "bad_chain": bad_chain,
            "type_rejection_works": type_rejection_works,
            "step0_type_gate_reason": type_gate_step0.reason,
        },
        "pass": overall,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
