"""Task 1.12 — depth-3-to-5 planner validation across the full library.

Stress test for the planner's beam search at increasing search horizons.
Up through Task 1.11 every test case used the toy 2-operator library
{agentive, plural}. This script enables the FULL 7-operator library
(plural, past_tense, comparative, superlative, opposite, agentive,
young) and runs `planner.search` (NO chain hint) at four max_depth
values on the six chain test cases.

The question this answers: with 7 operators and max_depth=5, the
search space holds 7^5 = 16,807 candidate chains plus shorter ones.
Does the cos-progress + step-bonus heuristic still find the correct
depth-2 chain `agentive ∘ plural` for paint→painters etc, or does
the planner get distracted by spurious longer chains that happen
to score similarly?

Per-depth metrics, all reported against the same six (verb,
plural_agent) pairs:
  - chain_recovery: planner.search(...).chain == ('agentive', 'plural')
  - end_state_correct: argmax(planner-output ψ) over FAIR pool ==
                       expected plural-agent

Acceptance gates (Task 1.12):
  - Depth 2:  chain_recovery ≥ 5/6  (matches Task 1.7 baseline)
  - Depth 3:  chain_recovery ≥ 5/6  (≥80% gate from plan §19.2)
  - Depth 4:  chain_recovery ≥ 4/6  (≥60% gate from plan §19.2)
  - Depth 5:  chain_recovery ≥ 3/6  (≥40% gate from plan §19.2)
  - All depths: end_state_correct ≥ 5/6  (operators work; argmax is
                                          mostly insensitive to chain
                                          choice once the embedding
                                          lands near the goal)

Output: per-depth row in a single summary table + per-case detail per
depth + JSON to results/stage1/planner_depth_validation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F

from selflearnai.planner import BeamSearchPlanner

from scripts.stage1_planner_beam_smoke import (
    CHAIN_TRIPLES,
    POOL_FAIR,
    ENCODERS,
    make_encode_fn,
)
from scripts.stage1_planner_prior_train import (
    CONCEPTS_DATA,
    build_concept_operators,
)


# Acceptance thresholds, one per max_depth, matching plan §19.2.
DEFAULT_DEPTH_GATES: dict[int, dict] = {
    2: {"chain_min": 5, "label": "≥83% (matches Task 1.7)"},
    3: {"chain_min": 5, "label": "≥83% (≥80% from plan §19.2)"},
    4: {"chain_min": 4, "label": "≥66% (≥60% from plan §19.2)"},
    5: {"chain_min": 3, "label": "≥50% (≥40% from plan §19.2)"},
}
END_STATE_MIN = 5  # at every depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="gte-base", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operator-epochs", type=int, default=2000)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--step-bonus", type=float, default=0.015)
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[2, 3, 4, 5],
        help="max_depth values to validate. Defaults to {2,3,4,5}.",
    )
    parser.add_argument("--out", default="results/stage1/planner_depth_validation.json")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Task 1.12 — depth-3-to-5 planner validation")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Library:  full 7-operator concept library")
    print(f"Depths:   {args.depths}")
    print(f"beam_width={args.beam_width}, step_bonus={args.step_bonus}")

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Operators (full 7-concept library) ----
    print("\nTraining all 7 ConceptOperators ...")
    operators = build_concept_operators(
        encode, CONCEPTS_DATA, dim=enc_cfg["dim"], device=args.device,
        seed=args.seed, epochs=args.operator_epochs,
    )
    print(f"  library: {sorted(operators.keys())}")

    # ---- FAIR pool encode once ----
    z_pool = encode(POOL_FAIR)
    pool_n = F.normalize(z_pool, dim=-1)

    expected_chain = ("agentive", "plural")
    n_cases = len(CHAIN_TRIPLES)
    n_ops = len(operators)

    # ---- Per-depth runs ----
    per_depth_records: dict[int, dict] = {}
    for depth in args.depths:
        planner = BeamSearchPlanner(
            operators=operators,
            beam_width=args.beam_width,
            max_depth=depth,
            step_bonus=args.step_bonus,
        )
        search_space = sum(n_ops ** k for k in range(1, depth + 1))

        print("\n" + "=" * 78)
        print(f"DEPTH = {depth}  (search space ≈ {search_space} chains; "
              f"7^{depth} = {n_ops**depth} length-{depth} alone)")
        print("=" * 78)

        chain_correct = 0
        end_state_correct = 0
        case_records: list[dict] = []
        for verb, _, plural_agent in CHAIN_TRIPLES:
            psi_start = encode([verb]).squeeze(0)
            psi_goal = encode([plural_agent]).squeeze(0)
            result = planner.search(psi_start, psi_goal)

            pred_n = F.normalize(result.psi.unsqueeze(0), dim=-1)
            sims = pred_n @ pool_n.T
            top1_word = POOL_FAIR[int(sims.argmax(dim=-1).item())]

            chain_pass = result.chain == expected_chain
            end_pass = top1_word == plural_agent
            if chain_pass:
                chain_correct += 1
            if end_pass:
                end_state_correct += 1

            chain_repr = " ∘ ".join(result.chain) if result.chain else "(no-op)"
            c_mark = "✓" if chain_pass else "✗"
            e_mark = "✓" if end_pass else "✗"
            print(
                f"  {verb:<6}→ {plural_agent:<10}  "
                f"{c_mark} chain: {chain_repr:<42}  "
                f"{e_mark} top-1: {top1_word:<10}  cos={result.cos_to_goal:+.3f}"
            )
            case_records.append({
                "verb": verb,
                "expected_target": plural_agent,
                "chain": list(result.chain),
                "chain_correct": chain_pass,
                "top1_pool_word": top1_word,
                "end_state_correct": end_pass,
                "cos_to_goal": result.cos_to_goal,
                "score": result.score,
            })

        gates = DEFAULT_DEPTH_GATES.get(
            depth, {"chain_min": max(1, n_cases // 2), "label": "soft"}
        )
        chain_pass_overall = chain_correct >= gates["chain_min"]
        end_state_pass_overall = end_state_correct >= END_STATE_MIN
        print(
            f"\n  depth={depth} summary: "
            f"chain={chain_correct}/{n_cases} (gate ≥ {gates['chain_min']}: "
            f"{'PASS' if chain_pass_overall else 'FAIL'}), "
            f"end_state={end_state_correct}/{n_cases} (gate ≥ {END_STATE_MIN}: "
            f"{'PASS' if end_state_pass_overall else 'FAIL'})"
        )
        per_depth_records[depth] = {
            "depth": depth,
            "search_space_size": search_space,
            "n_cases": n_cases,
            "chain_correct": chain_correct,
            "end_state_correct": end_state_correct,
            "chain_min": gates["chain_min"],
            "label": gates["label"],
            "chain_pass": chain_pass_overall,
            "end_state_pass": end_state_pass_overall,
            "cases": case_records,
        }

    # ---- Compact summary table ----
    print("\n" + "=" * 78)
    print("SUMMARY  (chain recovery / end-state correctness vs depth)")
    print("=" * 78)
    print(f"  {'depth':>5}  {'search_space':>13}  {'chain':>10}  {'end-state':>10}  status")
    print("  " + "-" * 75)
    for depth in args.depths:
        r = per_depth_records[depth]
        chain_cell = f"{r['chain_correct']}/{r['n_cases']} (gate ≥{r['chain_min']})"
        end_cell = f"{r['end_state_correct']}/{r['n_cases']} (gate ≥{END_STATE_MIN})"
        if r["chain_pass"] and r["end_state_pass"]:
            status = "PASS"
        elif r["end_state_pass"]:
            status = "soft-FAIL (end-state OK)"
        else:
            status = "FAIL"
        print(
            f"  {depth:>5}  {r['search_space_size']:>13,}  "
            f"{chain_cell:>10}  {end_cell:>10}  {status}"
        )

    # ---- Acceptance ----
    print("\n" + "=" * 78)
    print("ACCEPTANCE CHECK (Task 1.12)")
    print("=" * 78)
    chain_all_pass = all(r["chain_pass"] for r in per_depth_records.values())
    end_state_all_pass = all(r["end_state_pass"] for r in per_depth_records.values())
    print(
        f"  Per-depth chain-recovery gates (varying by depth): "
        f"{'PASS' if chain_all_pass else 'FAIL'}"
    )
    print(
        f"  Per-depth end-state ≥ {END_STATE_MIN}/{n_cases}: "
        f"{'PASS' if end_state_all_pass else 'FAIL'}"
    )
    overall = chain_all_pass and end_state_all_pass
    print(f"\n→ Task 1.12: {'PASS' if overall else 'FAIL'}")
    if not chain_all_pass:
        print(
            "\n  Note: chain-recovery degradation at higher depth is the "
            "expected\n"
            "  failure mode — the planner is finding a different chain "
            "with similar\n"
            "  end-state. If end-state stays ≥ 5/6, the *answer* is correct;\n"
            "  it's the *path* that drifted. That's where Task 1.9's deferred "
            "value\n"
            "  function would help."
        )

    # ---- Save JSON ----
    payload = {
        "task": "1.12",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "library": sorted(operators.keys()),
        "n_ops": n_ops,
        "n_cases": n_cases,
        "beam_width": args.beam_width,
        "step_bonus": args.step_bonus,
        "depths": args.depths,
        "expected_chain": list(expected_chain),
        "per_depth": {str(d): per_depth_records[d] for d in args.depths},
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
