"""§20.0 v2 — KNOWN-CONCEPT REDISCOVERY via real planner failures.

v1 (`scripts/stage20_0_wake_sleep_diagnostic.py`, commit 40d58e6) was
inconclusive: the absolute solve gate (cos_to_goal ≥ 0.85) was too
lenient for the encoder's natural geometry, so all 20 tasks "solved"
without any operator effort, leaving the wake buffer empty and never
testing wake-sleep at all. Recurring pattern from memory
`feedback_relative_gates_in_encoder_space`.

v2 fix: cleaner methodology. REMOVE a concept we know works (`plural`)
from the baseline operator library and check whether wake-sleep
RE-DISCOVERS it from REAL planner failures. Mirror of §19.10's smoke
(scripts/stage1_5_sleep_smoke.py) but driven by real planner output
instead of synthetic plants.

Two methodology fixes vs v1:
  1. RELATIVE solve gate: a task is solved only if applying an operator
     lifts cos_to_goal by ≥ 0.05 over no-op baseline. Catches the
     "encoder already there, no op needed" case.
  2. KNOWN-CONCEPT GAP: the gap is plural (we know plural works as a
     ConceptOperator from RESULTS.md baselines) — so any failure to
     rediscover it cleanly isolates "real failures vs synthetic" as
     the variable.

Setup:

  Baseline operator library (4 concepts — plural REMOVED):
    {past_tense, opposite, comparative, agentive}

  TRAINING task stream (16 chain tasks):
    Positive controls (4) — solvable with the 4 baseline ops:
       happy→unhappy, kind→unkind (opposite)
       walk→walked, jump→jumped (past_tense)

    GAP tasks (12) — require plural, which is missing:
       cat→cats, dog→dogs, hat→hats, bag→bags, cup→cups,
       log→logs, rat→rats, sun→suns, fan→fans, kid→kids,
       bird→birds, tree→trees

  HELD-OUT tasks (6 NEW plural pairs from data/plurality/text_pairs_held_out.tsv):
    book→books, pig→pigs, key→keys, lamp→lamps, party→parties, match→matches

Process:
  1. Train 4 baseline operators (plural NOT included).
  2. Run BeamSearchPlanner on training tasks.
     A task is "solved" if cos_to_goal - cos_no_op ≥ 0.05 (relative gate).
     Plural tasks SHOULD fail this — planner has no plural op.
  3. Plant failures into wake buffer.
  4. Run sleep cycle.
  5. Held-out: planner WITHOUT discovered concept vs WITH discovered concept,
     count tasks where relative-improvement gate is met.

Verdicts:
  DISCOVERED_AND_USEFUL    — ≥ 1 concept registered AND held-out lift ≥ 1.
  DISCOVERED_BUT_NOT_USEFUL — concept registered but no held-out lift.
  NOT_DISCOVERED           — sleep didn't propose, or all proposals failed.

If DISCOVERED_AND_USEFUL: §20 curriculum premise validated. The wake-sleep
mechanism scales from synthetic to real-failure data.

Run on the GPU box (~5–10 min):
  python scripts/stage20_0b_plural_rediscovery.py
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
import torch.nn.functional as F

from selflearnai.discovery import (
    ConceptRegistry,
    SleepConfig,
    WakeBuffer,
    run_sleep_cycle,
)
from selflearnai.planner import BeamSearchPlanner, PsiProgram

from scripts.stage1_planner_beam_smoke import (
    ENCODERS, make_encode_fn, read_pairs, train_operator,
)


# ---------------------------------------------------------------------------
# Baseline operator library — plural DELIBERATELY OMITTED.
# ---------------------------------------------------------------------------

BASELINE_CONCEPTS_DATA: list[tuple[str, str]] = [
    ("past_tense",   "data/past_tense"),
    ("opposite",     "data/opposite_v2"),
    ("comparative",  "data/comparative"),
    ("agentive",     "data/few_shot/agentive"),
]


# ---------------------------------------------------------------------------
# Task streams
# ---------------------------------------------------------------------------

# Positive controls — solvable with the 4 baseline operators.
POSITIVE_CONTROL_TASKS: list[tuple[str, str, str]] = [
    ("happy", "unhappy", "opposite"),
    ("kind",  "unkind",  "opposite"),
    ("walk",  "walked",  "past_tense"),
    ("jump",  "jumped",  "past_tense"),
]

# GAP tasks — require plural, which is NOT in the baseline library.
# These 12 are drawn from the first 12 of data/plurality/text_pairs_train.tsv
# so we can confirm them as real plural pairs the encoder DOES have signal on.
GAP_TASKS: list[tuple[str, str]] = [
    ("cat",   "cats"),
    ("dog",   "dogs"),
    ("hat",   "hats"),
    ("bag",   "bags"),
    ("cup",   "cups"),
    ("log",   "logs"),
    ("rat",   "rats"),
    ("sun",   "suns"),
    ("fan",   "fans"),
    ("kid",   "kids"),
    ("bird",  "birds"),
    ("tree",  "trees"),
]

# HELD-OUT — taken from data/plurality/text_pairs_held_out.tsv. NEVER in
# GAP_TASKS or in any operator's training data (these were exactly the
# pairs Stage 0 used as held-out for the plural concept).
HELD_OUT_TASKS: list[tuple[str, str]] = [
    ("book",  "books"),
    ("pig",   "pigs"),
    ("key",   "keys"),
    ("lamp",  "lamps"),
    ("party", "parties"),
    ("match", "matches"),
]


def assert_no_overlap() -> None:
    train_set = set(GAP_TASKS) | {(s, t) for s, t, _ in POSITIVE_CONTROL_TASKS}
    overlap = [p for p in HELD_OUT_TASKS if p in train_set]
    if overlap:
        print(f"FATAL: held-out leaks into training: {overlap}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Planner driver — relative-improvement solve gate
# ---------------------------------------------------------------------------

def make_failed_program(source: str, target: str, dim: int,
                         psi_start: torch.Tensor, psi_goal: torch.Tensor,
                         encoder_name: str) -> PsiProgram:
    return PsiProgram(
        source=source, target=target,
        encoder_name=encoder_name, encoder_dim=dim,
        psi_initial=psi_start.detach().cpu().tolist(),
        psi_goal=psi_goal.detach().cpu().tolist(),
        chain=[], steps=[],
        final_top1_word=None, final_cos_to_goal=None,
        timestamp=PsiProgram.now_timestamp(),
        metadata={"diagnostic": "20.0b", "channel": "failed"},
    )


def run_planner_on_stream(
    planner: BeamSearchPlanner,
    encode_fn,
    tasks: list[tuple[str, str]],
    *,
    relative_gate: float,
) -> list[dict]:
    """Run planner on each task with RELATIVE solve gate.

    A task is 'solved' if final_cos_to_goal - baseline_cos >= relative_gate.
    The baseline_cos is cos(encode(source), encode(target)) with NO operator
    applied — what the encoder gives for free."""
    records = []
    for source, target in tasks:
        psi_src = encode_fn([source]).squeeze(0).flatten()
        psi_tgt = encode_fn([target]).squeeze(0).flatten()
        baseline_cos = float(F.cosine_similarity(
            psi_src.unsqueeze(0), psi_tgt.unsqueeze(0), dim=-1).item())
        result = planner.search(psi_src, psi_tgt)
        improvement = result.cos_to_goal - baseline_cos
        solved = improvement >= relative_gate
        records.append({
            "source": source,
            "target": target,
            "psi_src": psi_src,
            "psi_tgt": psi_tgt,
            "baseline_cos": baseline_cos,
            "chain": list(result.chain),
            "final_cos": result.cos_to_goal,
            "improvement": improvement,
            "solved": solved,
        })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-epochs", type=int, default=2000)
    parser.add_argument("--operator-epochs", type=int, default=2000)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--relative-gate", type=float, default=0.05,
                        help="Solve metric: cos_to_goal - baseline_cos >= this. "
                             "Per memory feedback_relative_gates_in_encoder_space — "
                             "absolute thresholds fail when encoder geometry dominates.")
    parser.add_argument("--out", default="results/curriculum/wake_sleep_diagnostic_v2.json")
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("§20.0b Foundation diagnostic v2 — known-concept rediscovery")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Solve gate (RELATIVE): cos_to_goal - baseline_cos ≥ {args.relative_gate}")
    print(f"Beam: width={args.beam_width}  max_depth={args.max_depth}")
    print(f"Strategy: REMOVE 'plural' from baseline; see if sleep rediscovers it.")

    assert_no_overlap()

    # ---- Encoder ------------------------------------------------------
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)
    DIM = enc_cfg["dim"]

    # ---- Train baseline operator library (plural OMITTED) ------------
    print(f"\n[1] Training {len(BASELINE_CONCEPTS_DATA)} baseline operators "
          f"(plural DELIBERATELY excluded)")
    print("-" * 78)
    baseline_operators: dict[str, object] = {}
    for concept, ddir in BASELINE_CONCEPTS_DATA:
        path = Path(ddir) / "text_pairs_train.tsv"
        if not path.exists():
            print(f"  [SKIP] {concept}: no data at {path}")
            continue
        train_pairs = read_pairs(path)
        op = train_operator(
            encode, train_pairs,
            dim=DIM, device=args.device,
            seed=args.seed, epochs=args.baseline_epochs,
        )
        baseline_operators[concept] = op
        print(f"  ✓ {concept:<14} ({len(train_pairs)} pairs)")
    print(f"  baseline library: {sorted(baseline_operators.keys())}")

    # ---- Build planner -----------------------------------------------
    planner_baseline = BeamSearchPlanner(
        operators=baseline_operators,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        step_bonus=0.01,
    )

    # ---- Run planner on training stream ------------------------------
    all_train_tasks = (
        [(s, t) for s, t, _ in POSITIVE_CONTROL_TASKS] + GAP_TASKS
    )
    print(f"\n[2] Run planner on {len(all_train_tasks)} training tasks "
          f"({len(POSITIVE_CONTROL_TASKS)} positive + {len(GAP_TASKS)} gap)")
    print("-" * 78)
    train_records = run_planner_on_stream(
        planner_baseline, encode, all_train_tasks,
        relative_gate=args.relative_gate,
    )
    n_solved_train = sum(1 for r in train_records if r["solved"])
    print(f"  solved (relative gate ≥ {args.relative_gate}): "
          f"{n_solved_train}/{len(train_records)}")
    print(f"  per-task verdicts:")
    for r in train_records:
        mark = "✓" if r["solved"] else "✗"
        chain_str = "→".join(r["chain"]) if r["chain"] else "(no-op)"
        print(f"    {mark} {r['source']:>10} → {r['target']:<14}  "
              f"cos={r['final_cos']:.3f} (baseline {r['baseline_cos']:.3f}, "
              f"lift {r['improvement']:+.3f})  chain={chain_str}")

    # ---- Plant failures into wake buffer -----------------------------
    tmp_root = Path(tempfile.mkdtemp(prefix="diag_20_0b_"))
    buffer_root = tmp_root / "wake"
    registry_root = tmp_root / "registry"
    buf = WakeBuffer(buffer_root)
    n_planted = 0
    for r in train_records:
        if not r["solved"]:
            program = make_failed_program(
                r["source"], r["target"], DIM,
                r["psi_src"], r["psi_tgt"],
                encoder_name=args.encoder,
            )
            buf.append(
                program,
                channel="failed",
                routing_reason=(
                    f"relative-gate fail: improvement {r['improvement']:+.3f} "
                    f"< gate {args.relative_gate}"
                ),
                routing_metadata={
                    "diagnostic": "20.0b",
                    "improvement": r["improvement"],
                    "final_cos": r["final_cos"],
                    "baseline_cos": r["baseline_cos"],
                },
            )
            n_planted += 1
    print(f"\n[3] Planted {n_planted} failures in wake buffer "
          f"(channel='failed')")

    # ---- Sleep cycle on real residuals -------------------------------
    print("\n" + "=" * 78)
    print("[4] RUN SLEEP CYCLE on real planner residuals")
    print("=" * 78)
    registry = ConceptRegistry(registry_root, dim=DIM, max_active=10)
    config = SleepConfig(
        operator_epochs=args.operator_epochs,
        operator_seed=args.seed,
    )
    sleep_result = run_sleep_cycle(
        buf, registry,
        encode_fn=encode,
        operators=baseline_operators,
        macro_holdout_tasks=(),
        dim=DIM,
        device=args.device,
        config=config,
        cycle_tag="diag_20_0b",
    )

    print(f"  unexplained entries:  {sleep_result.n_unexplained_entries}")
    print(f"  clusters proposed:    {sleep_result.n_clusters_proposed}")
    print(f"  clusters consistent:  {sleep_result.n_clusters_consistent}")
    print(f"  clusters validated:   {sleep_result.n_clusters_validated}")
    print(f"  concepts registered:  {sleep_result.n_concepts_registered}")
    print(f"  registered ids:       {sleep_result.registered_concept_ids}")

    print(f"\n  per-cluster audit:")
    for row in sleep_result.cluster_audit:
        mark_c = "✓" if row.consistency_passed else "✗"
        mark_v = "✓" if row.validation_passed else "✗"
        print(f"    cluster {row.cluster_id}: n={row.n_members}  "
              f"{mark_c} consistency mean={row.consistency_mean:.3f} "
              f"var={row.consistency_var:.4f}  "
              f"{mark_v} validation  registered_as={row.registered_as}")
        for fc in row.failing_criteria:
            print(f"      · {fc}")

    # ---- Held-out evaluation -----------------------------------------
    print("\n" + "=" * 78)
    print(f"[5] HELD-OUT EVAL ({len(HELD_OUT_TASKS)} novel plural pairs)")
    print("=" * 78)

    print(f"\n  baseline planner (no plural, no discovered concept):")
    held_baseline = run_planner_on_stream(
        planner_baseline, encode, HELD_OUT_TASKS,
        relative_gate=args.relative_gate,
    )
    n_baseline_solved = sum(1 for r in held_baseline if r["solved"])
    for r in held_baseline:
        m = "✓" if r["solved"] else "✗"
        print(f"    {m} {r['source']:>8} → {r['target']:<10}  "
              f"cos={r['final_cos']:.3f} "
              f"(baseline {r['baseline_cos']:.3f}, lift {r['improvement']:+.3f})")
    print(f"  baseline solved: {n_baseline_solved}/{len(HELD_OUT_TASKS)}")

    n_treatment_solved = n_baseline_solved
    held_treatment = held_baseline
    if sleep_result.registered_concept_ids:
        treated_operators = dict(baseline_operators)
        for cid in sleep_result.registered_concept_ids:
            try:
                op = registry.get_op(cid).to(args.device)
                op.eval()
                for p in op.parameters():
                    p.requires_grad_(False)
                treated_operators[cid] = op
            except Exception as e:
                print(f"  WARNING: could not load discovered op {cid}: {e}")
        planner_treatment = BeamSearchPlanner(
            operators=treated_operators,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            step_bonus=0.01,
        )
        print(f"\n  treatment planner (now with discovered concept):")
        held_treatment = run_planner_on_stream(
            planner_treatment, encode, HELD_OUT_TASKS,
            relative_gate=args.relative_gate,
        )
        n_treatment_solved = sum(1 for r in held_treatment if r["solved"])
        for r in held_treatment:
            m = "✓" if r["solved"] else "✗"
            chain_str = "→".join(r["chain"]) if r["chain"] else "(no-op)"
            print(f"    {m} {r['source']:>8} → {r['target']:<10}  "
                  f"cos={r['final_cos']:.3f} "
                  f"(baseline {r['baseline_cos']:.3f}, lift {r['improvement']:+.3f})  "
                  f"chain={chain_str}")
        print(f"  treatment solved: {n_treatment_solved}/{len(HELD_OUT_TASKS)}")
    else:
        print("\n  no concepts registered → no treatment to compare")

    held_lift = n_treatment_solved - n_baseline_solved
    print(f"\n  held-out lift: {held_lift:+d} tasks")

    # ---- Verdict -----------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if (sleep_result.n_concepts_registered >= 1) and (held_lift >= 1):
        verdict = "DISCOVERED_AND_USEFUL"
        message = (
            f"Wake-sleep RE-DISCOVERED plural from {n_planted} real planner "
            f"failures. {sleep_result.n_concepts_registered} concept(s) "
            f"registered AND held-out lift = {held_lift:+d} task(s). "
            f"§20 curriculum premise validated: wake-sleep scales from "
            f"synthetic (§19.10 smoke) to real-failure data without "
            f"further mechanism. Proceed to revisit §20 (narrower scope: "
            f"Stages A + B only, drop C/D/E to v2/CSIL)."
        )
    elif sleep_result.n_concepts_registered >= 1 and held_lift <= 0:
        verdict = "DISCOVERED_BUT_NOT_USEFUL"
        message = (
            f"Wake-sleep registered {sleep_result.n_concepts_registered} "
            f"concept(s) but adding them did NOT lift held-out plural "
            f"performance (lift {held_lift:+d}). Possibilities: "
            f"(a) discovered operator overfit its training cluster, "
            f"(b) held-out plural pairs (book/pig/key/lamp/party/match) "
            f"have different ψ-shift geometry than the gap-task pairs, "
            f"(c) three-criterion gate is too lenient. Tighten before "
            f"any curriculum work."
        )
    else:
        verdict = "NOT_DISCOVERED"
        message = (
            f"Wake-sleep did NOT propose validated concepts even on "
            f"genuinely-failed plural tasks. clusters_proposed="
            f"{sleep_result.n_clusters_proposed} "
            f"clusters_consistent={sleep_result.n_clusters_consistent} "
            f"clusters_validated={sleep_result.n_clusters_validated}. "
            f"Critical: this is the SAME concept §19.10 rediscovered "
            f"from synthetic plants — so failure here pinpoints the "
            f"REAL-FAILURES path as the broken variable, not wake-sleep "
            f"itself. Investigate: what's different about real planner "
            f"output vs synthetic plants in the WakeBuffer entries?"
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ---------------------------------------------------
    payload = {
        "task": "20.0b",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "relative_gate": args.relative_gate,
        "n_baseline_concepts": len(baseline_operators),
        "baseline_concepts": sorted(baseline_operators.keys()),
        "removed_concept": "plural",
        "n_train_tasks": len(train_records),
        "n_train_solved": n_solved_train,
        "n_planted_failures": n_planted,
        "sleep_result": {
            "n_unexplained_entries": sleep_result.n_unexplained_entries,
            "n_clusters_proposed": sleep_result.n_clusters_proposed,
            "n_clusters_consistent": sleep_result.n_clusters_consistent,
            "n_clusters_validated": sleep_result.n_clusters_validated,
            "n_concepts_registered": sleep_result.n_concepts_registered,
            "registered_concept_ids": sleep_result.registered_concept_ids,
            "cluster_audit": [
                {
                    "cluster_id": r.cluster_id,
                    "n_members": r.n_members,
                    "consistency_passed": r.consistency_passed,
                    "consistency_mean": r.consistency_mean,
                    "consistency_var": r.consistency_var,
                    "validation_passed": r.validation_passed,
                    "registered_as": r.registered_as,
                    "failing_criteria": r.failing_criteria,
                }
                for r in sleep_result.cluster_audit
            ],
        },
        "held_out": {
            "n_held_out": len(HELD_OUT_TASKS),
            "n_baseline_solved": n_baseline_solved,
            "n_treatment_solved": n_treatment_solved,
            "lift": held_lift,
            "baseline_records": [
                {"source": r["source"], "target": r["target"],
                 "final_cos": r["final_cos"], "baseline_cos": r["baseline_cos"],
                 "improvement": r["improvement"], "solved": r["solved"],
                 "chain": r["chain"]}
                for r in held_baseline
            ],
            "treatment_records": [
                {"source": r["source"], "target": r["target"],
                 "final_cos": r["final_cos"], "baseline_cos": r["baseline_cos"],
                 "improvement": r["improvement"], "solved": r["solved"],
                 "chain": r["chain"]}
                for r in held_treatment
            ],
        },
        "training_records": [
            {"source": r["source"], "target": r["target"],
             "final_cos": r["final_cos"], "baseline_cos": r["baseline_cos"],
             "improvement": r["improvement"], "solved": r["solved"],
             "chain": r["chain"]}
            for r in train_records
        ],
        "verdict": verdict,
        "message": message,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"→ cleaned up {tmp_root}")
    else:
        print(f"→ kept tmp at {tmp_root}")

    raise SystemExit(0 if verdict == "DISCOVERED_AND_USEFUL" else 1)


if __name__ == "__main__":
    main()
