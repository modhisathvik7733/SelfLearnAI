"""§20.0 Foundation diagnostic — does Stage 1.5 wake-sleep discover useful
concepts when driven by REAL planner failures (not synthetic residuals)?

Stage 1.5's smoke (§19.10, scripts/stage1_5_sleep_smoke.py) validated the
sleep cycle with hand-planted residuals. This diagnostic feeds REAL
planner failures from a chain task stream. The whole §20 curriculum
design assumes wake-sleep discovers useful concepts in this regime; if
it doesn't, no curriculum scaffolding helps.

Setup:

  Baseline operator library (the planner's tools at the start):
    {plural, past_tense, opposite, comparative, agentive,
     superlative, young, definitional}     ← all 8 existing concepts

  TRAINING task stream (20 chain tasks):
    Positive controls (8)  — solvable with existing ops:
       paint→painters, drive→drivers, sing→singers, dance→dancers
       (agentive ∘ plural)
       cat→cats, box→boxes (plural)
       write→writer, build→builder (agentive)

    GAP tasks (12) — require a concept the registry DOESN'T have:
       negation prefix (un-/in-): happy→unhappy, kind→unkind,
         fair→unfair, friendly→unfriendly, wise→unwise, true→untrue,
         common→uncommon, lucky→unlucky, sane→insane, finite→infinite,
         direct→indirect, formal→informal

  Process:
    1. Run BeamSearchPlanner on the 20 training tasks.
       Tasks that fail (cos_to_goal < 0.85) → wake buffer's "failed" channel.
    2. Run run_sleep_cycle() on the populated buffer.
       Inspect what came out: clusters proposed, consistency, validation,
       registered concept(s).

  HELD-OUT task stream (4 NEW negation pairs, NEVER in training):
    locked→unlocked, sure→unsure, healthy→unhealthy, just→unjust

    Acceptance:
      Run BeamSearchPlanner on held-out tasks BEFORE adding the discovered
      concept (baseline) and AFTER (treatment). Count tasks where
      cos_to_goal ≥ 0.85 (solved). Held-out solved count must improve
      by ≥ 1 task (relative-improvement gate, per memory
      `feedback_relative_gates_in_encoder_space`).

Verdicts:
  DISCOVERED_AND_USEFUL    — ≥ 1 concept registered AND held-out lift ≥ 1.
  DISCOVERED_BUT_NOT_USEFUL — concept registered but no held-out lift.
  NOT_DISCOVERED           — sleep didn't propose, or all proposals failed.

Run on the GPU box (~5–10 min on e5-large-v2):
  python scripts/stage20_0_wake_sleep_diagnostic.py
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
# Baseline operator library — all 8 existing concepts.
# Each entry: (concept_name, data_dir).
# ---------------------------------------------------------------------------

BASELINE_CONCEPTS_DATA: list[tuple[str, str]] = [
    ("plural",       "data/plurality"),
    ("past_tense",   "data/past_tense"),
    ("opposite",     "data/opposite_v2"),
    ("comparative",  "data/comparative"),
    ("agentive",     "data/few_shot/agentive"),
    ("superlative",  "data/superlative"),
    ("young",        "data/young"),
    ("definitional", "data/definitional"),
]


# ---------------------------------------------------------------------------
# Task streams
# ---------------------------------------------------------------------------

# Positive controls: solvable with the 8 baseline operators.
POSITIVE_CONTROL_TASKS: list[tuple[str, str, str]] = [
    # (source, target, expected_chain_summary)
    ("paint",  "painters",  "agentive∘plural"),
    ("drive",  "drivers",   "agentive∘plural"),
    ("sing",   "singers",   "agentive∘plural"),
    ("dance",  "dancers",   "agentive∘plural"),
    ("cat",    "cats",      "plural"),
    ("box",    "boxes",     "plural"),
    ("write",  "writer",    "agentive"),
    ("build",  "builder",   "agentive"),
]

# Gap tasks: require a NEGATION-PREFIX concept (un-/in-) the registry
# does NOT have. 12 pairs to ensure cluster size ≥ min_cluster_size=5.
GAP_TASKS: list[tuple[str, str]] = [
    ("happy",     "unhappy"),
    ("kind",      "unkind"),
    ("fair",      "unfair"),
    ("friendly",  "unfriendly"),
    ("wise",      "unwise"),
    ("true",      "untrue"),
    ("common",    "uncommon"),
    ("lucky",     "unlucky"),
    ("sane",      "insane"),
    ("finite",    "infinite"),
    ("direct",    "indirect"),
    ("formal",    "informal"),
]

# HELD-OUT: 4 new negation pairs, NEVER in GAP_TASKS.
HELD_OUT_TASKS: list[tuple[str, str]] = [
    ("locked",   "unlocked"),
    ("sure",     "unsure"),
    ("healthy",  "unhealthy"),
    ("just",     "unjust"),
]


def assert_no_overlap() -> None:
    """Fatal if held-out pairs leak into training stream."""
    train_set = set(GAP_TASKS) | {(s, t) for s, t, _ in POSITIVE_CONTROL_TASKS}
    overlap = [p for p in HELD_OUT_TASKS if p in train_set]
    if overlap:
        print(f"FATAL: held-out leaks into training: {overlap}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Planner driver — runs one task, returns (solved, plan_state, residual_psi)
# ---------------------------------------------------------------------------

def make_failed_program(source: str, target: str, dim: int,
                         psi_start: torch.Tensor, psi_goal: torch.Tensor,
                         encoder_name: str) -> PsiProgram:
    """Build a PsiProgram for a failed planner attempt. Sleep only reads
    program.source and program.target — psi fields are populated for
    correctness even though they're not used by the discovery path."""
    return PsiProgram(
        source=source,
        target=target,
        encoder_name=encoder_name,
        encoder_dim=dim,
        psi_initial=psi_start.detach().cpu().tolist(),
        psi_goal=psi_goal.detach().cpu().tolist(),
        chain=[],            # planner found no useful chain → empty
        steps=[],
        final_top1_word=None,
        final_cos_to_goal=None,
        timestamp=PsiProgram.now_timestamp(),
        metadata={"diagnostic": "20.0", "channel": "failed"},
    )


def run_planner_on_stream(
    planner: BeamSearchPlanner,
    encode_fn,
    tasks: list[tuple[str, str]],
    *,
    solve_threshold: float,
) -> list[dict]:
    """Run planner on each task. Return per-task records."""
    records = []
    for source, target in tasks:
        psi_src = encode_fn([source]).squeeze(0).flatten()
        psi_tgt = encode_fn([target]).squeeze(0).flatten()
        baseline_cos = float(F.cosine_similarity(
            psi_src.unsqueeze(0), psi_tgt.unsqueeze(0), dim=-1).item())
        result = planner.search(psi_src, psi_tgt)
        solved = result.cos_to_goal >= solve_threshold
        records.append({
            "source": source,
            "target": target,
            "psi_src": psi_src,
            "psi_tgt": psi_tgt,
            "baseline_cos": baseline_cos,         # cos before any op applied
            "chain": list(result.chain),
            "final_cos": result.cos_to_goal,
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
    parser.add_argument("--solve-threshold", type=float, default=0.85,
                        help="cos_to_goal at which we call a task solved (RESULTS.md baseline)")
    parser.add_argument("--out", default="results/curriculum/wake_sleep_diagnostic.json")
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("§20.0 Foundation diagnostic — wake-sleep on real planner failures")
    print("=" * 78)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"Solve threshold: cos_to_goal ≥ {args.solve_threshold}")
    print(f"Beam: width={args.beam_width}  max_depth={args.max_depth}")

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

    # ---- Train baseline operator library -----------------------------
    print(f"\n[1] Training {len(BASELINE_CONCEPTS_DATA)} baseline operators "
          f"@ {args.baseline_epochs} epochs each")
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
        step_bonus=0.01,           # Task 1.7 baseline
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
        solve_threshold=args.solve_threshold,
    )
    n_solved_train = sum(1 for r in train_records if r["solved"])
    print(f"  solved: {n_solved_train}/{len(train_records)}")
    print(f"  per-task verdicts:")
    for r in train_records:
        mark = "✓" if r["solved"] else "✗"
        chain_str = "→".join(r["chain"]) if r["chain"] else "(no-op)"
        print(f"    {mark} {r['source']:>10} → {r['target']:<14}  "
              f"cos={r['final_cos']:.3f} (baseline {r['baseline_cos']:.3f})  "
              f"chain={chain_str}")

    # ---- Plant failures into wake buffer -----------------------------
    tmp_root = Path(tempfile.mkdtemp(prefix="diag_20_0_"))
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
                routing_reason="planner could not reach goal under baseline library",
                routing_metadata={"diagnostic": "20.0", "final_cos": r["final_cos"]},
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
        cycle_tag="diag_20_0",
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
    print(f"[5] HELD-OUT EVAL ({len(HELD_OUT_TASKS)} novel negation pairs)")
    print("=" * 78)

    # 5a. Baseline (without discovered concept)
    print(f"\n  baseline planner (no discovered concept):")
    held_baseline = run_planner_on_stream(
        planner_baseline, encode, HELD_OUT_TASKS,
        solve_threshold=args.solve_threshold,
    )
    n_baseline_solved = sum(1 for r in held_baseline if r["solved"])
    for r in held_baseline:
        m = "✓" if r["solved"] else "✗"
        print(f"    {m} {r['source']:>10} → {r['target']:<12}  "
              f"cos={r['final_cos']:.3f} (baseline {r['baseline_cos']:.3f})")
    print(f"  baseline solved: {n_baseline_solved}/{len(HELD_OUT_TASKS)}")

    # 5b. Treatment (with discovered concept added)
    n_treatment_solved = n_baseline_solved
    held_treatment = held_baseline       # default if no concept registered
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
        print(f"\n  treatment planner ({len(treated_operators)} operators "
              f"including discovered):")
        held_treatment = run_planner_on_stream(
            planner_treatment, encode, HELD_OUT_TASKS,
            solve_threshold=args.solve_threshold,
        )
        n_treatment_solved = sum(1 for r in held_treatment if r["solved"])
        for r in held_treatment:
            m = "✓" if r["solved"] else "✗"
            chain_str = "→".join(r["chain"]) if r["chain"] else "(no-op)"
            print(f"    {m} {r['source']:>10} → {r['target']:<12}  "
                  f"cos={r['final_cos']:.3f}  chain={chain_str}")
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
            f"Wake-sleep discovered {sleep_result.n_concepts_registered} "
            f"concept(s) AND adding them lifted held-out solved count "
            f"by {held_lift:+d} task(s). Curriculum premise sound — the "
            f"central mechanism (wake-sleep on real failures → useful "
            f"new operators) WORKS at this scale. Proceed to revisit §20 "
            f"(narrower scope: Stage A + B only)."
        )
    elif sleep_result.n_concepts_registered >= 1 and held_lift <= 0:
        verdict = "DISCOVERED_BUT_NOT_USEFUL"
        message = (
            f"Wake-sleep registered {sleep_result.n_concepts_registered} "
            f"concept(s) but adding them did NOT lift held-out performance "
            f"(lift {held_lift:+d}). Possibilities: (a) three-criterion "
            f"gate is too lenient, (b) discovered operator overfit to its "
            f"training cluster, (c) held-out distribution drifts from "
            f"training cluster geometry. Tighten the gate before any "
            f"curriculum work."
        )
    else:
        verdict = "NOT_DISCOVERED"
        message = (
            f"Wake-sleep did NOT propose validated concepts on real "
            f"planner failures. clusters_proposed={sleep_result.n_clusters_proposed} "
            f"clusters_consistent={sleep_result.n_clusters_consistent} "
            f"clusters_validated={sleep_result.n_clusters_validated}. "
            f"This means wake-sleep does NOT scale from synthetic to "
            f"real-failure data without further work. Curriculum design "
            f"breaks. Investigate: (a) cluster quality on real ψ-shifts, "
            f"(b) consistency-gate threshold for noisy data, (c) maybe "
            f"a different discovery mechanism is needed."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ---------------------------------------------------
    payload = {
        "task": "20.0",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "solve_threshold": args.solve_threshold,
        "n_baseline_concepts": len(baseline_operators),
        "baseline_concepts": sorted(baseline_operators.keys()),
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
                 "final_cos": r["final_cos"], "solved": r["solved"],
                 "chain": r["chain"]}
                for r in held_baseline
            ],
            "treatment_records": [
                {"source": r["source"], "target": r["target"],
                 "final_cos": r["final_cos"], "solved": r["solved"],
                 "chain": r["chain"]}
                for r in held_treatment
            ],
        },
        "training_records": [
            {"source": r["source"], "target": r["target"],
             "final_cos": r["final_cos"], "solved": r["solved"],
             "chain": r["chain"], "baseline_cos": r["baseline_cos"]}
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
