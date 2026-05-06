"""Path B demo — interactive Q&A grounded in a small fact corpus.

Project rule (revised 2026-05-06, see memory `feedback_no_llm_in_architecture`):
LMs are permitted at the FINAL surface-rendering layer ONLY. Brain
decides what's true; LM renders fluently; verifier gates output.

What this script does:
  1. Loads E5-large-v2 as encoder (frozen) — the brain's perception layer.
  2. Loads a small open-access LM (default Qwen 2.5 1.5B-Instruct) as
     the surface renderer — eval-only, frozen.
  3. Indexes a 30-fact corpus (hardcoded for the demo) by ψ.
  4. Drops to an interactive prompt: type a question, the system
     retrieves relevant facts, the LM renders an answer, the verifier
     gates the output. Full audit trail printed.

Run:
  python scripts/path_b_demo.py
  python scripts/path_b_demo.py --lm Qwen/Qwen2.5-0.5B-Instruct  # smaller
  python scripts/path_b_demo.py --query "What is a cat?"          # one-shot

VRAM estimate: ~1.3 GB (E5) + ~3 GB (Qwen 1.5B fp16) ≈ 4.5 GB.
For tighter budgets, use --lm Qwen/Qwen2.5-0.5B-Instruct (~1 GB).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selflearnai.memory import Corpus, Retriever
from selflearnai.renderer import LMRenderer, PathBPipeline

from scripts.stage1_planner_beam_smoke import ENCODERS, make_encode_fn


# ---------------------------------------------------------------------------
# 30-fact corpus — covers a few topics for the demo. Mix of clear facts
# and detail so retrieval has real ranking work to do.
# ---------------------------------------------------------------------------

FACTS: list[tuple[str, dict]] = [
    # animals (10)
    ("A cat is a small domesticated carnivorous mammal.",            {"topic": "animals"}),
    ("Cats are known for hunting mice and rodents.",                 {"topic": "animals"}),
    ("A dog is a domesticated descendant of the wolf.",              {"topic": "animals"}),
    ("Dogs are known as loyal companions to humans.",                {"topic": "animals"}),
    ("An eagle is a large bird of prey with sharp talons.",          {"topic": "animals"}),
    ("Eagles can spot small prey from very high altitudes.",         {"topic": "animals"}),
    ("A penguin is a flightless bird that lives in cold regions.",   {"topic": "animals"}),
    ("Penguins are excellent swimmers despite being birds.",         {"topic": "animals"}),
    ("Tigers are the largest cats in the cat family.",               {"topic": "animals"}),
    ("Tigers live primarily in the forests of Asia.",                {"topic": "animals"}),

    # geography (10)
    ("Paris is the capital city of France.",                         {"topic": "geography"}),
    ("Tokyo is the capital city of Japan and its largest city.",     {"topic": "geography"}),
    ("The Nile is one of the longest rivers in the world.",          {"topic": "geography"}),
    ("Mount Everest is the tallest mountain on Earth.",              {"topic": "geography"}),
    ("The Pacific is the largest ocean on Earth.",                   {"topic": "geography"}),
    ("Australia is both a country and a continent.",                 {"topic": "geography"}),
    ("The Sahara is the largest hot desert in the world.",           {"topic": "geography"}),
    ("Brazil is the largest country in South America.",              {"topic": "geography"}),
    ("Iceland is a Nordic island country in the North Atlantic.",    {"topic": "geography"}),
    ("The Amazon rainforest spans several countries in South America.", {"topic": "geography"}),

    # science / general (10)
    ("Water freezes at zero degrees Celsius at standard pressure.",  {"topic": "science"}),
    ("The Earth orbits the Sun once every 365.25 days.",             {"topic": "science"}),
    ("The human body has 206 bones in its skeleton.",                {"topic": "science"}),
    ("Photosynthesis is the process plants use to make food from sunlight.", {"topic": "science"}),
    ("Sound travels faster through water than through air.",         {"topic": "science"}),
    ("The Sun is a star at the center of our solar system.",         {"topic": "science"}),
    ("Lightning is an electrical discharge between clouds and the ground.", {"topic": "science"}),
    ("Gold is a chemical element with the symbol Au.",               {"topic": "science"}),
    ("Honey is produced by bees from the nectar of flowers.",        {"topic": "science"}),
    ("The Moon orbits the Earth roughly once every 27 days.",        {"topic": "science"}),
]


# ---------------------------------------------------------------------------
# Pretty-print one response with full audit trail.
# ---------------------------------------------------------------------------

def print_response(resp, *, show_prompt: bool = False) -> None:
    print()
    print("─" * 78)
    print(f"  query:           {resp.user_query}")
    print(f"  intent:          {resp.intent_kind}")

    # Brain admit diagnostics
    if resp.intent_kind == "factual_q":
        if resp.brain_refusal:
            print(f"  brain admit:     ✗ REFUSED  "
                  f"top={resp.brain_admit_top_score:.3f}  "
                  f"margin={resp.brain_admit_margin:+.3f}")
            print(f"  brain reason:    {resp.refusal_reason}")
        else:
            print(f"  brain admit:     ✓ admitted  "
                  f"top={resp.brain_admit_top_score:.3f}  "
                  f"margin={resp.brain_admit_margin:+.3f}")

    if resp.retrieved_facts:
        print(f"  retrieved ({len(resp.retrieved_facts)}):")
        for i, (fact, score) in enumerate(zip(resp.retrieved_facts, resp.retrieved_scores)):
            print(f"    {i+1}. [cos {score:.3f}] {fact}")
    elif resp.intent_kind == "factual_q":
        print(f"  retrieved:       (none — brain refused)")

    if show_prompt:
        print()
        print(f"  LM prompt sent:")
        for line in resp.lm_prompt.splitlines():
            print(f"    │ {line}")
    print()
    print(f"  LM raw output:   {resp.lm_output_raw!r}")
    print(f"  LM tokens:       in={resp.lm_n_input_tokens}  out={resp.lm_n_output_tokens}")

    # Three-axis verifier
    accept_mark = "✓" if resp.accepted else "✗"
    print(f"  verifier {accept_mark}:   "
          f"grounding {resp.grounding_cos:.3f} (≥{resp.grounding_threshold})  "
          f"relevance {resp.relevance_cos:.3f} (≥{resp.relevance_threshold})  "
          f"content {resp.content_overlap_rate:.3f} (≥{resp.content_threshold})")
    if resp.novel_content_words:
        novel_str = ", ".join(resp.novel_content_words[:8])
        if len(resp.novel_content_words) > 8:
            novel_str += f", +{len(resp.novel_content_words)-8} more"
        print(f"  novel words:     [{novel_str}]")
    if resp.verifier_failure_reason:
        print(f"  failure_reason:  {resp.verifier_failure_reason}")
    print()
    print(f"  ► response:      {resp.text}")
    print("─" * 78)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="e5-large-v2", choices=list(ENCODERS.keys()))
    parser.add_argument("--lm", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF model ID for the LM renderer (eval-only). "
                             "Smaller alternative: Qwen/Qwen2.5-0.5B-Instruct.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lm-dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--retrieval-k", type=int, default=3)
    parser.add_argument("--admit-threshold", type=float, default=0.55,
                        help="If best retrieval score < this, treat as no facts.")
    parser.add_argument("--grounding-threshold", type=float, default=0.65)
    parser.add_argument("--relevance-threshold", type=float, default=0.50)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--query", default=None,
                        help="If given: one-shot run with this query. Otherwise interactive.")
    parser.add_argument("--show-prompt", action="store_true",
                        help="Show the LM prompt in audit output.")
    parser.add_argument("--out", default="results/path_b/demo_session.jsonl",
                        help="Append every response (as JSON line) to this file.")
    args = parser.parse_args()

    print("Path B demo — brain (ψ-space) reasons, LM renders, verifier gates")
    print("=" * 78)

    # ---- Encoder (brain's perception) -------------------------------
    enc_cfg = ENCODERS[args.encoder]
    print(f"\n[1] Loading encoder: {enc_cfg['model']}")
    from transformers import AutoModel, AutoTokenizer
    enc_tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    enc_mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in enc_mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(enc_mdl, enc_tok, args.device)
    print(f"    encoder loaded ({enc_cfg['dim']}-dim, frozen)")

    # ---- Index the fact corpus (brain's memory) ---------------------
    print(f"\n[2] Indexing {len(FACTS)} facts as the brain's memory ...")
    texts = [t for t, _ in FACTS]
    metas = [m for _, m in FACTS]
    corpus = Corpus.build(texts, encode, metadata=metas, batch_size=32)
    retriever = Retriever(corpus, device=args.device)
    print(f"    corpus indexed ({len(corpus)} records)")

    # ---- LM renderer (eval-only) ------------------------------------
    print(f"\n[3] Loading LM renderer: {args.lm}")
    print(f"    (eval-only — never trained in this project)")
    renderer = LMRenderer(
        model_name=args.lm,
        device=args.device,
        dtype=args.lm_dtype,
    )
    print(f"    LM loaded ({renderer.n_params/1e6:.1f}M params, frozen)")

    # ---- Pipeline ----------------------------------------------------
    pipeline = PathBPipeline(
        encode_fn=encode,
        renderer=renderer,
        retriever=retriever,
        grounding_threshold=args.grounding_threshold,
        relevance_threshold=args.relevance_threshold,
        max_new_tokens=args.max_new_tokens,
        retrieval_k=args.retrieval_k,
        retrieval_admit_threshold=args.admit_threshold,
    )
    print(f"\n[4] Pipeline ready.")
    print(f"    grounding_threshold={args.grounding_threshold}  "
          f"relevance_threshold={args.relevance_threshold}  "
          f"admit_threshold={args.admit_threshold}")

    # ---- Output log file --------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(out_path, "a")

    def log_response(resp) -> None:
        log_handle.write(json.dumps(resp.to_dict()) + "\n")
        log_handle.flush()

    # ---- One-shot or interactive ------------------------------------
    if args.query is not None:
        resp = pipeline.respond(args.query)
        print_response(resp, show_prompt=args.show_prompt)
        log_response(resp)
        log_handle.close()
        return

    print("\n" + "=" * 78)
    print("Interactive mode. Type a question; type 'quit' / 'exit' to end.")
    print("Sample queries to try:")
    print("  - What is a cat?")
    print("  - What's the capital of France?")
    print("  - What is photosynthesis?")
    print("  - What is the largest desert?")
    print("  - Who is Einstein?           (out of corpus → should refuse)")
    print("  - What's 2 plus 2?            (out of corpus → should refuse)")
    print("=" * 78)

    try:
        while True:
            try:
                query = input("\n> ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in ("quit", "exit", ":q"):
                break
            try:
                resp = pipeline.respond(query)
            except Exception as e:
                print(f"  [error] {type(e).__name__}: {e}")
                continue
            print_response(resp, show_prompt=args.show_prompt)
            log_response(resp)
    finally:
        log_handle.close()
        print(f"\n→ session log saved to {out_path}")


if __name__ == "__main__":
    main()
