"""Task 1.6 — beam-search planner skeleton smoke test.

Trains the validated agentive + plural operators (from Stage 0
substrate), then runs the new BeamSearchPlanner at max_depth=2 on the
6 chain test cases from scripts/test_compositionality.py:

  paint  → painters
  drive  → drivers
  sing   → singers
  dance  → dancers
  run    → runners
  help   → helpers

For each case, the planner receives:
  - ψ_start = encode(verb)
  - ψ_goal  = encode(plural_agent)
  - operators = {agentive, plural}
and is expected to:
  1. Find chain == ("agentive", "plural") (chain-recovery test).
  2. Produce a final ψ whose argmax over the FAIR pool is the expected
     plural_agent (end-state-correctness test).

Two acceptance criteria, both gated:
  - Chain recovery ≥ 5/6.
  - End-state correctness ≥ 5/6.

Pure cosine-to-goal scoring; no learned heuristics yet (Tasks 1.7-1.9
add operator priors and value functions). With only 2 operators and
depth=2, the search space is tiny (4 length-2 chains); beam_width=4
explores it exhaustively.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.concepts import ConceptOperator
from selflearnai.planner import BeamSearchPlanner


# ---------------------------------------------------------------------------
# Data identical to scripts/test_compositionality.py so we can compare
# planner output to the validated 6/6 FAIR composition number.
# ---------------------------------------------------------------------------

CHAIN_TRIPLES = [
    ("paint", "painter", "painters"),
    ("drive", "driver",  "drivers"),
    ("sing",  "singer",  "singers"),
    ("dance", "dancer",  "dancers"),
    ("run",   "runner",  "runners"),
    ("help",  "helper",  "helpers"),
]

# FAIR pool from test_compositionality.py — single-step distractors removed.
POOL_FAIR = [
    "painters", "drivers", "singers", "dancers", "runners", "helpers",
    "writers", "builders", "teachers",
    "doctors", "lawyers", "bakers", "gardeners", "actors", "swimmers",
    "leaders", "workers", "speakers", "readers", "thinkers", "creators",
    "designers", "managers", "performers", "climbers", "fighters",
    "cats", "dogs", "trees", "books", "houses",
]

ENCODERS = {
    "gte-base":    {"model": "thenlper/gte-base",    "dim":  768},
    "e5-large-v2": {"model": "intfloat/e5-large-v2", "dim": 1024},
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _strip(s: str) -> str:
    idx = s.find("#")
    return (s[:idx] if idx >= 0 else s).strip()


def read_pairs(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                src, tgt = _strip(parts[0]), _strip(parts[1])
                if src and tgt:
                    out.append((src, tgt))
    return out


@torch.no_grad()
def make_encode_fn(model, tokenizer, device: str, max_length: int = 64):
    def encode(words: list[str]) -> torch.Tensor:
        inputs = tokenizer(
            words, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**inputs)
        last_hidden = out.last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.float()
    return encode


def train_operator(
    encode_fn, train_pairs, *, dim, device, seed, epochs=2000, lr=1e-3,
) -> ConceptOperator:
    torch.manual_seed(seed)
    op = ConceptOperator(dim=dim).to(device)
    opt = torch.optim.AdamW(op.parameters(), lr=lr)
    z_src = encode_fn([p[0] for p in train_pairs])
    z_tgt = encode_fn([p[1] for p in train_pairs])
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(op(z_src), z_tgt)
        loss.backward()
        opt.step()
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="gte-base", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument(
        "--chain-min", type=int, default=5,
        help="Acceptance: chain-recovery correct on >= this many of 6.",
    )
    parser.add_argument(
        "--end-state-min", type=int, default=5,
        help="Acceptance: end-state-correctness on >= this many of 6.",
    )
    parser.add_argument("--out", default="results/stage1/planner_beam_smoke.json")
    parser.add_argument(
        "--show-runner-up-k", type=int, default=3,
        help="For each test case, also print the top-K plans for diagnosis.",
    )
    args = parser.parse_args()

    enc_cfg = ENCODERS[args.encoder]
    print("Task 1.6 — beam-search planner skeleton smoke test")
    print("=" * 70)
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")
    print(f"beam_width={args.beam_width}, max_depth={args.max_depth}")

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Train operators ----
    print("\nTraining agentive operator ...")
    agentive_train = read_pairs(Path("data/few_shot/agentive/text_pairs_train.tsv"))
    print(f"  agentive: {len(agentive_train)} train pairs")
    op_agentive = train_operator(
        encode, agentive_train, dim=enc_cfg["dim"], device=args.device,
        seed=args.seed, epochs=args.epochs,
    )

    print("\nTraining plural operator ...")
    plural_train = read_pairs(Path("data/plurality/text_pairs_train.tsv"))
    print(f"  plural: {len(plural_train)} train pairs")
    op_plural = train_operator(
        encode, plural_train, dim=enc_cfg["dim"], device=args.device,
        seed=args.seed, epochs=args.epochs,
    )

    # ---- Build planner ----
    planner = BeamSearchPlanner(
        operators={"agentive": op_agentive, "plural": op_plural},
        beam_width=args.beam_width,
        max_depth=args.max_depth,
    )

    # ---- Encode the FAIR pool once (for end-state argmax) ----
    z_pool = encode(POOL_FAIR)
    pool_n = F.normalize(z_pool, dim=-1)

    # ---- Run planner on each chain test case ----
    print("\n" + "=" * 70)
    print(f"PER-CASE PLANNER RESULT  (depth={args.max_depth})")
    print("=" * 70)
    expected_chain = ("agentive", "plural")
    chain_correct = 0
    end_state_correct = 0
    case_records: list[dict] = []
    for verb, _, plural_agent in CHAIN_TRIPLES:
        psi_start = encode([verb]).squeeze(0)
        psi_goal = encode([plural_agent]).squeeze(0)

        result = planner.search(psi_start, psi_goal)

        # Argmax over FAIR pool for end-state correctness
        pred_n = F.normalize(result.psi.unsqueeze(0), dim=-1)
        sims = pred_n @ pool_n.T
        top1_idx = int(sims.argmax(dim=-1).item())
        top1_word = POOL_FAIR[top1_idx]
        end_state_pass = top1_word == plural_agent

        chain_pass = result.chain == expected_chain
        if chain_pass:
            chain_correct += 1
        if end_state_pass:
            end_state_correct += 1

        chain_repr = " ∘ ".join(result.chain) if result.chain else "(no-op)"
        c_mark = "✓" if chain_pass else "✗"
        e_mark = "✓" if end_state_pass else "✗"
        print(
            f"  {verb:<6}→ {plural_agent:<10}  "
            f"{c_mark} chain: {chain_repr:<22}  "
            f"{e_mark} top-1: {top1_word:<12}  "
            f"score={result.score:+.3f}"
        )

        # Top-K runners-up for transparency
        topk = planner.search_top_k(psi_start, psi_goal, k=args.show_runner_up_k)
        for r in topk[1:]:  # skip the best (already shown)
            r_repr = " ∘ ".join(r.chain) if r.chain else "(no-op)"
            print(f"        runner-up: {r_repr:<22}  score={r.score:+.3f}")

        case_records.append({
            "verb": verb,
            "expected_target": plural_agent,
            "chain": list(result.chain),
            "chain_correct": chain_pass,
            "top1_pool_word": top1_word,
            "end_state_correct": end_state_pass,
            "score": result.score,
        })

    # ---- Acceptance ----
    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK (Task 1.6)")
    print("=" * 70)
    chain_pass_overall = chain_correct >= args.chain_min
    end_state_pass_overall = end_state_correct >= args.end_state_min
    print(
        f"  Chain recovery: {chain_correct}/6  "
        f"(gate >= {args.chain_min})  → {'PASS' if chain_pass_overall else 'FAIL'}"
    )
    print(
        f"  End-state correctness: {end_state_correct}/6  "
        f"(gate >= {args.end_state_min})  → {'PASS' if end_state_pass_overall else 'FAIL'}"
    )
    overall = chain_pass_overall and end_state_pass_overall
    print(f"\n→ Task 1.6: {'PASS' if overall else 'FAIL'}")

    # ---- Save JSON ----
    payload = {
        "task": "1.6",
        "encoder": args.encoder,
        "encoder_dim": enc_cfg["dim"],
        "beam_width": args.beam_width,
        "max_depth": args.max_depth,
        "n_cases": len(CHAIN_TRIPLES),
        "chain_correct": chain_correct,
        "end_state_correct": end_state_correct,
        "chain_min": args.chain_min,
        "end_state_min": args.end_state_min,
        "cases": case_records,
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
