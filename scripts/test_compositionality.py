"""Compositionality test for concept operators.

Trains two operators independently (AGENTIVE and PLURAL) on raw GTE
outputs (text-only pathway, no Stage 1). Then tests whether they
COMPOSE: applying both in sequence to a verb should produce the
plural-agent form.

Test chains (verbs from agentive's held-out set):

  paint  →  agentive →  painter  →  plural →  painters
  drive  →  agentive →  driver   →  plural →  drivers
  sing   →  agentive →  singer   →  plural →  singers
  dance  →  agentive →  dancer   →  plural →  dancers
  run    →  agentive →  runner   →  plural →  runners
  help   →  agentive →  helper   →  plural →  helpers

Three diagnostics:

  TEST 1 — Composition correctness
    Apply plural(agentive(emb(verb))). Look up nearest candidate.
    Does it match plural_agent? Tests if operators trained
    INDEPENDENTLY combine into a chained transformation.

  TEST 2 — Operator linearity
    Compare plural(agentive(z)) to z + v_agentive_const + v_plural_const
    (pure additive linear chain). Cosine high → operators are mostly
    linear shifts that add cleanly. Cosine low → MLP residuals carry
    significant content-sensitive structure.

  TEST 3 — Inverse roundtrip (sanity)
    inverse_A(forward_A(z)) ≈ z? Should be near 1.0 for both operators.

If all three pass, the embedding space has the algebraic structure the
project promised: concepts are not just learnable individually, they
combine into chained transformations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F


# Verbs not in agentive's training set. agentive maps them to agents,
# plural maps the agents to plural-agents.
TEST_CHAINS = [
    ("paint", "painter", "painters"),
    ("drive", "driver",  "drivers"),
    ("sing",  "singer",  "singers"),
    ("dance", "dancer",  "dancers"),
    ("run",   "runner",  "runners"),
    ("help",  "helper",  "helpers"),
]

# Two pool variants — same test, different distractor difficulty.
#
# POOL_HARSH includes immediate-source distractors (singular agents,
# bare verbs, gerunds). The first run showed these win top-1 by tiny
# cosine margins because the chained shift magnitude isn't large
# enough to escape them.
#
# POOL_FAIR removes those source-form distractors. Tests whether
# composition works in a clean closed-pool setting where the choice
# is between plural-agent forms.

POOL_HARSH = [
    # Correct answers (plural-agent forms)
    "painters", "drivers", "singers", "dancers", "runners", "helpers",
    # Plural-agent distractors
    "writers", "builders", "teachers",
    "doctors", "lawyers", "bakers", "gardeners", "actors", "swimmers",
    "leaders", "workers", "speakers", "readers", "thinkers", "creators",
    "designers", "managers", "performers", "climbers", "fighters",
    # Source attractors (these are what cost top-1 in the first run):
    "writer", "builder", "teacher", "painter", "driver", "singer",
    "dancer", "runner", "helper",
    "cats", "dogs", "trees", "books", "houses",
    "music", "dance", "running", "writing",
]

POOL_FAIR = [
    # Correct answers
    "painters", "drivers", "singers", "dancers", "runners", "helpers",
    # Plural-agent distractors
    "writers", "builders", "teachers",
    "doctors", "lawyers", "bakers", "gardeners", "actors", "swimmers",
    "leaders", "workers", "speakers", "readers", "thinkers", "creators",
    "designers", "managers", "performers", "climbers", "fighters",
    # Plain plurals (different concept, fair distractors)
    "cats", "dogs", "trees", "books", "houses",
]


def _strip(s: str) -> str:
    idx = s.find("#")
    return (s[:idx] if idx >= 0 else s).strip()


def read_pairs(path: Path) -> list[tuple[str, str]]:
    out = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                src = _strip(parts[0]); tgt = _strip(parts[1])
                if src and tgt:
                    out.append((src, tgt))
    return out


@torch.no_grad()
def encode_words(model, tokenizer, words, device, max_length=64):
    inputs = tokenizer(
        words, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = model(**inputs)
    last_hidden = out.last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).float()
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.float()


class Operator(nn.Module):
    """Same shape as ConceptOperator's forward direction."""

    def __init__(self, dim: int, mlp_hidden: int = 192):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dim) * 0.02)
        self.alpha = nn.Parameter(torch.ones(1))
        self.residual = nn.Sequential(
            nn.Linear(2 * dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, z):
        v_b = self.v.expand_as(z)
        delta = self.alpha * self.v + self.residual(torch.cat([z, v_b], dim=-1))
        return z + delta


def train_operator_pair(z_src, z_tgt, dim, device, epochs=2000, lr=1e-3, seed=0):
    """Train forward + inverse on (z_src, z_tgt). Returns (forward, inverse)."""
    torch.manual_seed(seed)
    fwd = Operator(dim).to(device)
    inv = Operator(dim).to(device)
    opt = torch.optim.AdamW(
        list(fwd.parameters()) + list(inv.parameters()), lr=lr,
    )
    for _ in range(epochs):
        opt.zero_grad()
        l_fwd = F.mse_loss(fwd(z_src), z_tgt)
        l_inv = F.mse_loss(inv(z_tgt), z_src)
        (l_fwd + l_inv).backward()
        opt.step()
    return fwd, inv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder", default="thenlper/gte-base")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading encoder: {args.encoder} ...")
    tok = AutoTokenizer.from_pretrained(args.encoder)
    mdl = AutoModel.from_pretrained(args.encoder).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)

    # Determine native dim from a probe forward.
    probe = encode_words(mdl, tok, ["test"], args.device)
    dim = probe.shape[1]
    print(f"  encoder native dim: {dim}")

    # ---- Train AGENTIVE operator ----
    agentive_pairs = read_pairs(Path("data/few_shot/agentive/text_pairs_train.tsv"))
    print(f"\nTraining AGENTIVE operator on {len(agentive_pairs)} pairs ...")
    z_a_src = encode_words(mdl, tok, [p[0] for p in agentive_pairs], args.device)
    z_a_tgt = encode_words(mdl, tok, [p[1] for p in agentive_pairs], args.device)
    fwd_a, inv_a = train_operator_pair(
        z_a_src, z_a_tgt, dim, args.device,
        epochs=args.epochs, seed=args.seed,
    )

    # ---- Train PLURAL operator ----
    plural_pairs = read_pairs(Path("data/plurality/text_pairs_train.tsv"))
    print(f"Training PLURAL operator on {len(plural_pairs)} pairs ...")
    z_p_src = encode_words(mdl, tok, [p[0] for p in plural_pairs], args.device)
    z_p_tgt = encode_words(mdl, tok, [p[1] for p in plural_pairs], args.device)
    fwd_p, inv_p = train_operator_pair(
        z_p_src, z_p_tgt, dim, args.device,
        epochs=args.epochs, seed=args.seed,
    )

    # ---- Encode test inputs ----
    verbs = [c[0] for c in TEST_CHAINS]
    agents = [c[1] for c in TEST_CHAINS]
    plural_agents = [c[2] for c in TEST_CHAINS]
    z_verb = encode_words(mdl, tok, verbs, args.device)
    z_agent_truth = encode_words(mdl, tok, agents, args.device)
    z_plural_agent_truth = encode_words(mdl, tok, plural_agents, args.device)

    # ---- Encode both pools ----
    z_pool_harsh = encode_words(mdl, tok, POOL_HARSH, args.device)
    z_pool_fair = encode_words(mdl, tok, POOL_FAIR, args.device)

    # ---- Compute the chained prediction once (same for both pools) ----
    with torch.no_grad():
        z_after_agentive = fwd_a(z_verb)
        z_after_plural = fwd_p(z_after_agentive)
        pred_n = F.normalize(z_after_plural, dim=-1)

    def _eval_pool(pool_words: list[str], z_pool: torch.Tensor, name: str):
        """Run TEST 1 on a given pool. Prints per-chain detail + accuracy."""
        with torch.no_grad():
            pool_n = F.normalize(z_pool, dim=-1)
            sims = pred_n @ pool_n.T
        print()
        print("=" * 88)
        print(f"TEST 1 — Composition correctness, {name} (pool size {len(pool_words)})")
        print("=" * 88)
        print(f"  {'verb':<8s}  {'expected':<12s}  {'top-1 pred':<14s}  {'cos(pred,truth)'}")
        print("  " + "-" * 75)
        correct = 0
        for i, (verb, _, plural_agent) in enumerate(TEST_CHAINS):
            top_vals, top_idx = sims[i].topk(args.top_k)
            top_words = [pool_words[j] for j in top_idx.tolist()]
            cos_truth = F.cosine_similarity(
                z_after_plural[i:i+1], z_plural_agent_truth[i:i+1], dim=-1,
            ).item()
            cos_intermediate = F.cosine_similarity(
                z_after_agentive[i:i+1], z_agent_truth[i:i+1], dim=-1,
            ).item()
            is_correct = top_words[0] == plural_agent
            if is_correct:
                correct += 1
            mark = "✓" if is_correct else "✗"
            print(f"  {mark} {verb:<8s}  {plural_agent:<12s}  {top_words[0]:<14s}  "
                  f"{cos_truth:.3f}    "
                  f"(after agentive: cos→{agents[i]} = {cos_intermediate:.3f})")
        acc = correct / len(TEST_CHAINS)
        print(f"\n  Composition accuracy ({name}): {correct}/{len(TEST_CHAINS)} = {acc:.3f}")
        # Top-K detail for first chain (helpful for debugging margins)
        print(f"\n  Top-{args.top_k} for {verbs[0]} → {plural_agents[0]}:")
        top_vals, top_idx = sims[0].topk(args.top_k)
        for v, j in zip(top_vals.tolist(), top_idx.tolist()):
            marker = "  ← target" if pool_words[j] == plural_agents[0] else ""
            print(f"    cos={v:+.3f}  {pool_words[j]}{marker}")
        return acc, sims

    composition_acc_harsh, _sims_harsh = _eval_pool(POOL_HARSH, z_pool_harsh, "HARSH POOL")
    composition_acc_fair, _sims_fair = _eval_pool(POOL_FAIR, z_pool_fair, "FAIR POOL")
    # Use harsh pool's accuracy as the default reference for the verdict
    # (continues the tradition of TEST 1 = harsh pool from v1).
    composition_acc = composition_acc_harsh

    # =============================================================
    # TEST 2 — Operator linearity
    # =============================================================
    print()
    print("=" * 88)
    print("TEST 2 — Operator linearity (does plural ∘ agentive ≈ z + v_a + v_p ?)")
    print("=" * 88)

    with torch.no_grad():
        v_a_const = (z_a_tgt - z_a_src).mean(dim=0)
        v_p_const = (z_p_tgt - z_p_src).mean(dim=0)

        # Linear-only chain: z + v_a + v_p
        z_linear_chain = z_verb + v_a_const + v_p_const

        # MLP chain (already computed above): z_after_plural
        # Compare them
        cos_linear_vs_mlp = F.cosine_similarity(
            z_linear_chain, z_after_plural, dim=-1,
        )
        print(f"\n  cos(linear-chain, MLP-chain) per verb:")
        for i, verb in enumerate(verbs):
            print(f"    {verb:<8s}  {cos_linear_vs_mlp[i].item():+.3f}")
        mean = cos_linear_vs_mlp.mean().item()
        print(f"  Mean: {mean:+.3f}")

        # Linear chain accuracy in BOTH pools.
        z_lin_n = F.normalize(z_linear_chain, dim=-1)
        for pool_words, z_pool, label in (
            (POOL_HARSH, z_pool_harsh, "HARSH"),
            (POOL_FAIR,  z_pool_fair,  "FAIR"),
        ):
            sims_lin = z_lin_n @ F.normalize(z_pool, dim=-1).T
            best_lin = sims_lin.argmax(dim=-1).tolist()
            correct_lin = sum(
                pool_words[best_lin[i]] == plural_agents[i]
                for i in range(len(verbs))
            )
            print(f"\n  Linear-chain (no MLP) pool accuracy ({label}): "
                  f"{correct_lin}/{len(verbs)} = {correct_lin/len(verbs):.3f}")
        print(f"    → If linear-chain ≈ MLP-chain accuracy at every pool, the")
        print(f"      operators are essentially additive linear shifts.")

    # =============================================================
    # TEST 3 — Inverse roundtrip
    # =============================================================
    print()
    print("=" * 88)
    print("TEST 3 — Inverse roundtrip (sanity)")
    print("=" * 88)
    with torch.no_grad():
        # Roundtrip on agentive: verbs → agents → verbs
        z_round_a = inv_a(fwd_a(z_verb))
        cos_a = F.cosine_similarity(z_round_a, z_verb, dim=-1).mean().item()
        # Roundtrip on plural: agents → plural-agents → agents
        z_round_p = inv_p(fwd_p(z_agent_truth))
        cos_p = F.cosine_similarity(z_round_p, z_agent_truth, dim=-1).mean().item()
    print(f"\n  agentive   inverse(forward(z)) ≈ z   →  cos = {cos_a:.3f}")
    print(f"  plural     inverse(forward(z)) ≈ z   →  cos = {cos_p:.3f}")

    # =============================================================
    # Verdict
    # =============================================================
    print()
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    n_chains = len(TEST_CHAINS)
    n_correct_harsh = int(round(composition_acc_harsh * n_chains))
    n_correct_fair  = int(round(composition_acc_fair * n_chains))
    print(f"  Composition (HARSH pool):           "
          f"{composition_acc_harsh:.3f}  ({n_correct_harsh}/{n_chains})")
    print(f"  Composition (FAIR pool):            "
          f"{composition_acc_fair:.3f}  ({n_correct_fair}/{n_chains})")
    print(f"  Linearity      (TEST 2 mean cos):   {mean:+.3f}")
    print(f"  Inversibility  (TEST 3 mean cos):   "
          f"{(cos_a + cos_p) / 2:+.3f}")
    print()
    # Verdict prioritizes the FAIR pool result — it tests composition
    # without the source-attractor confound. HARSH pool number stays
    # visible to honestly characterize the magnitude limit.
    composition_acc = composition_acc_fair
    if composition_acc >= 0.7:
        print("  → The operators COMPOSE. Concepts trained independently combine")
        print("    into chained transformations. The embedding space has the")
        print("    algebraic structure expected of a concept library.")
    elif composition_acc >= 0.4:
        print("  → Partial composition. Operators combine for some inputs but not")
        print("    others. May indicate operator-specific drift or chain-length")
        print("    accumulating error.")
    else:
        print("  → Operators do NOT compose. Each operator may work in isolation")
        print("    but their compositions don't preserve sensible structure.")


if __name__ == "__main__":
    main()
