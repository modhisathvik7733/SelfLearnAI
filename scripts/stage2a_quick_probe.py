"""Phase 2a quick probe — does the encoder admit continuous-vector inputs?

Empirical resolution of Approach A vs Approach B (plan §19.12).
~1–3 min of GPU time on a single A100.

Question: can we optimize a free continuous matrix X ∈ R^(T×D) so that
encode(X) ≈ ψ_target, snap X to vocabulary tokens, and produce text
that BOTH (a) re-encodes back to ψ_target with cos≥0.85 AND
(b) is grammatically valid?

The probe runs THREE snap modes (in order, on the same X):

  RAW SNAP — independent per-position argmax over the full vocabulary.
    Original behavior. Often picks subword BPE fragments.

  REFINED SNAP (cos-only) — start from complete-word argmax, then
    coordinate-ascent on cos: cycle through positions, try the
    top-K alternatives at each, accept any swap that improves
    cos(encode(text), ψ_target). λ_grammar = 0.

  REFINED SNAP (cos + grammar) — same coordinate-ascent but optimizes
    score = cos + λ_grammar · grammar_proxy(text). The grammar proxy
    uses `wordfreq.zipf_frequency` (microseconds per call) as a fast
    proxy for "is this real English". The final grade is then run
    through `language_tool_python` (rule-based, no LLM) for the
    ground-truth grammaticality measurement (≥95% sentences with 0
    grammar errors is the plan §9.4 gate).

If the cos+grammar mode produces text that's BOTH high-cos AND
LanguageTool-clean, Approach A's ceiling includes grammatical text
and the architecture is viable. If grammar stays broken regardless
of λ_grammar, Approach A is dead and we commit to B.

Run:
  python scripts/stage2a_quick_probe.py
  python scripts/stage2a_quick_probe.py --lambda-grammar 0.3
  python scripts/stage2a_quick_probe.py --no-refine    # legacy raw-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


TARGET_SENTENCES = [
    "the plural of cat is cats",
    "the past tense of run is ran",
    "the comparative of big is bigger",
    "a person who paints is a painter",
    "the opposite of hot is cold",
]

# Verdict thresholds — same shape as the full plan's 2a.0 gate.
LOSS_OK = 1e-3
COS_RECOVERED_OK = 0.85


# ---------------------------------------------------------------------------
# Grammar scoring (fast proxy + slow gold standard)
# ---------------------------------------------------------------------------

# Soft import — grammar tooling is optional. The probe still produces
# meaningful output without these (just no grammar info).
try:
    from wordfreq import zipf_frequency
    _HAS_WORDFREQ = True
except ImportError:
    _HAS_WORDFREQ = False
    def zipf_frequency(word: str, lang: str) -> float:        # type: ignore
        del word, lang  # stub fallback — real fn is wordfreq.zipf_frequency
        return 0.0

try:
    import language_tool_python      # type: ignore
    _HAS_LANGUAGE_TOOL = True
except ImportError:
    _HAS_LANGUAGE_TOOL = False
    language_tool_python = None      # type: ignore


_LT_INSTANCE = None


def get_language_tool():
    """Lazy-init LanguageTool. Returns None if not installed."""
    global _LT_INSTANCE
    if not _HAS_LANGUAGE_TOOL:
        return None
    if _LT_INSTANCE is None:
        # 'en-US' is rule-based + statistical; no LLM.
        _LT_INSTANCE = language_tool_python.LanguageTool("en-US")
    return _LT_INSTANCE


def grammar_proxy(text: str) -> float:
    """Fast proxy for grammaticality. Returns a score where higher = more
    grammatical-looking. Used in the REFINEMENT INNER LOOP (must be
    fast — called thousands of times per probe).

    Components:
      - Mean Zipf frequency of the words (common words → high)
      - Penalty for adjacent identical tokens (gibberish hallmark)
      - Penalty for ALL-very-low-freq sequences

    Range is roughly [0, 1] but not strictly bounded. The optimizer
    just needs a relative gradient; absolute scale is tuned by
    λ_grammar in the caller.
    """
    if not _HAS_WORDFREQ:
        return 0.0
    words = text.lower().split()
    if not words:
        return 0.0
    # Mean Zipf frequency (1 = very rare, 7 = "the"). Normalize to ~[0,1].
    zipfs = [zipf_frequency(w, "en") for w in words]
    mean_zipf = sum(zipfs) / len(zipfs)
    common_score = mean_zipf / 6.0       # ~"the" caps near 1.0
    # Repetition penalty (count adjacent duplicates).
    adjacent_dups = sum(
        1 for i in range(1, len(words)) if words[i] == words[i - 1]
    )
    rep_penalty = adjacent_dups / max(1, len(words) - 1)
    # All-rare penalty.
    n_rare = sum(1 for z in zipfs if z < 2.0)
    rare_penalty = n_rare / len(words)
    return float(max(0.0, common_score - 0.3 * rep_penalty - 0.2 * rare_penalty))


def grammar_grade(text: str) -> tuple[int, bool]:
    """Ground-truth grammar grade via LanguageTool (rule-based, no LLM).

    Returns (n_errors, passes_gate) where passes_gate = (n_errors == 0).
    Plan §9.4 gate: ≥95% sentences pass (0 errors).

    Falls back to a strict version of the proxy if LanguageTool isn't
    installed: passes iff proxy ≥ 0.55 (heuristic threshold).
    """
    if not _HAS_LANGUAGE_TOOL:
        proxy = grammar_proxy(text)
        return (0 if proxy >= 0.55 else 1, proxy >= 0.55)
    lt = get_language_tool()
    matches = lt.check(text)
    n = len(matches)
    return (n, n == 0)


def build_complete_word_mask(tok) -> torch.Tensor:
    """Return a [vocab_size] bool tensor: True for tokens that look like
    complete words. Filters out ## BPE continuations + special tokens.

    Heuristic: keep tokens whose decoded form starts with a non-#, has
    length ≥ 2 (drops single-letter junk), and is alphanumeric or
    contains common punctuation. Dropping these is what fixes the
    'water catss covers thames pluraled' word-salad problem in the raw
    snap.
    """
    vocab_size = len(tok)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    special_ids = set(tok.all_special_ids)
    for i in range(vocab_size):
        if i in special_ids:
            continue
        s = tok.convert_ids_to_tokens(i)
        if not s or s.startswith("##"):
            continue
        if len(s) < 2:
            continue
        # Allow alphanumeric + apostrophe + dash. Excludes weird
        # symbols that snuck into the snap last run (⁺, etc.)
        if not all(c.isalnum() or c in "'-" for c in s):
            continue
        mask[i] = True
    return mask


def refine_tokens(
    initial_tokens: list[int],
    X_pos: torch.Tensor,            # [T, D] — the optimized per-position vectors
    word_emb: torch.Tensor,         # [V, D]
    *,
    encode_fn,                      # callable: list[str] -> tensor [B, D]
    psi_target: torch.Tensor,       # [D]
    tokenizer,
    vocab_mask: torch.Tensor,       # [V] bool — restrict candidates
    top_k: int = 20,
    max_iters: int = 5,
    lambda_grammar: float = 0.0,
) -> tuple[list[int], dict, int]:
    """Coordinate-ascent refinement over discrete tokens.

    Objective: score = cos(encode(text), psi_target) + λ_grammar · grammar_proxy(text).

    Cycle through positions. At each, get the top-K candidate tokens
    (by per-position cosine, restricted by vocab_mask). Try each
    candidate; keep the swap iff it improves the full score.

    With lambda_grammar=0, this is pure-cos (legacy).
    With lambda_grammar>0, the optimizer trades some cos for
    grammaticality. Set λ low (~0.1–0.3) to keep cos primary.

    Returns (refined_tokens, score_breakdown, iterations_used).
    score_breakdown is {"cos": float, "grammar_proxy": float, "score": float}.
    """
    tokens = list(initial_tokens)

    def score_tokens(toks: list[int]) -> tuple[float, float, float]:
        text = tokenizer.decode(toks, skip_special_tokens=True)
        if not text.strip():
            return -1.0, 0.0, -1.0
        psi = encode_fn([text])[0]
        cos = float(F.cosine_similarity(psi, psi_target, dim=0).item())
        gram = grammar_proxy(text) if lambda_grammar > 0.0 else 0.0
        sc = cos + lambda_grammar * gram
        return cos, gram, sc

    # Restrict per-position similarity scores to the masked vocab.
    we_n = F.normalize(word_emb, dim=-1)                # [V, D]
    X_n = F.normalize(X_pos, dim=-1)                    # [T, D]
    sims = X_n @ we_n.T                                  # [T, V]
    masked_sims = sims.clone()
    masked_sims[:, ~vocab_mask] = -1e9
    top_k_per_pos = masked_sims.topk(top_k, dim=-1).indices.tolist()  # [T][K]

    current_cos, current_gram, current_score = score_tokens(tokens)
    iters_used = 0
    for it in range(max_iters):
        any_improvement = False
        for pos in range(len(tokens)):
            best_token = tokens[pos]
            best_cos, best_gram, best_score = current_cos, current_gram, current_score
            for cand in top_k_per_pos[pos]:
                if cand == best_token:
                    continue
                tokens[pos] = cand
                new_cos, new_gram, new_score = score_tokens(tokens)
                if new_score > best_score + 1e-6:
                    best_cos, best_gram, best_score = new_cos, new_gram, new_score
                    best_token = cand
                    any_improvement = True
            tokens[pos] = best_token
            current_cos, current_gram, current_score = best_cos, best_gram, best_score
        iters_used = it + 1
        if not any_improvement:
            break
    return (
        tokens,
        {"cos": current_cos, "grammar_proxy": current_gram, "score": current_score},
        iters_used,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Number of continuous positions to optimize.")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Adam steps per target sentence.")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--refine", dest="refine", action="store_true", default=True,
        help="Run coordinate-ascent refinement after raw snap (default).",
    )
    parser.add_argument(
        "--no-refine", dest="refine", action="store_false",
        help="Skip refinement (legacy raw-snap-only mode).",
    )
    parser.add_argument(
        "--refine-top-k", type=int, default=20,
        help="Per-position candidate count during refinement.",
    )
    parser.add_argument(
        "--refine-max-iters", type=int, default=5,
        help="Max coordinate-ascent passes (early-stops on convergence).",
    )
    parser.add_argument(
        "--lambda-grammar", type=float, nargs="+", default=[0.0, 0.1, 0.3],
        help=(
            "List of grammar-penalty coefficients to try in refinement. "
            "λ=0 → pure cos (legacy). λ>0 → score = cos + λ·grammar_proxy. "
            "Each sentence is refined under each λ; LanguageTool grades each."
        ),
    )
    parser.add_argument("--out", default="results/stage2a/quick_probe.json")
    args = parser.parse_args()

    print("Phase 2a quick probe — encoder-input sanity")
    print("=" * 70)
    print(f"Encoder: {args.encoder}")
    print(f"Device:  {args.device}")
    print(f"T (positions): {args.seq_len}   steps/target: {args.steps}   lr: {args.lr}")

    print(f"\nLoading {args.encoder} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.encoder)
    mdl = AutoModel.from_pretrained(args.encoder).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    DIM = mdl.config.hidden_size
    print(f"  hidden_size: {DIM}")

    word_emb = mdl.get_input_embeddings().weight        # [V, D]
    we_std = float(word_emb.std().item())
    print(f"  vocab size: {word_emb.shape[0]}, word_emb std: {we_std:.4f}")

    # ---- Encoders -----------------------------------------------------
    def encode_text(texts: list[str]) -> torch.Tensor:
        """Standard mean-pooled encode (mirrors stage1's make_encode_fn).

        Wrapped in no_grad — we only need its OUTPUT as a target (no
        gradients flow back to the model from this path)."""
        with torch.no_grad():
            batch = tok(
                texts, padding=True, truncation=True, max_length=64,
                return_tensors="pt",
            ).to(args.device)
            out = mdl(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def encode_continuous(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode pre-computed embeddings. Bypasses tokenization but
        keeps the rest of the model's forward path identical
        (position embeddings, layernorm, attention, mean-pool).

        NOT wrapped in no_grad — gradients DO flow back to
        `inputs_embeds` so we can optimize it. Model parameters
        themselves remain frozen (set above)."""
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long, device=inputs_embeds.device,
            )
        out = mdl(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    # ---- [0] Consistency check: encode_text vs encode_continuous on real tokens
    print("\n[0] Consistency check (paths agree on real tokens?)")
    print("-" * 70)
    sentence = TARGET_SENTENCES[0]
    psi_text = encode_text([sentence])[0]
    with torch.no_grad():
        batch = tok(
            [sentence], padding=True, truncation=True, max_length=64,
            return_tensors="pt",
        ).to(args.device)
        real_embeds = mdl.get_input_embeddings()(batch["input_ids"])
        psi_cont = encode_continuous(real_embeds, batch["attention_mask"])[0]
    consistency_cos = float(F.cosine_similarity(psi_text, psi_cont, dim=0).item())
    print(f"  cos(encode_text, encode_continuous on real tokens) = {consistency_cos:.6f}")
    if consistency_cos < 0.99:
        print("  WARNING: paths diverge — investigate before trusting probe results")
    else:
        print("  OK: paths produce ~identical ψ for real tokens")

    # ---- [1] Per-sentence probe ---------------------------------------
    torch.manual_seed(args.seed)

    # Build the complete-word vocabulary mask once. Used during
    # refinement to filter out ## BPE pieces and weird symbols.
    print("\nBuilding complete-word vocabulary mask ...")
    complete_mask = build_complete_word_mask(tok).to(args.device)
    n_kept = int(complete_mask.sum().item())
    print(f"  vocab kept under complete-word filter: "
          f"{n_kept}/{complete_mask.numel()}")

    results: list[dict] = []
    print("\n[1] Per-sentence probe")
    print("-" * 70)
    for i, sent in enumerate(TARGET_SENTENCES):
        psi_target = encode_text([sent])[0]                    # [D]

        # Initialize X randomly, scaled to word-embedding distribution.
        X = torch.randn(1, args.seq_len, DIM, device=args.device) * we_std
        X.requires_grad_(True)
        opt = torch.optim.Adam([X], lr=args.lr)

        for step in range(args.steps):
            opt.zero_grad()
            psi_pred = encode_continuous(X)[0]
            loss = F.mse_loss(psi_pred, psi_target)
            loss.backward()
            opt.step()
        final_loss = float(loss.item())

        # ---- RAW SNAP: independent argmax over full vocabulary -------
        with torch.no_grad():
            X_n = F.normalize(X[0], dim=-1)                    # [T, D]
            we_n = F.normalize(word_emb, dim=-1)               # [V, D]
            sims = X_n @ we_n.T                                # [T, V]
            raw_tokens = sims.argmax(dim=-1).tolist()
        raw_text = tok.decode(raw_tokens, skip_special_tokens=True)
        psi_recovered_raw = encode_text([raw_text])[0]
        cos_raw = float(F.cosine_similarity(
            psi_recovered_raw, psi_target, dim=0,
        ).item())

        # Sanity: cos at the optimization endpoint (BEFORE snap).
        with torch.no_grad():
            psi_pred_final = encode_continuous(X)[0]
            cos_pre_snap = float(F.cosine_similarity(
                psi_pred_final, psi_target, dim=0,
            ).item())

        # ---- REFINED SNAP under each λ_grammar: coordinate-ascent on
        # score = cos + λ · grammar_proxy. λ=0 reproduces legacy cos-only.
        with torch.no_grad():
            masked_sims = sims.clone()
            masked_sims[:, ~complete_mask] = -1e9
            init_tokens_refined = masked_sims.argmax(dim=-1).tolist()

        # Grade RAW snap text with LanguageTool too (baseline).
        raw_n_errors, raw_grammar_pass = grammar_grade(raw_text)

        refined_runs: list[dict] = []
        if args.refine:
            for lam in args.lambda_grammar:
                ref_tokens, scores, ref_iters = refine_tokens(
                    list(init_tokens_refined),
                    X[0].detach(),
                    word_emb,
                    encode_fn=encode_text,
                    psi_target=psi_target,
                    tokenizer=tok,
                    vocab_mask=complete_mask,
                    top_k=args.refine_top_k,
                    max_iters=args.refine_max_iters,
                    lambda_grammar=lam,
                )
                ref_text = tok.decode(ref_tokens, skip_special_tokens=True)
                # Re-encode for the cos number and language-tool grade.
                psi_ref = encode_text([ref_text])[0]
                cos_ref = float(F.cosine_similarity(
                    psi_ref, psi_target, dim=0,
                ).item())
                n_errors, grammar_pass = grammar_grade(ref_text)
                refined_runs.append({
                    "lambda_grammar": lam,
                    "tokens": ref_tokens,
                    "text": ref_text,
                    "cos_recovered": cos_ref,
                    "grammar_proxy": scores["grammar_proxy"],
                    "lt_n_errors": n_errors,
                    "lt_passes": grammar_pass,
                    "iterations": ref_iters,
                })

        # Headline = best λ run (highest cos with grammar passing, else
        # highest cos overall).
        if refined_runs:
            grammar_passers = [r for r in refined_runs if r["lt_passes"]]
            best_run = max(
                grammar_passers if grammar_passers else refined_runs,
                key=lambda r: r["cos_recovered"],
            )
        else:
            best_run = {
                "lambda_grammar": None, "text": raw_text,
                "cos_recovered": cos_raw, "lt_n_errors": raw_n_errors,
                "lt_passes": raw_grammar_pass, "iterations": 0,
            }
        cos_recovered = best_run["cos_recovered"]

        record = {
            "target": sent,
            "final_loss": final_loss,
            "cos_pre_snap": cos_pre_snap,
            "raw_snap": {
                "text": raw_text,
                "token_ids": raw_tokens,
                "cos_recovered": cos_raw,
                "lt_n_errors": raw_n_errors,
                "lt_passes": raw_grammar_pass,
            },
            "refined_runs": refined_runs,
            "best_run": best_run,
            "cos_recovered_after_snap": cos_recovered,
        }
        results.append(record)

        loss_mark = "✓" if final_loss < LOSS_OK else "✗"
        cos_mark_raw = "✓" if cos_raw >= COS_RECOVERED_OK else "✗"
        gr_mark_raw = "✓" if raw_grammar_pass else "✗"
        print(f"\n  [{i+1}/{len(TARGET_SENTENCES)}] target: {sent!r}")
        print(f"    {loss_mark} final loss:        {final_loss:.6f} "
              f"(threshold < {LOSS_OK})")
        print(f"      cos(pre-snap, target):  {cos_pre_snap:.4f}")
        print(f"      RAW snap:               {raw_text!r}")
        print(f"    {cos_mark_raw} cos={cos_raw:.4f}  "
              f"{gr_mark_raw} LT errors={raw_n_errors}  "
              f"(grammar pass={raw_grammar_pass})")
        for r in refined_runs:
            cos_mark = "✓" if r["cos_recovered"] >= COS_RECOVERED_OK else "✗"
            gr_mark = "✓" if r["lt_passes"] else "✗"
            print(f"      λ={r['lambda_grammar']:.1f} ({r['iterations']} iters): "
                  f"{r['text']!r}")
            print(f"      {cos_mark} cos={r['cos_recovered']:.4f}  "
                  f"proxy={r['grammar_proxy']:.3f}  "
                  f"{gr_mark} LT errors={r['lt_n_errors']}")
        print(f"      → best: λ={best_run.get('lambda_grammar')}, "
              f"cos={best_run['cos_recovered']:.4f}, "
              f"LT errors={best_run['lt_n_errors']}, "
              f"text={best_run['text']!r}")

    # ---- Verdict -----------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    losses = sorted(r["final_loss"] for r in results)
    coses = sorted(r["cos_recovered_after_snap"] for r in results)
    median_loss = losses[len(losses) // 2]
    median_cos = coses[len(coses) // 2]
    n_loss_ok = sum(1 for r in results if r["final_loss"] < LOSS_OK)
    n_cos_ok = sum(1 for r in results if r["cos_recovered_after_snap"] >= COS_RECOVERED_OK)
    n_grammar_ok = sum(1 for r in results if r["best_run"]["lt_passes"])
    n_both = sum(
        1 for r in results
        if r["cos_recovered_after_snap"] >= COS_RECOVERED_OK
        and r["best_run"]["lt_passes"]
    )

    grammar_label = (
        "LanguageTool errors=0"
        if _HAS_LANGUAGE_TOOL
        else "wordfreq proxy ≥ 0.55 (LanguageTool not installed)"
    )

    print(f"Median final loss:               {median_loss:.6f}  "
          f"(target < {LOSS_OK})")
    print(f"Median cos(re-encoded, target):  {median_cos:.4f}  "
          f"(target ≥ {COS_RECOVERED_OK})")
    print(f"Sentences passing loss gate:     {n_loss_ok}/{len(results)}")
    print(f"Sentences passing cos gate:      {n_cos_ok}/{len(results)}")
    print(f"Sentences passing grammar gate:  {n_grammar_ok}/{len(results)}  "
          f"({grammar_label})")
    print(f"Sentences passing BOTH:          {n_both}/{len(results)}")

    if not _HAS_WORDFREQ:
        print("\n  NOTE: `wordfreq` not installed — grammar proxy returned 0.")
        print("  Install: pip install wordfreq")
    if not _HAS_LANGUAGE_TOOL:
        print("\n  NOTE: `language_tool_python` not installed — final grammar")
        print("  grade fell back to the proxy threshold. The proxy is")
        print("  approximate; LanguageTool is the gold standard.")
        print("  Install: pip install language-tool-python")

    loss_gate = median_loss < LOSS_OK
    cos_gate = median_cos >= COS_RECOVERED_OK
    grammar_gate = n_grammar_ok >= int(0.95 * len(results))   # plan §9.4

    if loss_gate and cos_gate and grammar_gate:
        verdict = "APPROACH_A_VIABLE"
        message = (
            "All gates passed. The encoder accepts continuous inputs, the "
            "token-snap step recovers ψ, AND the snapped text is "
            "grammatically valid. Approach A is on the table for the full "
            "Phase 2a build (continuous-output decoder + token snap)."
        )
    elif loss_gate and cos_gate and not grammar_gate:
        verdict = "COS_OK_BUT_GRAMMAR_FAILS"
        message = (
            "Cos gate passes but grammaticality gate fails: the snapped "
            "text contains the right semantic content but is not "
            "grammatically valid English. Plan §9.4 requires ≥95% "
            "LanguageTool-clean — currently far below. Increasing λ_grammar "
            "didn't recover grammar (or it did, at the cost of cos). "
            "Approach A's ceiling appears to NOT include grammatical text "
            "from per-position discrete snap. Commit to Approach B "
            "(Gumbel-STE discrete diffusion) — its joint token distribution "
            "is more likely to produce coherent sequences."
        )
    elif loss_gate and not cos_gate:
        verdict = "OFF_MANIFOLD_CONFIRMED"
        message = (
            "Optimization converges but snapped tokens don't recover ψ. "
            "E5 satisfies the loss with adversarial continuous solutions "
            "that don't correspond to coherent token sequences. Approach A "
            "is structurally dead. Commit to Approach B."
        )
    elif not loss_gate and cos_gate:
        verdict = "ANOMALY_LOSS_HIGH_BUT_RECOVERY_OK"
        message = (
            "Unexpected: loss didn't fully converge but the snap still "
            "recovered ψ. Investigate optimization (more steps? higher "
            "lr?) before committing to either approach."
        )
    else:
        verdict = "ENCODER_GRADIENTS_UNWORKABLE"
        message = (
            "Optimization didn't converge. Investigate before either approach."
        )

    print(f"\n→ {verdict}")
    print(f"\n{message}")

    # ---- Save JSON ----------------------------------------------------
    payload = {
        "task": "2a.quick_probe",
        "encoder": args.encoder,
        "encoder_dim": DIM,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "lambda_grammar_values": args.lambda_grammar,
        "wordfreq_installed": _HAS_WORDFREQ,
        "language_tool_installed": _HAS_LANGUAGE_TOOL,
        "thresholds": {
            "loss_ok": LOSS_OK,
            "cos_recovered_ok": COS_RECOVERED_OK,
            "grammar_pass_rate_ok": 0.95,
        },
        "consistency_cos_real_tokens": consistency_cos,
        "per_sentence": results,
        "summary": {
            "median_loss": median_loss,
            "median_cos_recovered": median_cos,
            "n_loss_ok": n_loss_ok,
            "n_cos_ok": n_cos_ok,
            "n_grammar_ok": n_grammar_ok,
            "n_both": n_both,
            "n_total": len(results),
        },
        "verdict": verdict,
        "message": message,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n→ saved JSON to {out_path}")


if __name__ == "__main__":
    main()
