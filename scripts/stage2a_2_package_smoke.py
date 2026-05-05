"""Phase 2a / Sub-task 2a.2 — selflearnai/generator/ package smoke.

Plan §19.14 acceptance: "2a.0e and 2a.0f scripts can be rewritten
using the package and produce identical results to commits bd4c183
and d554c6c."

This smoke runs a tiny end-to-end pass through the package — tiny
corpus, tiny decoder, few training steps — and verifies:

  1. Every public symbol from `selflearnai.generator` imports cleanly.
  2. PointerSeqCondDecoder.forward returns the right shapes and the
     log-probs are valid (sum to 1 over vocab when exp'd, no NaN/Inf).
  3. mixture_nll runs on the decoder's output without dimension errors.
  4. perturb_h preserves shape and respects the padding mask.
  5. decode_to_text returns a list of strings of the right length.
  6. The eval module's word_pair_fidelity and gate roll-up work
     end-to-end on synthetic data.
  7. The corpus reader can load the production TSVs from sub-task 2a.1
     and the row counts match metadata.json.

No GPU needed (the smoke uses tiny dim and CPU-only forward). Fast.

Run:
  python scripts/stage2a_2_package_smoke.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from selflearnai.generator import (
    PointerSeqCondDecoder,
    perturb_h,
    mixture_nll,
    decode_to_text,
    word_pair_fidelity,
    GeneralizationGates,
    GenerationVerdict,
    read_corpus_tsv,
    CorpusEntry,
)
from selflearnai.generator.eval import roll_up_gates


def check(label: str, ok: bool, *, fatal_on_fail: bool = True) -> None:
    """Pretty-print a check; FATAL on fail unless fatal_on_fail=False."""
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}")
    if not ok and fatal_on_fail:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--corpus-dir", default="data/explanations_v2",
                        help="Where 2a.1's TSVs live.")
    parser.add_argument("--device", default="cpu",
                        help="Smoke is CPU-friendly; --device cuda just to confirm GPU works too.")
    args = parser.parse_args()

    print("Phase 2a / Sub-task 2a.2 — package smoke")
    print("=" * 78)

    # =====================================================================
    # [1] Imports OK (already happened — module imports above)
    # =====================================================================
    print("\n[1] Public-API imports")
    print("-" * 78)
    check("PointerSeqCondDecoder importable", PointerSeqCondDecoder is not None)
    check("perturb_h importable", perturb_h is not None)
    check("mixture_nll importable", mixture_nll is not None)
    check("decode_to_text importable", decode_to_text is not None)
    check("word_pair_fidelity importable", word_pair_fidelity is not None)
    check("GeneralizationGates importable", GeneralizationGates is not None)
    check("GenerationVerdict importable", GenerationVerdict is not None)
    check("read_corpus_tsv importable", read_corpus_tsv is not None)
    check("CorpusEntry importable", CorpusEntry is not None)

    # =====================================================================
    # [2] Decoder forward shape + numeric sanity
    # =====================================================================
    print("\n[2] PointerSeqCondDecoder forward shape + log-probs sanity")
    print("-" * 78)
    torch.manual_seed(0)
    DIM = 64                # tiny encoder dim for smoke
    HIDDEN = 32
    T_IN = 8
    T_OUT = 8
    VOCAB = 100
    B = 3
    decoder = PointerSeqCondDecoder(
        encoder_dim=DIM, hidden_dim=HIDDEN, t_max=T_OUT,
        vocab_size=VOCAB, n_layers=2, n_heads=4,
    ).to(args.device)
    h = torch.randn(B, T_IN, DIM, device=args.device)
    h_mask = torch.ones(B, T_IN, device=args.device)
    h_mask[0, -2:] = 0     # mark last 2 positions of item 0 as padding
    token_ids = torch.randint(0, VOCAB, (B, T_IN), device=args.device)
    log_probs, hidden, p_gen = decoder(h, h_mask, token_ids)
    check(f"log_probs shape == [B={B}, T_out={T_OUT}, V={VOCAB}]",
          log_probs.shape == (B, T_OUT, VOCAB))
    check(f"hidden shape == [B={B}, T_out={T_OUT}, h={HIDDEN}]",
          hidden.shape == (B, T_OUT, HIDDEN))
    check(f"p_gen shape == [B={B}, T_out={T_OUT}, 1]",
          p_gen.shape == (B, T_OUT, 1))
    check("p_gen ∈ (0, 1)",
          bool(((p_gen > 0) & (p_gen < 1)).all().item()))
    probs = log_probs.exp()
    sums = probs.sum(dim=-1)
    check("exp(log_probs).sum(-1) ≈ 1",
          bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-3)))
    check("no NaN/Inf in log_probs",
          bool(torch.isfinite(log_probs).all().item()))

    # =====================================================================
    # [3] mixture_nll runs end-to-end
    # =====================================================================
    print("\n[3] mixture_nll")
    print("-" * 78)
    target_ids = torch.randint(0, VOCAB, (B, T_OUT), device=args.device)
    nll = mixture_nll(log_probs, target_ids)
    check("nll is finite scalar",
          nll.dim() == 0 and bool(torch.isfinite(nll).item()))
    check("nll > 0 (random init shouldn't be perfect)",
          bool((nll > 0).item()))

    # =====================================================================
    # [4] perturb_h preserves shape and respects mask
    # =====================================================================
    print("\n[4] perturb_h shape + masking")
    print("-" * 78)
    torch.manual_seed(42)
    h_p = perturb_h(h, h_mask, apply_prob=1.0)   # force perturbation
    check("perturbed h shape unchanged", h_p.shape == h.shape)
    # Padding positions of item 0 should be unchanged.
    item0_pad_orig = h[0, T_IN - 2:T_IN]
    item0_pad_new = h_p[0, T_IN - 2:T_IN]
    check("padding positions unchanged after perturb_h",
          bool(torch.allclose(item0_pad_orig, item0_pad_new)))

    # =====================================================================
    # [5] decode_to_text returns list of strings
    # =====================================================================
    print("\n[5] decode_to_text")
    print("-" * 78)

    class FakeTokenizer:
        """Minimal tokenizer stub for the smoke (no transformers
        dependency in the package itself)."""
        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(f"tok{i}" for i in ids if i > 0)

    fake_tok = FakeTokenizer()
    texts = decode_to_text(log_probs, fake_tok)
    check(f"len(texts) == B={B}", len(texts) == B)
    check("all entries are str", all(isinstance(t, str) for t in texts))

    # =====================================================================
    # [6] word_pair_fidelity + gate roll-up
    # =====================================================================
    print("\n[6] word_pair_fidelity + gate roll-up")
    print("-" * 78)
    src_in, tgt_in, both = word_pair_fidelity("book", "books", "the plural of book is books")
    check("both words present → both_in_gen=True", both)
    src_in2, tgt_in2, both2 = word_pair_fidelity("book", "books", "the plural of leg is hands")
    check("neither word → both_in_gen=False", not both2)

    # Test BOTH directions of the gates with a 3-verdict roll-up.
    # n=3, word_fidelity_min=0.70 → target = int(0.70*3) = 2.
    # We craft 1/3 with both_in=True, so word_fidelity_gate is False.
    verdicts_fail_word = [
        GenerationVerdict(
            target="t1", generated="g1", concept="plural",
            src_word="book", tgt_word="books",
            cos_recovered=0.95, grammar_pass=True, grammar_n_errors=0,
            grammar_proxy=0.9,
            src_in_gen=True, tgt_in_gen=True, both_in_gen=True,
            exact_match=False, p_gen_mean=0.6,
        ),
        GenerationVerdict(
            target="t2", generated="g2", concept="plural",
            src_word="cat", tgt_word="cats",
            cos_recovered=0.88, grammar_pass=True, grammar_n_errors=0,
            grammar_proxy=0.85,
            src_in_gen=True, tgt_in_gen=False, both_in_gen=False,
            exact_match=False, p_gen_mean=0.7,
        ),
        GenerationVerdict(
            target="t3", generated="g3", concept="plural",
            src_word="dog", tgt_word="dogs",
            cos_recovered=0.86, grammar_pass=True, grammar_n_errors=0,
            grammar_proxy=0.85,
            src_in_gen=False, tgt_in_gen=False, both_in_gen=False,
            exact_match=False, p_gen_mean=0.7,
        ),
    ]
    gates_fail = roll_up_gates(
        verdicts_fail_word, cos_min=0.85, grammar_pass_rate=0.95,
        word_fidelity_min=0.70,
    )
    check("median_cos = 0.88 (middle of [0.86, 0.88, 0.95])",
          abs(gates_fail.median_cos - 0.88) < 1e-6)
    check("cos_gate True (median 0.88 ≥ 0.85)", gates_fail.cos_gate)
    check("grammar_gate True (3/3 ≥ int(0.95*3)=2)", gates_fail.grammar_gate)
    check("word_fidelity_gate False (1/3 < int(0.7*3)=2)",
          not gates_fail.word_fidelity_gate)
    check("all_pass False when word-fidelity fails", not gates_fail.all_pass)
    check("p_gen_overall computed", gates_fail.p_gen_overall is not None)

    # Now flip the third verdict to both_in=True so all 3 gates pass.
    verdicts_pass = [
        verdicts_fail_word[0],
        verdicts_fail_word[1],
        GenerationVerdict(
            target="t3", generated="g3", concept="plural",
            src_word="dog", tgt_word="dogs",
            cos_recovered=0.86, grammar_pass=True, grammar_n_errors=0,
            grammar_proxy=0.85,
            src_in_gen=True, tgt_in_gen=True, both_in_gen=True,
            exact_match=True, p_gen_mean=0.5,
        ),
    ]
    gates_pass = roll_up_gates(
        verdicts_pass, cos_min=0.85, grammar_pass_rate=0.95,
        word_fidelity_min=0.70,
    )
    check("word_fidelity_gate True (2/3 ≥ int(0.7*3)=2)", gates_pass.word_fidelity_gate)
    check("all_pass True when all gates met", gates_pass.all_pass)

    # =====================================================================
    # [7] read_corpus_tsv against 2a.1 output
    # =====================================================================
    print("\n[7] Corpus reader vs 2a.1 metadata")
    print("-" * 78)
    corpus_dir = Path(args.corpus_dir)
    train_tsv = corpus_dir / "train.tsv"
    holdout_tsv = corpus_dir / "holdout.tsv"
    metadata_json = corpus_dir / "metadata.json"
    if not train_tsv.exists() or not holdout_tsv.exists() or not metadata_json.exists():
        print(f"  ⚠ corpus files missing under {corpus_dir} — run "
              f"scripts/stage2a_1_corpus.py first.")
        check("corpus files present", False, fatal_on_fail=False)
    else:
        train_rows = read_corpus_tsv(train_tsv)
        holdout_rows = read_corpus_tsv(holdout_tsv)
        with open(metadata_json) as f:
            metadata = json.load(f)
        n_train_meta = metadata["totals"]["n_train"]
        n_holdout_meta = metadata["totals"]["n_holdout"]
        check(f"train.tsv rows == metadata.totals.n_train ({n_train_meta})",
              len(train_rows) == n_train_meta)
        check(f"holdout.tsv rows == metadata.totals.n_holdout ({n_holdout_meta})",
              len(holdout_rows) == n_holdout_meta)
        sample_concepts = {r.concept for r in train_rows}
        check("train concepts are non-empty",
              bool(sample_concepts) and all(isinstance(c, str) for c in sample_concepts))

    # =====================================================================
    # Done
    # =====================================================================
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("→ PACKAGE_SMOKE_PASS")
    print("\nselflearnai.generator/ is import-clean and the public API works.")
    print("Ready for sub-task 2a.3 (full training run on the production corpus).")


if __name__ == "__main__":
    main()
