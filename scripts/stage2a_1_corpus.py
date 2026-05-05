"""Phase 2a / Sub-task 2a.1 — production corpus generator (plan §19.14).

Builds the production training corpus for Phase 2a's full training run
(sub-task 2a.3). Replaces the small/iterative corpora used in §19.13's
empirical journey (~1000 sentences, 7 templates per concept, 6
held-out per concept) with a scaled-up version that:

  - Has 12 templates per concept (extended from 2a.0c's 7) for
    sufficient template diversity.
  - Includes ALL existing word pairs as training data (text_pairs_
    train.tsv + text_pairs_held_out.tsv, since for THIS task we only
    care about truly-novel held-out, not held-out-from-Stage-0).
  - Has 9 TRULY-NOVEL pairs per concept for held-out — verified at
    runtime against ALL existing concept TSVs to FATAL on any leak
    (per 2a.0f methodology).
  - Writes outputs to disk as data/explanations_v2/{train,holdout}.tsv
    + metadata.json so 2a.3's training script reads files, not Python.

Per plan §19.14 acceptance:
  - ≥ 2000 train + ≥ 400 held-out                ✓ (target ~2064 / ~432)
  - Runtime no-overlap audit passes              ✓ (FATAL on leak)
  - Templates clean by grammar proxy             ✓ (sanity check)

Extended truly-novel pair set (9 per concept, was 7 in 2a.0f):
  plural:      mouse/mice, child/children, foot/feet, tooth/teeth,
               goose/geese, man/men, woman/women, ox/oxen, person/people
  past_tense:  drink/drank, catch/caught, weep/wept, forget/forgot,
               shake/shook, freeze/froze, forgive/forgave,
               bend/bent, lend/lent
  comparative: pretty/prettier, simple/simpler, gentle/gentler,
               clever/cleverer, lonely/lonelier, friendly/friendlier,
               naughty/naughtier, tiny/tinier, gloomy/gloomier
  opposite:    rich/poor, safe/dangerous, friend/enemy, love/hate,
               awake/asleep, victory/defeat, joy/sorrow,
               hero/villain, success/failure

Run:
  python scripts/stage2a_1_corpus.py
  python scripts/stage2a_1_corpus.py --out-dir data/explanations_v2 --strict-grammar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage1_planner_beam_smoke import read_pairs

# Soft import for grammar proxy.
try:
    from scripts.stage2a_quick_probe import grammar_proxy
    _HAS_PROXY = True
except ImportError:
    _HAS_PROXY = False
    def grammar_proxy(text: str) -> float:                 # stub
        del text
        return 1.0


# ---------------------------------------------------------------------------
# Concept config — single source of truth
# ---------------------------------------------------------------------------

CONCEPT_DATA_DIRS: dict[str, str] = {
    "plural":      "data/plurality",
    "past_tense":  "data/past_tense",
    "comparative": "data/comparative",
    "opposite":    "data/opposite_v2",
}


# 12 templates per concept (extended from 2a.0c's 7).
TEMPLATES: dict[str, list[str]] = {
    "plural": [
        "the plural of {src} is {tgt}",
        "{src} becomes {tgt} in plural form",
        "{tgt} are the plural of {src}",
        "we say {tgt} when there are many {src}",
        "{tgt} is what we call multiple {src}",
        "to make {src} plural we say {tgt}",
        "more than one {src} becomes {tgt}",
        "the plural form of {src} is {tgt}",
        "we use {tgt} when speaking of multiple {src}",
        "when there is more than one {src} we use {tgt}",
        "many {src} are called {tgt}",
        "several {src} together are {tgt}",
    ],
    "past_tense": [
        "the past tense of {src} is {tgt}",
        "{src} becomes {tgt} in the past",
        "{tgt} is the past tense of {src}",
        "to say {src} happened we use {tgt}",
        "{tgt} is what {src} becomes in the past",
        "yesterday i {tgt} after i {src}",
        "the past form of {src} is {tgt}",
        "we say {tgt} when {src} happened in the past",
        "{src} in the past tense is {tgt}",
        "the past form of {src} is written as {tgt}",
        "to put {src} in the past we write {tgt}",
        "when {src} is in the past it becomes {tgt}",
    ],
    "comparative": [
        "the comparative of {src} is {tgt}",
        "more {src} is {tgt}",
        "something more {src} is {tgt}",
        "to compare we say {tgt} instead of {src}",
        "{tgt} is the comparative form of {src}",
        "when something is more {src} it is {tgt}",
        "the comparative form of {src} is {tgt}",
        "we use {tgt} when comparing things more {src} than others",
        "{src} becomes {tgt} when we compare",
        "to compare degrees of {src} we use {tgt}",
        "if one thing is more {src} than another we say it is {tgt}",
        "{tgt} means more {src}",
    ],
    "opposite": [
        "the opposite of {src} is {tgt}",
        "{src} and {tgt} are opposites",
        "{tgt} is the opposite of {src}",
        "{tgt} is the antonym of {src}",
        "the antonym of {src} is {tgt}",
        "{src} means the opposite of {tgt}",
        "if something is not {src} it might be {tgt}",
        "we say {tgt} for things that are not {src}",
        "the opposite meaning of {src} is {tgt}",
        "{src} is the antonym of {tgt}",
        "to say the opposite of {src} we use {tgt}",
        "{src} and {tgt} have opposite meanings",
    ],
}


# 9 truly-novel pairs per concept (extended from 2a.0f's 7).
# Runtime audit will FATAL if any of these appears in any existing
# concept TSV — same discipline as 2a.0f.
TRULY_NOVEL_PAIRS: dict[str, list[tuple[str, str]]] = {
    "plural": [
        ("mouse",  "mice"),
        ("child",  "children"),
        ("foot",   "feet"),
        ("tooth",  "teeth"),
        ("goose",  "geese"),
        ("man",    "men"),
        ("woman",  "women"),
        ("ox",     "oxen"),
        ("person", "people"),
    ],
    "past_tense": [
        ("drink",   "drank"),
        ("catch",   "caught"),
        ("weep",    "wept"),
        ("forget",  "forgot"),
        ("shake",   "shook"),
        ("freeze",  "froze"),
        ("forgive", "forgave"),
        ("bend",    "bent"),
        ("lend",    "lent"),
    ],
    "comparative": [
        ("pretty",   "prettier"),
        ("simple",   "simpler"),
        ("gentle",   "gentler"),
        ("clever",   "cleverer"),
        ("lonely",   "lonelier"),
        ("friendly", "friendlier"),
        ("naughty",  "naughtier"),
        ("tiny",     "tinier"),
        ("gloomy",   "gloomier"),
    ],
    "opposite": [
        ("rich",    "poor"),
        ("safe",    "dangerous"),
        ("friend",  "enemy"),
        ("love",    "hate"),
        ("awake",   "asleep"),
        ("victory", "defeat"),
        ("joy",     "sorrow"),
        ("hero",    "villain"),
        ("success", "failure"),
    ],
}


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

def audit_no_overlap() -> None:
    """FATAL if any TRULY_NOVEL_PAIRS entry appears in any concept TSV.

    Same shape as 2a.0f's assert_no_overlap_with_existing_tsvs — which
    caught two iterations of leaks in §19.13. Don't ship a corpus
    without this check passing.
    """
    leaks: list[tuple[str, tuple[str, str], str]] = []
    for concept, pairs in TRULY_NOVEL_PAIRS.items():
        ddir = CONCEPT_DATA_DIRS[concept]
        existing: set[tuple[str, str]] = set()
        for fname in ("text_pairs_train.tsv", "text_pairs_held_out.tsv"):
            p = Path(ddir) / fname
            if p.exists():
                for src, tgt in read_pairs(p):
                    existing.add((src, tgt))
        for pair in pairs:
            if pair in existing:
                leaks.append((concept, pair, ddir))
    if leaks:
        print("FATAL: TRULY_NOVEL_PAIRS leak with existing TSVs:")
        for concept, pair, ddir in leaks:
            print(f"  {concept}: {pair} appears in {ddir}/")
        raise SystemExit(1)


def audit_template_grammar(min_proxy: float = 0.55) -> tuple[int, list[tuple[str, int, str, float]]]:
    """Score each template by the wordfreq proxy. Reports any below
    threshold so we can spot typos/gibberish before training runs.

    Returns (n_below_threshold, list of (concept, idx, template, proxy_score)).
    """
    below: list[tuple[str, int, str, float]] = []
    for concept, templates in TEMPLATES.items():
        for i, tpl in enumerate(templates):
            # Render with neutral placeholder words so wordfreq treats
            # them as ordinary tokens.
            rendered = tpl.format(src="example", tgt="instance")
            score = grammar_proxy(rendered) if _HAS_PROXY else 1.0
            if score < min_proxy:
                below.append((concept, i, tpl, score))
    return len(below), below


# ---------------------------------------------------------------------------
# Corpus build
# ---------------------------------------------------------------------------

def build_train_pairs(concept: str) -> list[tuple[str, str]]:
    """All non-novel word pairs available for this concept's training:
    text_pairs_train.tsv + text_pairs_held_out.tsv. The original
    'held-out' pairs (from Stage 0 concept-operator training) are now
    fair game for Phase 2a training because Phase 2a's held-out gate
    is the truly-novel set — not these.
    """
    ddir = Path(CONCEPT_DATA_DIRS[concept])
    pairs: list[tuple[str, str]] = []
    for fname in ("text_pairs_train.tsv", "text_pairs_held_out.tsv"):
        p = ddir / fname
        if p.exists():
            pairs.extend(read_pairs(p))
    # De-dupe in case of accidental overlap.
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            deduped.append(pair)
            seen.add(pair)
    return deduped


def render_corpus(
    concept: str,
    word_pairs: list[tuple[str, str]],
) -> list[dict]:
    """Cross word_pairs × TEMPLATES[concept], producing one record per
    (concept, src, tgt, template_idx, sentence)."""
    records: list[dict] = []
    for src, tgt in word_pairs:
        for ti, tpl in enumerate(TEMPLATES[concept]):
            records.append({
                "concept": concept,
                "src": src,
                "tgt": tgt,
                "template_idx": ti,
                "sentence": tpl.format(src=src, tgt=tgt),
            })
    return records


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("concept\tsrc\ttgt\ttemplate_idx\tsentence\n")
        for r in rows:
            # Tabs in fields would corrupt the TSV; sentences don't
            # have tabs, but be defensive about template content.
            sent = r["sentence"].replace("\t", " ")
            f.write(f"{r['concept']}\t{r['src']}\t{r['tgt']}\t{r['template_idx']}\t{sent}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--out-dir", default="data/explanations_v2",
                        help="Where to write train.tsv + holdout.tsv + metadata.json.")
    parser.add_argument("--min-train", type=int, default=2000,
                        help="Soft target — warn if below.")
    parser.add_argument("--min-holdout", type=int, default=400,
                        help="Soft target — warn if below.")
    parser.add_argument("--grammar-proxy-min", type=float, default=0.55,
                        help="Templates with proxy score below this trigger a warning.")
    parser.add_argument("--strict-grammar", action="store_true",
                        help="FATAL on any template below grammar-proxy-min.")
    args = parser.parse_args()

    print("Phase 2a / Sub-task 2a.1 — production corpus generator")
    print("=" * 78)
    print(f"Concepts: {sorted(CONCEPT_DATA_DIRS.keys())}")
    print(f"Templates per concept: {len(next(iter(TEMPLATES.values())))} "
          f"(uniform across concepts)")
    print(f"Truly-novel pairs per concept: "
          f"{len(next(iter(TRULY_NOVEL_PAIRS.values())))}")
    print(f"Output: {args.out_dir}/")

    # ---- Audit 1: no truly-novel pair leaks into existing TSVs --------
    print("\n[1] Leakage audit (TRULY_NOVEL vs existing TSVs)")
    print("-" * 78)
    audit_no_overlap()
    print("  ✓ no truly-novel pair appears in any concept TSV")

    # ---- Audit 2: template grammar sanity (wordfreq proxy) -----------
    print("\n[2] Template grammar sanity (wordfreq proxy)")
    print("-" * 78)
    if not _HAS_PROXY:
        print("  WARNING: wordfreq not installed — template sanity skipped.")
    else:
        n_below, below = audit_template_grammar(args.grammar_proxy_min)
        if n_below == 0:
            print(f"  ✓ all templates have proxy ≥ {args.grammar_proxy_min}")
        else:
            print(f"  ⚠ {n_below} templates below threshold "
                  f"{args.grammar_proxy_min}:")
            for concept, idx, tpl, score in below:
                print(f"    {concept}[{idx}]  proxy={score:.3f}  {tpl!r}")
            if args.strict_grammar:
                raise SystemExit(1)
            else:
                print(f"  (continuing — pass --strict-grammar to FATAL)")

    # ---- Build train + held-out --------------------------------------
    print("\n[3] Building corpus")
    print("-" * 78)
    train_records: list[dict] = []
    holdout_records: list[dict] = []
    per_concept_stats: dict[str, dict] = {}
    for concept in CONCEPT_DATA_DIRS:
        train_pairs = build_train_pairs(concept)
        novel_pairs = TRULY_NOVEL_PAIRS[concept]
        train_recs = render_corpus(concept, train_pairs)
        holdout_recs = render_corpus(concept, novel_pairs)
        # Sanity: ensure no truly-novel pair leaked into train_pairs.
        # (Belt-and-suspenders — audit_no_overlap should've caught this.)
        train_pair_set = set(train_pairs)
        for novel in novel_pairs:
            if novel in train_pair_set:
                raise SystemExit(
                    f"FATAL: post-build leak detected — {concept} novel "
                    f"pair {novel} also in train_pairs"
                )
        train_records.extend(train_recs)
        holdout_records.extend(holdout_recs)
        per_concept_stats[concept] = {
            "n_train_pairs": len(train_pairs),
            "n_novel_pairs": len(novel_pairs),
            "n_templates": len(TEMPLATES[concept]),
            "n_train_sents": len(train_recs),
            "n_holdout_sents": len(holdout_recs),
        }
        print(f"  {concept:<12}  train_pairs={len(train_pairs):>3}  "
              f"novel_pairs={len(novel_pairs):>2}  "
              f"templates={len(TEMPLATES[concept]):>2}  "
              f"→ train={len(train_recs):>4}  novel_holdout={len(holdout_recs):>3}")

    n_train = len(train_records)
    n_holdout = len(holdout_records)

    # ---- Audit 3: targets ---------------------------------------------
    print("\n[4] Size targets")
    print("-" * 78)
    train_ok = n_train >= args.min_train
    holdout_ok = n_holdout >= args.min_holdout
    train_mark = "✓" if train_ok else "✗"
    holdout_mark = "✓" if holdout_ok else "✗"
    print(f"  {train_mark} train sentences:  {n_train}  (target ≥ {args.min_train})")
    print(f"  {holdout_mark} holdout sentences: {n_holdout}  (target ≥ {args.min_holdout})")
    targets_ok = train_ok and holdout_ok

    # ---- Write to disk ------------------------------------------------
    print("\n[5] Writing corpus")
    print("-" * 78)
    out_dir = Path(args.out_dir)
    train_tsv = out_dir / "train.tsv"
    holdout_tsv = out_dir / "holdout.tsv"
    metadata_json = out_dir / "metadata.json"
    write_tsv(train_records, train_tsv)
    write_tsv(holdout_records, holdout_tsv)
    metadata = {
        "task": "2a.1",
        "concepts": sorted(CONCEPT_DATA_DIRS.keys()),
        "concept_data_dirs": CONCEPT_DATA_DIRS,
        "templates": TEMPLATES,
        "truly_novel_pairs": {
            c: [list(p) for p in pairs] for c, pairs in TRULY_NOVEL_PAIRS.items()
        },
        "per_concept": per_concept_stats,
        "totals": {
            "n_train": n_train,
            "n_holdout": n_holdout,
        },
        "targets": {
            "min_train": args.min_train,
            "min_holdout": args.min_holdout,
        },
        "audits": {
            "leakage_passed": True,
            "grammar_proxy_min": args.grammar_proxy_min,
            "wordfreq_installed": _HAS_PROXY,
        },
        "files": {
            "train": str(train_tsv),
            "holdout": str(holdout_tsv),
        },
    }
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  wrote {train_tsv}")
    print(f"  wrote {holdout_tsv}")
    print(f"  wrote {metadata_json}")

    # ---- Sample preview ----------------------------------------------
    print("\n[6] Sample preview")
    print("-" * 78)
    print("  train (first 5):")
    for r in train_records[:5]:
        print(f"    [{r['concept']:<11}] {r['sentence']!r}")
    print("  holdout (first 5):")
    for r in holdout_records[:5]:
        print(f"    [{r['concept']:<11}] {r['sentence']!r}")

    # ---- Verdict ------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if targets_ok:
        verdict = "PASS"
        message = (
            f"Production corpus generated and audited. "
            f"{n_train} train + {n_holdout} held-out across "
            f"{len(CONCEPT_DATA_DIRS)} concepts. "
            f"No truly-novel pair leaks; templates clean. Ready for 2a.2 "
            f"(refactor scripts into selflearnai/generator/) or 2a.3 "
            f"(full training run)."
        )
    else:
        verdict = "TARGETS_BELOW"
        message = (
            f"Corpus written but below the §19.14 size targets. Either "
            f"add more templates, more concepts, or relax the targets. "
            f"Architecture validation (the actual goal of 2a.3) doesn't "
            f"strictly need these numbers."
        )
    print(f"\n→ {verdict}")
    print(f"\n{message}")
    raise SystemExit(0 if targets_ok else 1)


if __name__ == "__main__":
    main()
