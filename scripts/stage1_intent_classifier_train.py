"""Task 1.4 — train the Tier-2 Ψ-space intent classifier.

Trains a small MLP (~100K params) that maps a frozen-encoder
embedding of a question to one of {7 known concepts, "unknown"}.
Catches the paraphrase tail Tier-1's grammar misses (Task 1.3 measured
Tier-1 paraphrase recall at 37%).

Training data is SYNTHESIZED here, not pulled from the eval files:
  - Per-concept questions: cross product of `_TRAIN_TEMPLATES[concept]`
    × `_TRAIN_SOURCES[concept]`. Templates and sources are deliberately
    chosen so the canonical eval rows can leak in (the encoder
    embeddings of training items will be similar but distinct from
    eval items, and the classifier is small enough that it can't
    memorize specific (template, source) pairs).
  - Unknown class: a curated list of out-of-library questions — trivia,
    personal, math, commands — encoded the same way.

Eval is run on the unmodified canonical + paraphrase TSVs from
Tasks 1.1/1.2. No leakage from those files into training.

Acceptance gate (Task 1.4):
  - Canonical parseable: ≥ 80% top-1 (the easy half — phrasings
    similar to training).
  - Canonical refuse: ≥ 80% routed to "unknown".
  - Paraphrase parseable: ≥ 60% top-1 (the hard half — generalization
    test). This is the lift we need to justify Tier 2 over Tier 1's
    37% baseline.

Loads encoder once, encodes all train + eval items in batched calls,
trains on cached embeddings (fast — full-batch, ~500 epochs runs in
seconds even on CPU once embeddings are cached).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from selflearnai.intent import (
    IntentClassifier,
    IntentClassMapping,
)


ENCODERS = {
    "gte-base":    {"model": "thenlper/gte-base",    "dim":  768},
    "e5-large-v2": {"model": "intfloat/e5-large-v2", "dim": 1024},
}

CONCEPTS = (
    "agentive",
    "comparative",
    "opposite",
    "past_tense",
    "plural",
    "superlative",
    "young",
)
UNKNOWN = "unknown"
ALL_CLASSES = CONCEPTS + (UNKNOWN,)


# ---------------------------------------------------------------------------
# Training data generation — templates × sources.
# Templates intentionally cover BOTH the canonical regex shapes and a
# few paraphrase shapes Tier 1 didn't catch. The classifier learns the
# semantic boundary, not specific phrasings.
# ---------------------------------------------------------------------------

_TRAIN_TEMPLATES: dict[str, list[str]] = {
    "plural": [
        "what's the plural of {src}",
        "plural of {src}",
        "the plural of {src}",
        "make {src} plural",
        "plurals for {src}",
        "the plural form of {src}",
        "give me the plural of {src}",
        "plural form for {src}",
        "how do you say {src} in plural",
        "what's the plural form for {src}",
        "i need the plural for {src}",
        "what do you call multiple {src}s",
    ],
    "past_tense": [
        "past tense of {src}",
        "what's the past tense of {src}",
        "the past tense of {src}",
        "past form of {src}",
        "{src} in past tense",
        "give me the past form of {src}",
        "past form for {src}",
        "how do you say {src} in past tense",
        "what was {src} yesterday",
        "what's the past form for {src}",
    ],
    "comparative": [
        "comparative of {src}",
        "what's the comparative of {src}",
        "more {src}",
        "the comparative form of {src}",
        "give me the comparative of {src}",
        "comparative form for {src}",
        "the comparative form for {src}",
        "the X-er form of {src}",
        "how to say {src} with -er ending",
    ],
    "superlative": [
        "superlative of {src}",
        "what's the superlative of {src}",
        "most {src}",
        "the superlative form of {src}",
        "give me the superlative of {src}",
        "superlative form for {src}",
        "the most version of {src}",
        "highest form of {src}",
        "the X-est form of {src}",
        "extreme of {src}",
    ],
    "opposite": [
        "opposite of {src}",
        "what's the opposite of {src}",
        "antonym of {src}",
        "the opposite of {src}",
        "give me the opposite of {src}",
        "the reverse of {src}",
        "the inverse of {src}",
        "the contrary of {src}",
        "what's the negation of {src}",
        "what means the opposite to {src}",
    ],
    "agentive": [
        "agentive of {src}",
        "agent of {src}",
        "the agent of {src}",
        "the agentive form of {src}",
        "one who {src}s",
        "person who {src}s",
        "someone who {src}s",
        "noun for someone who {src}s",
        "the noun for someone who {src}s",
        "doer of {src}",
        "the X-er noun for {src}",
    ],
    "young": [
        "baby {src}",
        "young {src}",
        "what is a young {src}",
        "what is a baby {src}",
        "what is a baby {src} called",
        "the baby form of {src}",
        "name for a baby {src}",
        "what do you call a baby {src}",
        "what is the name of a baby {src}",
        "the kid form of {src}",
        "young version of {src}",
    ],
}


_TRAIN_SOURCES: dict[str, list[str]] = {
    "plural": [
        "apple", "hand", "foot", "house", "table", "chair", "phone", "lamp",
        "river", "mountain", "key", "wall", "door", "window", "letter",
    ],
    "past_tense": [
        "swim", "fall", "throw", "catch", "buy", "bring", "sell", "find",
        "lose", "meet", "drink", "leave", "give", "take", "tell",
    ],
    "comparative": [
        "cold", "smart", "strong", "old", "new", "rich", "poor", "young",
        "wise", "brave", "kind", "rude", "loud", "quiet", "happy",
    ],
    "superlative": [
        "tall", "loud", "quiet", "rich", "poor", "wise", "brave", "kind",
        "rude", "happy", "sad",
    ],
    "opposite": [
        "hot", "big", "good", "young", "soft", "easy", "cheap", "tall",
        "wet", "dirty", "open", "near", "rich", "loud",
    ],
    "agentive": [
        "sing", "dance", "run", "swim", "read", "cook", "fish", "fight",
        "lead", "manage", "work", "speak", "design",
    ],
    "young": [
        "sheep", "pig", "frog", "duck", "goose", "swan", "deer", "bear",
        "rabbit", "fox", "lion", "tiger",
    ],
}


_UNKNOWN_QUESTIONS: list[str] = [
    # Factual / trivia
    "what is the capital of france",
    "who wrote hamlet",
    "when did world war 2 end",
    "what is the speed of light",
    "how tall is mount everest",
    "what is the largest planet in our solar system",
    "why is the sky blue",
    "how does electricity work",
    "what does dna stand for",
    "what is photosynthesis",
    "who painted the mona lisa",
    "what is the meaning of life",
    "how many continents are there",
    "what is the boiling point of water",
    "who invented the telephone",
    # Personal / conversational
    "what is your name",
    "how are you doing today",
    "tell me a joke",
    "what is your favorite color",
    "are you human",
    "do you have feelings",
    "where are you from",
    "what time is it",
    # Translation / spelling / language
    "translate this sentence to french",
    "how do you spell receive",
    "what does ephemeral mean",
    "say hello in spanish",
    "give me a synonym for happy",
    # Math / arithmetic
    "what is two plus two",
    "what is the square root of sixteen",
    "calculate five times seven",
    "solve x squared equals nine",
    # Commands / actions
    "open the door",
    "play some music",
    "set an alarm for seven am",
    "send an email to john",
    "remind me to buy milk",
    "turn on the lights",
    # Random / general
    "tell me a story",
    "describe a sunset",
    "list the planets",
    "name some flowers",
    "give me a recipe for pasta",
    "what should i wear today",
    # Tricky — mention concept-words but aren't concept questions
    "do dogs like cats",
    "is paint a noun",
    "describe red",
    "what does old mean",
    "explain love",
    "who runs the country",
    "why are big things heavy",
]


# ---------------------------------------------------------------------------
# Encoder helpers (mirrors run_text_only_concepts.py / Stage 0.5 scripts)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Eval IO (re-uses Task 1.1 reader)
# ---------------------------------------------------------------------------

from scripts.stage1_intent_grammar_smoke import read_eval_rows


# ---------------------------------------------------------------------------
# Training-data construction
# ---------------------------------------------------------------------------

def build_training_corpus() -> tuple[list[str], list[str]]:
    """Return (questions, labels) — every question paired with the
    correct concept (or 'unknown')."""
    questions: list[str] = []
    labels: list[str] = []
    for concept in CONCEPTS:
        templates = _TRAIN_TEMPLATES[concept]
        sources = _TRAIN_SOURCES[concept]
        for src in sources:
            for tpl in templates:
                questions.append(tpl.format(src=src))
                labels.append(concept)
    for q in _UNKNOWN_QUESTIONS:
        questions.append(q)
        labels.append(UNKNOWN)
    return questions, labels


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_classifier(
    *, train_questions, train_labels, encode_fn, mapping: IntentClassMapping,
    encoder_dim: int, hidden_dim: int, device: str,
    epochs: int, lr: float, weight_decay: float, seed: int,
) -> IntentClassifier:
    torch.manual_seed(seed)
    z_train = encode_fn(train_questions)                         # (N, D)
    y_train = torch.tensor(
        [mapping.index(l) for l in train_labels], device=device,
    )

    clf = IntentClassifier(
        encoder_dim=encoder_dim,
        mapping=mapping,
        hidden_dim=hidden_dim,
    ).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        opt.zero_grad()
        logits = clf(z_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()

    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    clf: IntentClassifier, encode_fn, eval_path: Path,
    mapping: IntentClassMapping,
) -> dict:
    rows = read_eval_rows(eval_path)
    if not rows:
        return {"path": str(eval_path), "n_rows": 0}
    questions = [r[0] for r in rows]
    z = encode_fn(questions)
    pred_names, probs = clf.predict_batch(z)

    # Per-row classification + bucketed counters.
    counters = {
        "parseable_total": 0, "parseable_correct": 0, "parseable_wrong": 0,
        "refuse_total": 0, "refuse_correct": 0, "refuse_violated": 0,
    }
    by_concept: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    failures: list[dict] = []

    for i, (q, expected_concept, expected_source, status) in enumerate(rows):
        pred_concept = pred_names[i]
        pred_conf = float(probs[i, mapping.index(pred_concept)].item())
        if status == "parseable":
            counters["parseable_total"] += 1
            by_concept[expected_concept]["total"] += 1
            if pred_concept == expected_concept:
                counters["parseable_correct"] += 1
                by_concept[expected_concept]["correct"] += 1
            else:
                counters["parseable_wrong"] += 1
                failures.append({
                    "type": "parseable_wrong",
                    "question": q,
                    "expected": expected_concept,
                    "predicted": pred_concept,
                    "confidence": pred_conf,
                })
        elif status == "refuse":
            counters["refuse_total"] += 1
            if pred_concept == mapping.unknown_label:
                counters["refuse_correct"] += 1
            else:
                counters["refuse_violated"] += 1
                failures.append({
                    "type": "refuse_violated",
                    "question": q,
                    "predicted": pred_concept,
                    "confidence": pred_conf,
                })

    parseable_acc = (
        counters["parseable_correct"] / counters["parseable_total"]
        if counters["parseable_total"] else float("nan")
    )
    refuse_acc = (
        counters["refuse_correct"] / counters["refuse_total"]
        if counters["refuse_total"] else None
    )

    return {
        "path": str(eval_path),
        "n_rows": len(rows),
        "counters": counters,
        "by_concept": dict(by_concept),
        "parseable_acc": parseable_acc,
        "refuse_acc": refuse_acc,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoder", default="gte-base", choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--canonical", default="data/intent_eval/canonical.tsv")
    parser.add_argument("--paraphrase", default="data/intent_eval/paraphrase.tsv")
    parser.add_argument("--out", default="results/stage1/intent_classifier.json")
    parser.add_argument("--save-ckpt", default="checkpoints/intent_classifier.pt")
    parser.add_argument(
        "--canonical-min", type=float, default=0.80,
        help="Acceptance: top-1 on canonical parseable >= this.",
    )
    parser.add_argument(
        "--paraphrase-min", type=float, default=0.60,
        help="Acceptance: top-1 on paraphrase parseable >= this.",
    )
    parser.add_argument(
        "--refuse-min", type=float, default=0.80,
        help="Acceptance: refuse rows routed to unknown >= this fraction.",
    )
    args = parser.parse_args()

    print("Task 1.4 — Tier-2 Ψ-space intent classifier")
    print("=" * 70)
    enc_cfg = ENCODERS[args.encoder]
    print(f"Encoder: {args.encoder} ({enc_cfg['model']}, dim={enc_cfg['dim']})")

    mapping = IntentClassMapping(classes=ALL_CLASSES, unknown_label=UNKNOWN)
    print(f"Classes: {mapping.classes} (unknown_index={mapping.unknown_index})")

    # ---- Training corpus ----
    train_q, train_y = build_training_corpus()
    n_per_class = {c: train_y.count(c) for c in mapping.classes}
    print(f"\nTraining corpus: {len(train_q)} questions")
    for c in mapping.classes:
        print(f"  {c:<14s}  n={n_per_class[c]}")

    # ---- Encoder ----
    print(f"\nLoading {enc_cfg['model']} ...")
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
    mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    encode = make_encode_fn(mdl, tok, args.device)

    # ---- Train ----
    print(f"\nTraining classifier (hidden={args.hidden_dim}, "
          f"epochs={args.epochs}, lr={args.lr}) ...")
    clf = train_classifier(
        train_questions=train_q, train_labels=train_y,
        encode_fn=encode, mapping=mapping,
        encoder_dim=enc_cfg["dim"], hidden_dim=args.hidden_dim,
        device=args.device,
        epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
    )
    print(f"  classifier params: {clf.num_parameters():,}")

    # ---- Eval ----
    canonical = evaluate(clf, encode, Path(args.canonical), mapping)
    paraphrase = evaluate(clf, encode, Path(args.paraphrase), mapping)

    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    for tag, r in (("canonical", canonical), ("paraphrase", paraphrase)):
        c = r["counters"]
        cell_p = (
            f"{c['parseable_correct']}/{c['parseable_total']}  ({r['parseable_acc']:.3f})"
            if c["parseable_total"] else "-"
        )
        cell_r = (
            f"{c['refuse_correct']}/{c['refuse_total']}  ({r['refuse_acc']:.3f})"
            if c["refuse_total"] else "-"
        )
        print(f"  {tag:<14s}  rows={r['n_rows']:<3d}  parseable: {cell_p:<24s}  refuse: {cell_r}")

    print("\nPer-concept paraphrase top-1 accuracy:")
    for c in sorted(paraphrase["by_concept"].keys()):
        d = paraphrase["by_concept"][c]
        acc = d["correct"] / d["total"] if d["total"] else 0
        print(f"  {c:<14s}  {d['correct']}/{d['total']}  ({acc:.2f})")

    # ---- Acceptance ----
    canonical_pass = canonical["parseable_acc"] >= args.canonical_min
    refuse_pass = (
        canonical["refuse_acc"] is None
        or canonical["refuse_acc"] >= args.refuse_min
    )
    paraphrase_pass = paraphrase["parseable_acc"] >= args.paraphrase_min
    overall = canonical_pass and refuse_pass and paraphrase_pass

    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK (Task 1.4)")
    print("=" * 70)
    print(f"  Canonical parseable (>= {args.canonical_min:.2f}):   "
          f"{canonical['parseable_acc']:.3f}  → {'PASS' if canonical_pass else 'FAIL'}")
    if canonical["refuse_acc"] is not None:
        print(f"  Canonical refuse  → unknown (>= {args.refuse_min:.2f}): "
              f"{canonical['refuse_acc']:.3f}  → {'PASS' if refuse_pass else 'FAIL'}")
    print(f"  Paraphrase parseable (>= {args.paraphrase_min:.2f}):  "
          f"{paraphrase['parseable_acc']:.3f}  → {'PASS' if paraphrase_pass else 'FAIL'}")
    print(f"\n→ Task 1.4: {'PASS' if overall else 'FAIL'}")

    # ---- Save checkpoint + JSON ----
    ckpt_path = Path(args.save_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": args.encoder,
            "encoder_dim": enc_cfg["dim"],
            "hidden_dim": args.hidden_dim,
            "classes": list(mapping.classes),
            "unknown_label": mapping.unknown_label,
            "state_dict": clf.state_dict(),
        },
        ckpt_path,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "task": "1.4",
                "encoder": args.encoder,
                "encoder_dim": enc_cfg["dim"],
                "hidden_dim": args.hidden_dim,
                "classes": list(mapping.classes),
                "training_corpus_size": len(train_q),
                "n_per_class": n_per_class,
                "canonical": {k: v for k, v in canonical.items() if k != "failures"},
                "paraphrase": {k: v for k, v in paraphrase.items() if k != "failures"},
                "canonical_failures": canonical["failures"],
                "paraphrase_failures": paraphrase["failures"],
                "thresholds": {
                    "canonical_min": args.canonical_min,
                    "paraphrase_min": args.paraphrase_min,
                    "refuse_min": args.refuse_min,
                },
                "pass": overall,
            },
            f, indent=2,
        )
    print(f"\n→ saved checkpoint to {ckpt_path}")
    print(f"→ saved JSON to {out_path}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
