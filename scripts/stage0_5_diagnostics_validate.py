"""Task 0.5.4 — validate the encoder calibration diagnostic test sets.

Reads `data/encoder_diagnostics/{antonym, code_struct, paraphrase}/triples.tsv`
and runs four checks:

  1. Parse cleanly (3 columns per row, after stripping inline comments).
  2. Expected row counts (30, 10, 30).
  3. No within-file duplicate rows.
  4. No within-row leak (anchor != positive != negative).

If --encode is passed, an additional check encodes a small sample and
verifies the per-row diagnostic score has a non-degenerate range (max
− min over rows > 0.05). This rules out the case where the encoder
gives all-rows-the-same score (trivially passing structurally but not
discriminating) — the data sets must produce meaningful signal.

Acceptance gate (Task 0.5.4):
  All four structural checks PASS. With --encode, additional gate:
  per-test-family score range > 0.05 on at least one encoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch / numpy only needed for the optional --encode path; lazy-imported
# inside the encoder-check function so structural-only runs work without
# them (e.g., on a workstation that doesn't have CUDA installed).


TEST_FAMILIES: list[dict] = [
    {"name": "antonym",     "path": "data/encoder_diagnostics/antonym/triples.tsv",     "expected_rows": 30},
    {"name": "code_struct", "path": "data/encoder_diagnostics/code_struct/triples.tsv", "expected_rows": 10},
    {"name": "paraphrase",  "path": "data/encoder_diagnostics/paraphrase/triples.tsv",  "expected_rows": 30},
]

ENCODERS = {
    "gte-base":    {"model": "thenlper/gte-base",    "dim":  768},
    "e5-large-v2": {"model": "intfloat/e5-large-v2", "dim": 1024},
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _strip(s: str) -> str:
    """Strip inline TSV comments. Code snippets sometimes contain '#' as
    Python comments, so for the diagnostic data we forbid leading-whitespace
    '#' inline (only full-line comments are allowed). The simpler rule:
    if a '#' appears OUTSIDE the column boundaries it's a problem; here we
    just don't strip inline '#' to preserve code snippets verbatim."""
    return s.strip()


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"{path}: row has < 3 tab-separated columns: {line!r}"
                )
            a, p, n = _strip(parts[0]), _strip(parts[1]), _strip(parts[2])
            if not a or not p or not n:
                raise ValueError(f"{path}: empty field in row: {line!r}")
            rows.append((a, p, n))
    return rows


# ---------------------------------------------------------------------------
# Structural checks (no encoder needed)
# ---------------------------------------------------------------------------

def structural_check(family: dict) -> dict:
    path = Path(family["path"])
    if not path.exists():
        return {"name": family["name"], "pass": False, "reason": f"missing file {path}"}

    try:
        rows = read_triples(path)
    except ValueError as e:
        return {"name": family["name"], "pass": False, "reason": f"parse error: {e}"}

    # Count check
    n = len(rows)
    if n != family["expected_rows"]:
        return {
            "name": family["name"], "pass": False,
            "reason": f"expected {family['expected_rows']} rows, got {n}",
        }

    # Duplicate rows
    seen: set[tuple[str, str, str]] = set()
    dups: list[tuple[str, str, str]] = []
    for r in rows:
        if r in seen:
            dups.append(r)
        seen.add(r)
    if dups:
        return {
            "name": family["name"], "pass": False,
            "reason": f"{len(dups)} duplicate rows: {dups[:3]}{'...' if len(dups) > 3 else ''}",
        }

    # Within-row leaks
    leaks = [r for r in rows if r[0] == r[1] or r[0] == r[2] or r[1] == r[2]]
    if leaks:
        return {
            "name": family["name"], "pass": False,
            "reason": f"{len(leaks)} within-row leaks: {leaks[:3]}",
        }

    return {
        "name": family["name"], "pass": True, "n_rows": n,
        "n_unique": len(seen),
    }


# ---------------------------------------------------------------------------
# Encoder check (optional)
# ---------------------------------------------------------------------------

def make_encode_fn(model, tokenizer, device: str, max_length: int = 128):
    """Build a list[str] -> Tensor closure with mean-pooled embeddings.
    torch is imported lazily by the caller; we just close over it here."""
    import torch

    @torch.no_grad()
    def encode(words):
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


def encoded_score_check(family: dict, encode_fn, *, range_threshold: float = 0.05) -> dict:
    """Compute per-row diagnostic score = cos(anchor, positive) -
    cos(anchor, negative). Verify the score range across rows is
    non-degenerate (max - min > range_threshold).

    Lazy-imports torch + numpy so structural-only runs (without --encode)
    work on workstations that lack the GPU stack.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    rows = read_triples(Path(family["path"]))
    anchors = [r[0] for r in rows]
    positives = [r[1] for r in rows]
    negatives = [r[2] for r in rows]
    with torch.no_grad():
        z_a = encode_fn(anchors)
        z_p = encode_fn(positives)
        z_n = encode_fn(negatives)
        cos_p = F.cosine_similarity(z_a, z_p, dim=-1).cpu().numpy()
        cos_n = F.cosine_similarity(z_a, z_n, dim=-1).cpu().numpy()
    scores = cos_p - cos_n
    mean = float(np.mean(scores))
    rng = float(np.max(scores) - np.min(scores))
    return {
        "name": family["name"],
        "n_rows": len(rows),
        "mean_score": mean,
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "range": rng,
        "non_degenerate_pass": rng > range_threshold,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encode", action="store_true",
                        help="If set, also encode triples and verify "
                             "non-degenerate score range per family.")
    parser.add_argument("--encoder", default="gte-base",
                        choices=list(ENCODERS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/stage0_5/diagnostics_validate.json")
    args = parser.parse_args()

    print("Task 0.5.4 — encoder calibration diagnostic validation")
    print("=" * 64)

    # ---- Structural checks ----
    print("\n[1/2] Structural checks (no encoder needed):")
    structural = [structural_check(fam) for fam in TEST_FAMILIES]
    structural_pass = all(r["pass"] for r in structural)
    for r in structural:
        if r["pass"]:
            print(f"  ✓ {r['name']:<12s}  rows={r['n_rows']}  unique={r['n_unique']}")
        else:
            print(f"  ✗ {r['name']:<12s}  FAIL: {r['reason']}")

    # ---- Encoder score check (optional) ----
    encoded: list[dict] = []
    encoded_pass = True
    if args.encode:
        if not structural_pass:
            print("\n[2/2] Skipping encoder check — structural FAIL above must be fixed first.")
        else:
            enc_cfg = ENCODERS[args.encoder]
            print(f"\n[2/2] Encoder check ({args.encoder}, dim={enc_cfg['dim']}):")
            print(f"  loading {enc_cfg['model']} ...")
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(enc_cfg["model"])
            mdl = AutoModel.from_pretrained(enc_cfg["model"]).to(args.device).eval()
            for p in mdl.parameters():
                p.requires_grad_(False)
            encode = make_encode_fn(mdl, tok, args.device)

            for fam in TEST_FAMILIES:
                r = encoded_score_check(fam, encode)
                encoded.append(r)
                tag = "✓" if r["non_degenerate_pass"] else "✗"
                print(f"  {tag} {r['name']:<12s}  mean_score={r['mean_score']:+.4f}  "
                      f"range={r['range']:.4f}  "
                      f"min={r['min_score']:+.4f}  max={r['max_score']:+.4f}")
            encoded_pass = all(r["non_degenerate_pass"] for r in encoded)
    else:
        print("\n[2/2] Skipped (no --encode flag). Run with --encode to verify "
              "scores. Structural-only acceptance still applies.")

    # ---- Acceptance ----
    overall = structural_pass and (encoded_pass if args.encode else True)
    print("\n" + "=" * 64)
    print("ACCEPTANCE CHECK (Task 0.5.4)")
    print("=" * 64)
    print(f"  Structural checks: {'PASS' if structural_pass else 'FAIL'}")
    if args.encode:
        print(f"  Encoder score range > 0.05 per family: "
              f"{'PASS' if encoded_pass else 'FAIL'}")
    print(f"\n→ Task 0.5.4: {'PASS' if overall else 'FAIL'}")

    payload = {
        "task": "0.5.4",
        "structural": structural,
        "encoded": encoded if args.encode else None,
        "encoder": args.encoder if args.encode else None,
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
