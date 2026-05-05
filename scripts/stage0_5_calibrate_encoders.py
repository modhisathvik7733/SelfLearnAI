"""Task 0.5.5 — encoder calibration runner across all configured encoders.

Produces the (encoder × task-family) score table that drives:
  - Per-family encoder selection (default to highest-scoring encoder
    for that domain).
  - Adapter-watch flags (reactive policy: a family below the watch
    threshold flags for adapter consideration but does NOT auto-train).

Acceptance gate (Task 0.5.5):
  Score table generates cleanly across all (encoder × family) cells.
  All encoder mean scores are positive on every family (no encoder is
  fundamentally broken on any diagnostic). The watch threshold is 0.05;
  families flagged for watch are reported but do not fail the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# torch loaded lazily in build_encode_fn so this script can do --help
# / --dry-run without needing a GPU stack.

from selflearnai.encoders import EncoderCalibrator, load_default_suite


ENCODERS = {
    "gte-base":    {"model": "thenlper/gte-base",    "dim":  768},
    "e5-large-v2": {"model": "intfloat/e5-large-v2", "dim": 1024},
}


def build_encode_fn(model_name: str, device: str, max_length: int = 128):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in mdl.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def encode(words):
        inputs = tok(
            words, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = mdl(**inputs)
        last_hidden = out.last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.float()

    return encode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()),
                        help="Encoders to score. Default: all configured.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--watch-threshold", type=float, default=0.05,
                        help="Family mean-score below this → adapter-watch flag.")
    parser.add_argument("--out", default="results/stage0_5/encoder_calibration.json")
    args = parser.parse_args()

    print("Task 0.5.5 — encoder calibration runner")
    print("=" * 70)
    print(f"Encoders: {args.encoders}")
    print(f"Device:   {args.device}")
    print(f"Watch threshold: {args.watch_threshold}")

    # ---- Build encode_fn per encoder ----
    encode_fns: dict[str, callable] = {}
    for name in args.encoders:
        if name not in ENCODERS:
            raise SystemExit(f"Unknown encoder {name!r}. Configured: {list(ENCODERS.keys())}")
        cfg = ENCODERS[name]
        print(f"\nLoading {name} ({cfg['model']}) ...")
        encode_fns[name] = build_encode_fn(cfg["model"], args.device)

    # ---- Score ----
    suite = load_default_suite()
    print(f"\nDiagnostic suite: {[f.name for f in suite]} "
          f"({sum(len(f.triples) for f in suite)} triples total)")
    cal = EncoderCalibrator(suite, watch_threshold=args.watch_threshold)
    result = cal.score(encode_fns)

    # ---- Print table ----
    print("\n" + "=" * 90)
    print("ENCODER × FAMILY SCORE TABLE  (mean = cos(anchor, positive) − cos(anchor, negative))")
    print("=" * 90)
    families = result["families"]
    encoders = result["encoders"]
    header = f"  {'family':<14s}  " + "  ".join(f"{e:>14s}" for e in encoders) + "       best"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fam in families:
        cells = []
        for enc in encoders:
            s = result["table"][enc][fam]["mean_score"]
            cells.append(f"{s:>+14.4f}")
        rec = result["recommendations"][fam]
        marker = " ⚠ adapter-watch" if rec["adapter_watch"] else ""
        print(f"  {fam:<14s}  " + "  ".join(cells) +
              f"   {rec['best_encoder']:<14s}{marker}")

    # ---- Per-family detail ----
    print("\nPer-family detail (positive mean / negative mean / fraction rows positive):")
    for fam in families:
        print(f"\n  {fam}:")
        for enc in encoders:
            s = result["table"][enc][fam]
            print(f"    {enc:<14s}  cos(a,pos)={s['cos_positive_mean']:.4f}  "
                  f"cos(a,neg)={s['cos_negative_mean']:.4f}  "
                  f"frac_rows_positive={s['frac_rows_positive']:.2f}  "
                  f"std={s['std_score']:.4f}")

    # ---- Acceptance gate ----
    table = result["table"]
    all_positive = all(
        table[enc][fam]["mean_score"] > 0
        for enc in encoders for fam in families
    )
    all_complete = all(
        "mean_score" in table[enc][fam]
        for enc in encoders for fam in families
    )
    overall = all_positive and all_complete

    print("\n" + "=" * 70)
    print("ACCEPTANCE CHECK (Task 0.5.5)")
    print("=" * 70)
    print(f"  Score table complete (all cells filled): "
          f"{'PASS' if all_complete else 'FAIL'}")
    print(f"  All encoders give mean_score > 0 on every family: "
          f"{'PASS' if all_positive else 'FAIL'}")
    n_watch = sum(1 for fam in families if result["recommendations"][fam]["adapter_watch"])
    print(f"\n  Adapter-watch families ({n_watch}): "
          + ", ".join(f for f in families if result["recommendations"][f]["adapter_watch"])
          + ("  (informational, not gated)" if n_watch else ""))
    print(f"\n→ Task 0.5.5: {'PASS' if overall else 'FAIL'}")

    payload = {
        "task": "0.5.5",
        "encoders": encoders,
        "families": families,
        "watch_threshold": args.watch_threshold,
        "table": result["table"],
        "recommendations": result["recommendations"],
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
