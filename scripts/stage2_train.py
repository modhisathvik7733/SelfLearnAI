"""Stage 2 entrypoint — train ONE concept operator end-to-end.

Concept-agnostic: works for plurality, past tense, negation, etc.

Usage:
    # Plurality (with image pairs for visual grounding):
    python3 scripts/stage2_train.py \\
        --config configs/stage2_plurality.yaml \\
        --data-dir data/plurality

    # Past tense (text only — no easy visual grounding):
    python3 scripts/stage2_train.py \\
        --config configs/stage2_past_tense.yaml \\
        --data-dir data/past_tense \\
        --no-images

The data dir must contain:

    text_pairs_train.tsv      source<TAB>target
    text_pairs_held_out.tsv   same format, for evaluation
    candidate_pool.txt        target words + distractors, one per line
    image_pairs.tsv           src_path<TAB>tgt_path<TAB>label   (optional)

For text-only concepts, pass --no-images to skip image-pair loading.
The cross-modal consistency loss is automatically disabled when no image
pairs are present.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from selflearnai.concepts.train_concept import (
    ConceptImagePair,
    ConceptTextPair,
    Stage2Config,
    Stage2Trainer,
)


def _read_tsv_pairs(path: Path) -> list[tuple[str, str]]:
    """Read (source, target) tab-separated pairs. Skips blanks + # comments."""
    out: list[tuple[str, str]] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            out.append((parts[0], parts[1]))
    return out


def _read_image_pairs(path: Path) -> list[ConceptImagePair]:
    """Read (src_path, tgt_path, label) triples, one per line.
    Skips blanks + # comments. Validates each path actually exists.
    """
    out: list[ConceptImagePair] = []
    bad: list[tuple[int, str]] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                bad.append((lineno, f"need 3 tab-separated fields, got {len(parts)}"))
                continue
            src_p, tgt_p, label = Path(parts[0]), Path(parts[1]), parts[2]
            if not src_p.exists():
                bad.append((lineno, f"missing file: {src_p}"))
                continue
            if not tgt_p.exists():
                bad.append((lineno, f"missing file: {tgt_p}"))
                continue
            out.append(ConceptImagePair(src_p, tgt_p, label))
    if bad:
        for lineno, msg in bad[:5]:
            print(f"  [warning] {path}:{lineno}  {msg}")
        if len(bad) > 5:
            print(f"  ... and {len(bad) - 5} more")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--data-dir", required=True,
        help="dir containing text_pairs_train.tsv (and optionally image_pairs.tsv)",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="text-only training; skip image_pairs.tsv even if present",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Stage2Config(**cfg_dict)

    data_dir = Path(args.data_dir)
    train_tsv = _read_tsv_pairs(data_dir / "text_pairs_train.tsv")
    text_pairs = [ConceptTextPair(s, t) for s, t in train_tsv]

    image_pairs: list[ConceptImagePair] = []
    if not args.no_images:
        img_path = data_dir / "image_pairs.tsv"
        if img_path.exists():
            image_pairs = _read_image_pairs(img_path)

    print(f"Loaded {len(text_pairs)} text pairs and {len(image_pairs)} image pairs"
          f" for concept '{cfg.concept_name}'.")

    if not text_pairs:
        sys.exit(f"\nERROR: no valid text pairs in {data_dir / 'text_pairs_train.tsv'}.")

    trainer = Stage2Trainer(cfg, text_pairs, image_pairs)
    trainer.fit()


if __name__ == "__main__":
    main()
