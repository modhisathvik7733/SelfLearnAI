"""Stage 2 entrypoint — train the plurality concept operator.

Usage:
    python3 scripts/stage2_train.py --config configs/stage2_plurality.yaml

The training and held-out data lives in `data/plurality/`:

    data/plurality/text_pairs_train.tsv     50 sing\tplur lines
    data/plurality/text_pairs_held_out.tsv  20 sing\tplur lines
    data/plurality/text_pairs_other_cat.tsv 20 sing\tplur lines (different category, for cross-cat test)
    data/plurality/image_pairs.tsv          path_one\tpath_many\tnoun
    data/plurality/candidate_pool.txt       1 word/line — must include held-out plurals + distractors

A starter set is provided in data/plurality/. Replace with your own as the
project grows.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from selflearnai.concepts.train_concept import (
    Stage2Config,
    Stage2Trainer,
    TextPair,
    ImagePair,
)


def _read_tsv_pairs(path: Path) -> list[tuple[str, str]]:
    with open(path) as f:
        return [tuple(line.strip().split("\t"))[:2] for line in f if line.strip()]


def _read_image_pairs(path: Path) -> list[ImagePair]:
    out: list[ImagePair] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            one, many, noun = parts[:3]
            out.append(ImagePair(Path(one), Path(many), noun))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--data-dir", default="data/plurality",
        help="dir containing text_pairs_train.tsv, image_pairs.tsv, etc.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Stage2Config(**cfg_dict)

    data_dir = Path(args.data_dir)
    train_tsv  = _read_tsv_pairs(data_dir / "text_pairs_train.tsv")
    text_pairs = [TextPair(s, p) for s, p in train_tsv]
    image_pairs = _read_image_pairs(data_dir / "image_pairs.tsv")

    print(f"Loaded {len(text_pairs)} text pairs and {len(image_pairs)} image pairs.")

    trainer = Stage2Trainer(cfg, text_pairs, image_pairs)
    trainer.fit()


if __name__ == "__main__":
    main()
