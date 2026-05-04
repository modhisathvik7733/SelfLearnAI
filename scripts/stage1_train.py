"""Stage 1 entrypoint — train adapters with star-topology alignment.

Usage:
    python3 scripts/stage1_train.py --config configs/stage1_alignment.yaml

Prerequisites:
    pip install pycocotools
    Download COCO 2017 (val + annotations) into the directory pointed to by
    `coco_root` in the config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from selflearnai.adapters.train_adapters import Stage1Config, Stage1Trainer
from selflearnai.grounding import load_coco_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = Stage1Config(**cfg_dict)

    print(f"Loading COCO samples from {cfg.coco_root} (split={cfg.coco_split}) ...")
    samples, image_paths = load_coco_samples(
        coco_root=cfg.coco_root,
        split=cfg.coco_split,
        max_samples=cfg.max_samples,
    )
    print(f"  loaded {len(samples)} (text, image) samples covering "
          f"{len({s.category for s in samples})} categories.")

    trainer = Stage1Trainer(cfg, samples, image_paths)
    trainer.fit()


if __name__ == "__main__":
    main()
