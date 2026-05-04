"""COCO data loader for Stage 1 alignment.

We use COCO 2017 because it gives us, for free, the three things batch
construction needs (plan section 1.5):
  • image-caption pairs (alignment supervision)
  • category labels (cross-category coverage)
  • per-image instance counts per category (count variation)

Requires `pycocotools` — install on the training machine:
    pip install pycocotools

Expected directory layout:
    coco_root/
      images/
        train2017/<image_id>.jpg
        val2017/<image_id>.jpg
      annotations/
        instances_train2017.json
        instances_val2017.json
        captions_train2017.json
        captions_val2017.json
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

from .batching import Sample


# Coarse semantic groups for stratified batching. Maps COCO supercategories
# (which are too few — only ~12) into broader groups. Adjust as needed.
_SUPERCATEGORY_TO_GROUP = {
    "person":          "person",
    "vehicle":         "vehicle",
    "outdoor":         "outdoor_object",
    "animal":          "animal",
    "accessory":       "accessory",
    "sports":          "object",
    "kitchen":         "container",
    "food":            "food",
    "furniture":       "furniture",
    "electronic":      "object",
    "appliance":       "appliance",
    "indoor":          "indoor_object",
}


def _map_category(supercategory: str) -> str:
    return _SUPERCATEGORY_TO_GROUP.get(supercategory, "other")


def _bucket_count(n: int) -> int:
    """Bucket raw counts into stratification levels.

    Raw counts are heavy-tailed (many images have 1, few have 30+). We
    bucket into discrete levels so the StratifiedBatchBuilder sees a
    manageable count vocabulary.
    """
    if n <= 1:    return 1
    if n == 2:    return 2
    if n <= 4:    return 3       # 'few'
    if n <= 9:    return 5       # 'several'
    return 10                    # 'many'


def load_coco_samples(
    coco_root: str | Path,
    split: str = "val2017",
    max_samples: int | None = None,
) -> tuple[list[Sample], dict[str, Path]]:
    """Build a list of `Sample` objects from a COCO split.

    Returns:
        samples : list[Sample]              — for the StratifiedBatchBuilder.
        image_paths : dict[image_id → Path] — for the loader to open lazily.
    """
    try:
        from pycocotools.coco import COCO
    except ImportError as e:
        raise ImportError(
            "pycocotools is required for COCO loading. "
            "Install with: pip install pycocotools"
        ) from e

    coco_root = Path(coco_root)
    inst_path = coco_root / "annotations" / f"instances_{split}.json"
    cap_path = coco_root / "annotations" / f"captions_{split}.json"
    img_dir = coco_root / "images" / split

    coco_inst = COCO(str(inst_path))
    coco_caps = COCO(str(cap_path))

    # category_id → supercategory string
    cat_id_to_super = {c["id"]: c["supercategory"] for c in coco_inst.loadCats(coco_inst.getCatIds())}

    # Build per-image: dominant category (most-instances), total count.
    img_ids = coco_inst.getImgIds()
    if max_samples is not None:
        img_ids = img_ids[: max_samples * 2]   # captions multiply

    samples: list[Sample] = []
    image_paths: dict[str, Path] = {}

    for img_id in img_ids:
        ann_ids = coco_inst.getAnnIds(imgIds=img_id)
        anns = coco_inst.loadAnns(ann_ids)
        if not anns:
            continue
        # Dominant category by instance count.
        cat_counter = Counter(a["category_id"] for a in anns)
        top_cat_id, top_cat_count = cat_counter.most_common(1)[0]
        category = _map_category(cat_id_to_super.get(top_cat_id, "other"))
        count = _bucket_count(top_cat_count)

        # Pull all captions for this image.
        cap_ids = coco_caps.getAnnIds(imgIds=img_id)
        captions = [c["caption"] for c in coco_caps.loadAnns(cap_ids)]
        if not captions:
            continue

        img_meta = coco_inst.loadImgs(img_id)[0]
        path = img_dir / img_meta["file_name"]
        image_paths[str(img_id)] = path

        # One Sample per (caption, image). Most COCO images have 5 captions —
        # this 5x's our effective dataset size at no extra image cost.
        for cap in captions:
            samples.append(Sample(
                text=cap.strip(),
                image_id=str(img_id),
                category=category,
                count=count,
            ))

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples, image_paths


def load_image(path: Path) -> Image.Image:
    """Open a COCO image as RGB. Caller is responsible for caching / batching."""
    return Image.open(path).convert("RGB")
