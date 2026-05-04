"""Stage 1 trainer — star-topology adapter alignment.

Topology (locked, see plan section 1):
    [CLIP] = passive anchor (only VICReg)
       ↑↑
        \\
   align→ \\         align→
           \\
       [GTE]            [V-JEPA-2]

Stage-1 losses (locked, see plan section 2):
    L = L_text_to_anchor (InfoNCE)
      + L_vision_to_anchor (InfoNCE)
      + L_vicreg_text + L_vicreg_vision + L_vicreg_anchor
    NO direct GTE↔V-JEPA loss. NO JEPA span loss in Stage 1.

Inputs:
    - frozen GTE, CLIP, V-JEPA-2 encoders (selflearnai.foundations).
    - StratifiedBatchBuilder yielding `Sample` objects (caption + image_id +
      category + count).
    - image_paths dict resolving image_id → path on disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from selflearnai import SHARED_DIM
from selflearnai.adapters import AdapterBundle
from selflearnai.foundations import FrozenCLIP, FrozenGTE, FrozenVJEPA2
from selflearnai.grounding import (
    EmbeddingMemoryBank,
    Sample,
    StratifiedBatchBuilder,
    load_image,
)
from selflearnai.losses import Projector, info_nce_loss, vicreg_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Stage1Config:
    # Data
    coco_root: str = ""                # path to COCO root
    coco_split: str = "val2017"
    max_samples: int = 30_000

    # Foundations
    gte_name: str = "thenlper/gte-base"
    clip_name: str = "openai/clip-vit-base-patch32"
    vjepa_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"

    # Training
    steps: int = 2000
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    temperature: float = 0.07
    hard_negatives_per_modality: int = 64
    memory_bank_size: int = 4096
    min_categories: int = 4
    min_count_levels: int = 3

    # Loss weights
    w_align_text: float = 1.0
    w_align_vision: float = 1.0
    w_vicreg_text: float = 1.0
    w_vicreg_vision: float = 1.0
    w_vicreg_anchor: float = 1.0       # only used if freeze_anchor=False
    vicreg_proj_dim: int = 768

    # Anchor policy. When True (default), adapter_c is frozen at random init
    # and never receives gradient. Star topology then becomes a clean
    # distillation target: text and vision adapters learn to map into a
    # FIXED random projection of CLIP space. Empirically: even VICReg
    # pressure on the anchor drifts it (anchor_cos: 1.00 → 0.66 in 300
    # steps), so freezing is the safer default.
    freeze_anchor: bool = True

    # Logging / checkpointing
    log_every: int = 50
    ckpt_every: int = 500
    out_dir: str = "checkpoints/stage1"
    device: str = "cuda"
    seed: int = 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Stage1Trainer:
    """Trains adapter_t and adapter_v to align with the (passive) adapter_c.

    Adapter_c gets only VICReg pressure — never alignment loss. This is what
    makes the topology a star, not a triangle.
    """

    def __init__(
        self,
        cfg: Stage1Config,
        samples: list[Sample],
        image_paths: dict[str, Path],
    ):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # --- Frozen foundations ---
        self.gte = FrozenGTE(cfg.gte_name, device=cfg.device)
        self.clip = FrozenCLIP(cfg.clip_name, device=cfg.device)
        self.vjepa = FrozenVJEPA2(cfg.vjepa_name, device=cfg.device)

        # --- Trainable adapters ---
        self.bundle = AdapterBundle(
            clip_dim=self.clip.native_dim,
            gte_dim=self.gte.native_dim,
            vjepa_dim=self.vjepa.native_dim,
            shared_dim=SHARED_DIM,
        ).to(cfg.device)

        # --- VICReg projection heads (one per branch) ---
        self.proj_text = Projector(SHARED_DIM, cfg.vicreg_proj_dim).to(cfg.device)
        self.proj_vision = Projector(SHARED_DIM, cfg.vicreg_proj_dim).to(cfg.device)
        self.proj_anchor = Projector(SHARED_DIM, cfg.vicreg_proj_dim).to(cfg.device)

        # --- Lock the anchor if configured (default: True) ---
        # Star topology requires adapter_c to be a stable target. Even VICReg
        # pressure was empirically enough to drift it (anchor_cos collapses
        # ~1.0 → 0.66 in a few hundred steps). Freezing makes the system a
        # distillation-into-fixed-projection: text and vision adapters learn
        # to align with adapter_c's frozen-at-init outputs.
        if cfg.freeze_anchor:
            for p in self.bundle.adapter_c.parameters():
                p.requires_grad_(False)
            self.bundle.adapter_c.eval()

        # --- Optimizer (only trainable modules) ---
        trainable = []
        if not cfg.freeze_anchor:
            trainable += list(self.bundle.adapter_c.parameters())
            trainable += list(self.proj_anchor.parameters())
        trainable += list(self.bundle.adapter_t.parameters())
        trainable += list(self.bundle.adapter_v.parameters())
        trainable += list(self.proj_text.parameters())
        trainable += list(self.proj_vision.parameters())
        self.opt = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        # --- Memory banks for hard-negative mining ---
        self.bank_text = EmbeddingMemoryBank(SHARED_DIM, cfg.memory_bank_size, cfg.device)
        self.bank_vision = EmbeddingMemoryBank(SHARED_DIM, cfg.memory_bank_size, cfg.device)

        # --- Data ---
        self.samples = samples
        self.image_paths = image_paths
        self.batch_iter = StratifiedBatchBuilder(
            samples,
            batch_size=cfg.batch_size,
            min_categories=cfg.min_categories,
            min_count_levels=cfg.min_count_levels,
            seed=cfg.seed,
        )

        # --- Snapshot of anchor at step 0 (for drift monitoring) ---
        self._anchor_init_state = {
            k: v.detach().clone() for k, v in self.bundle.adapter_c.state_dict().items()
        }

    # -----------------------------------------------------------------------
    # One training step
    # -----------------------------------------------------------------------
    def step(self, step_idx: int) -> dict:
        cfg = self.cfg
        batch: list[Sample] = next(self.batch_iter)

        texts = [s.text for s in batch]
        image_ids = [s.image_id for s in batch]

        # Load images (caller's responsibility to pre-cache to RAM/disk).
        images = [load_image(self.image_paths[i]) for i in image_ids]

        # --- Foundation forward (no grad) ---
        z_gte = self.gte.encode(texts)                                  # (B, 768)
        z_clip_text = self.clip.encode_text(texts)                      # (B, 512)
        z_clip_vision = self.clip.encode_image(images)                  # (B, 512)
        z_vjepa_patches = self.vjepa.encode_patches(images)             # (B, N, 1024)

        # --- Adapter forward (gradients flow) ---
        # adapter_c (anchor) — passive: outputs only feed VICReg, never alignment.
        a_text = self.bundle.adapter_c(z_clip_text)                     # (B, SHARED_DIM)
        a_vision = self.bundle.adapter_c(z_clip_vision)                 # (B, SHARED_DIM)

        # adapter_t (text) and adapter_v (vision) — these align to anchor.
        t = self.bundle.adapter_t(z_gte)                                # (B, SHARED_DIM)
        v = self.bundle.adapter_v(z_vjepa_patches)                      # (B, SHARED_DIM)

        # --- Hard negatives from memory banks ---
        with torch.no_grad():
            neg_text = self.bank_text.hard_negatives(
                t, exclude_ids=image_ids, k=cfg.hard_negatives_per_modality,
            )
            neg_vision = self.bank_vision.hard_negatives(
                v, exclude_ids=image_ids, k=cfg.hard_negatives_per_modality,
            )

        # --- Alignment losses (InfoNCE with stop-grad on anchor side) ---
        # text → CLIP_text(anchor side, stop-grad to keep anchor passive)
        L_text = info_nce_loss(
            t, a_text.detach(),
            temperature=cfg.temperature,
            extra_negatives=neg_text,
        )
        # vision → CLIP_vision(anchor side, stop-grad)
        L_vision = info_nce_loss(
            v, a_vision.detach(),
            temperature=cfg.temperature,
            extra_negatives=neg_vision,
        )

        # --- VICReg per modality (anti-collapse) ---
        L_vic_text = vicreg_loss(self.proj_text(t))
        L_vic_vision = vicreg_loss(self.proj_vision(v))

        loss = (
            cfg.w_align_text * L_text
            + cfg.w_align_vision * L_vision
            + cfg.w_vicreg_text * L_vic_text
            + cfg.w_vicreg_vision * L_vic_vision
        )

        # Anchor VICReg: only included when anchor is unfrozen. With the
        # default freeze_anchor=True path, adapter_c receives no gradient
        # and the anchor projector is unused.
        if not cfg.freeze_anchor:
            a_combined = torch.cat([a_text, a_vision], dim=0)
            L_vic_anchor = vicreg_loss(self.proj_anchor(a_combined))
            loss = loss + cfg.w_vicreg_anchor * L_vic_anchor
        else:
            L_vic_anchor = torch.tensor(0.0, device=t.device)

        self.opt.zero_grad()
        loss.backward()
        # Grad clip on actually-trainable params only.
        clip_params = (
            list(self.bundle.adapter_t.parameters())
            + list(self.bundle.adapter_v.parameters())
            + list(self.proj_text.parameters())
            + list(self.proj_vision.parameters())
        )
        if not cfg.freeze_anchor:
            clip_params += list(self.bundle.adapter_c.parameters())
            clip_params += list(self.proj_anchor.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
        self.opt.step()

        # --- Update memory banks ---
        self.bank_text.push(t.detach(), image_ids)
        self.bank_vision.push(v.detach(), image_ids)

        return {
            "loss": loss.item(),
            "L_text": L_text.item(),
            "L_vision": L_vision.item(),
            "L_vic_text": L_vic_text.item(),
            "L_vic_vision": L_vic_vision.item(),
            "L_vic_anchor": L_vic_anchor.item(),
        }

    # -----------------------------------------------------------------------
    # Anchor drift check — locked metric from plan section 6c
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def anchor_drift_cosine(self) -> float:
        """cos(adapter_c at epoch_0, adapter_c at now), averaged over a sample
        of inputs. Plan threshold: > 0.85. Below = anchor moved too far."""
        # Restore initial state into a copy of adapter_c, run on a fixed
        # probe set, compare to current adapter_c outputs on the same probes.
        from copy import deepcopy
        probe = self._anchor_init_state
        current = deepcopy(self.bundle.adapter_c.state_dict())
        # Build a fresh adapter with init weights:
        from selflearnai.adapters.adapter import Adapter
        a_init = Adapter(self.clip.native_dim, SHARED_DIM).to(self.cfg.device)
        a_init.load_state_dict(probe)
        a_init.eval()

        # Probe over 32 random captions from the dataset.
        import random as _r
        rng = _r.Random(0)
        probe_samples = rng.sample(self.samples, k=min(32, len(self.samples)))
        texts = [s.text for s in probe_samples]
        z_clip_text = self.clip.encode_text(texts)
        a_then = a_init(z_clip_text)
        a_now = self.bundle.adapter_c(z_clip_text)
        cos = F.cosine_similarity(a_then, a_now, dim=-1).mean().item()
        return cos

    # -----------------------------------------------------------------------
    # Top-level training loop
    # -----------------------------------------------------------------------
    def fit(self) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Stage 1 — star-topology alignment ===")
        print(f"  device: {cfg.device}   shared_dim: {SHARED_DIM}")
        print(f"  steps: {cfg.steps}   batch: {cfg.batch_size}")
        print(f"  freeze_anchor: {cfg.freeze_anchor}")
        print(f"  samples: {len(self.samples)}   categories: "
              f"{len({s.category for s in self.samples})}   "
              f"count levels: {len({s.count for s in self.samples})}")
        print()

        for step in range(cfg.steps):
            metrics = self.step(step)
            if step % cfg.log_every == 0:
                if cfg.freeze_anchor:
                    # Anchor is frozen by construction; drift is trivially 1.0.
                    drift = 1.0
                    drift_label = "FROZEN"
                else:
                    drift = self.anchor_drift_cosine() if step > 0 else 1.0
                    drift_label = f"{drift:+.3f}"
                vic_anchor = metrics['L_vic_anchor']
                print(
                    f"[step {step:5d}]  loss={metrics['loss']:.4f}  "
                    f"L_text={metrics['L_text']:.3f}  "
                    f"L_vis={metrics['L_vision']:.3f}  "
                    f"L_vic={metrics['L_vic_text']:.2f}/"
                    f"{metrics['L_vic_vision']:.2f}/{vic_anchor:.2f}  "
                    f"anchor={drift_label}"
                )
                if not cfg.freeze_anchor and drift < 0.85 and step > 200:
                    print("  ⚠ anchor drift below 0.85 — star topology compromised. "
                          "Set freeze_anchor: True in the config and restart.")
            if step > 0 and step % cfg.ckpt_every == 0:
                self.save(out_dir / f"step_{step}.pt")

        self.save(out_dir / "final.pt")
        print("\nStage 1 training complete.")

    def save(self, path: Path) -> None:
        torch.save({
            "adapter_c": self.bundle.adapter_c.state_dict(),
            "adapter_t": self.bundle.adapter_t.state_dict(),
            "adapter_v": self.bundle.adapter_v.state_dict(),
            "proj_text": self.proj_text.state_dict(),
            "proj_vision": self.proj_vision.state_dict(),
            "proj_anchor": self.proj_anchor.state_dict(),
            "config": self.cfg.__dict__,
        }, path)

    @classmethod
    def load_adapters(cls, path: Path | str, cfg: Stage1Config) -> AdapterBundle:
        """Load adapters from a checkpoint. Used by Stage 2 + metric scripts."""
        import torch
        from selflearnai.adapters import AdapterBundle
        from selflearnai.foundations import (
            CLIP_NATIVE_DIM, GTE_NATIVE_DIM, VJEPA2_NATIVE_DIM,
        )
        bundle = AdapterBundle(
            clip_dim=CLIP_NATIVE_DIM,
            gte_dim=GTE_NATIVE_DIM,
            vjepa_dim=VJEPA2_NATIVE_DIM,
            shared_dim=SHARED_DIM,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        bundle.adapter_c.load_state_dict(ckpt["adapter_c"])
        bundle.adapter_t.load_state_dict(ckpt["adapter_t"])
        bundle.adapter_v.load_state_dict(ckpt["adapter_v"])
        bundle.to(cfg.device)
        bundle.eval()
        return bundle
