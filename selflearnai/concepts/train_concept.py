"""Stage 2 — train ONE concept operator end-to-end.

Concept-agnostic: works for plurality (cat→cats), past tense (walk→walked),
negation, comparatives, etc. The trainer doesn't know the concept's name —
it just learns a generic (source → target) mapping in shared-space.

Inputs:
  • Frozen foundations (V-JEPA-2, GTE, CLIP).
  • Frozen Stage-1 adapters (loaded from checkpoint).
  • A small set of (source_text, target_text) pairs.
  • Optional: grounded image pairs (image_of_source_state, image_of_target_state)
    for concepts where vision can show the transformation. Cross-modal
    consistency loss is skipped when no image pairs are provided.

Trains:
  • Forward concept operator (ConceptOperator).
  • Inverse concept operator (InverseConceptOperator).

Losses:
  L_forward = MSE(forward(z_source_text), z_target_text)
  L_inverse = MSE(inverse(z_target_text), z_source_text)
  L_xmodal  = 1 − cos(v_text_shift, v_visual_shift)              [if image pairs given]
      where v_text_shift   = mean over text pairs   of (z_target - z_source)
            v_visual_shift = mean over image pairs  of (z_visual_target - z_visual_source)
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
from selflearnai.concepts import ConceptOperator, InverseConceptOperator
from selflearnai.foundations import FrozenCLIP, FrozenGTE, FrozenVJEPA2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Stage2Config:
    # Stage-1 checkpoint to load adapters from.
    stage1_ckpt: str = "checkpoints/stage1/final.pt"

    # Foundations
    gte_name: str = "thenlper/gte-base"
    clip_name: str = "openai/clip-vit-base-patch32"
    vjepa_name: str = "facebook/vjepa2-vitl-fpc16-256-ssv2"

    # Training
    epochs: int = 1500
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Loss weights
    w_forward: float = 1.0
    w_inverse: float = 1.0
    w_xmodal: float = 1.0          # cross-modal consistency

    # Logging / output
    log_every: int = 50
    concept_name: str = "concept"   # for log banners only
    out_dir: str = "checkpoints/stage2_concept"
    device: str = "cuda"
    seed: int = 0


# ---------------------------------------------------------------------------
# Encoded-pair caches (pre-encoded once, reused across epochs — fast)
# ---------------------------------------------------------------------------
@dataclass
class ConceptTextPair:
    """One (source, target) text pair. Naming is concept-agnostic.
    For plurality: source='cat', target='cats'.
    For past tense: source='walk', target='walked'."""
    source: str
    target: str


@dataclass
class ConceptImagePair:
    """One (source-state-image, target-state-image) pair.
    For plurality: source=image of 1 X, target=image of many Xs.
    For past tense (if grounded): source=action-in-progress, target=after-action.
    label is just for diagnostic logging."""
    source_path: Path
    target_path: Path
    label: str


# Backward-compat aliases (existing scripts may still import these names).
TextPair = ConceptTextPair
ImagePair = ConceptImagePair


@dataclass
class EncodedPairs:
    """Pre-encoded shared-space embeddings for every training pair.
    Computed once at the start; then we train the operator over them.
    Image fields are None when no image pairs were provided."""
    text_source: torch.Tensor                  # (N_text, SHARED_DIM)
    text_target: torch.Tensor                  # (N_text, SHARED_DIM)
    img_source: torch.Tensor | None = None     # (N_img,  SHARED_DIM)
    img_target: torch.Tensor | None = None     # (N_img,  SHARED_DIM)

    @property
    def has_images(self) -> bool:
        return self.img_source is not None and self.img_source.numel() > 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Stage2Trainer:
    """Trains a single concept operator + inverse on the frozen Stage-1 latent
    space. Concept-agnostic: works for plurality, past tense, etc."""

    def __init__(
        self,
        cfg: Stage2Config,
        text_pairs: list[ConceptTextPair],
        image_pairs: list[ConceptImagePair] | None = None,
    ):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        if image_pairs is None:
            image_pairs = []

        # --- Frozen everything (foundations + Stage-1 adapters) ---
        # Vision foundation only loaded if we actually have image pairs.
        self.gte = FrozenGTE(cfg.gte_name, device=cfg.device)
        self.clip = FrozenCLIP(cfg.clip_name, device=cfg.device)
        if image_pairs:
            self.vjepa = FrozenVJEPA2(cfg.vjepa_name, device=cfg.device)
        else:
            self.vjepa = None
        from selflearnai.adapters.train_adapters import Stage1Trainer
        self.bundle = Stage1Trainer.load_adapters(cfg.stage1_ckpt, cfg)
        for p in self.bundle.parameters():
            p.requires_grad_(False)
        self.bundle.eval()

        # --- Trainable: forward + inverse operators ---
        self.fwd = ConceptOperator(SHARED_DIM).to(cfg.device)
        self.inv = InverseConceptOperator(SHARED_DIM).to(cfg.device)
        self.opt = torch.optim.AdamW(
            list(self.fwd.parameters()) + list(self.inv.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        # --- Pre-encode all training pairs ---
        self.enc = self._encode_pairs(text_pairs, image_pairs)
        self.text_pairs = text_pairs
        self.image_pairs = image_pairs

    @torch.no_grad()
    def _encode_pairs(
        self,
        text_pairs: list[ConceptTextPair],
        image_pairs: list[ConceptImagePair],
    ) -> EncodedPairs:
        """Map every (source, target) text pair (and optional image pair) into
        the SHARED_DIM space. Image branch is skipped if no image pairs."""

        # ---- Text side (always run) ----
        source_texts = [p.source for p in text_pairs]
        target_texts = [p.target for p in text_pairs]
        z_gte_s = self.gte.encode(source_texts)
        z_gte_t = self.gte.encode(target_texts)
        text_source = self.bundle.adapter_t(z_gte_s)
        text_target = self.bundle.adapter_t(z_gte_t)

        # ---- Image side (only if we have pairs) ----
        if image_pairs:
            src_imgs = [Image.open(p.source_path).convert("RGB") for p in image_pairs]
            tgt_imgs = [Image.open(p.target_path).convert("RGB") for p in image_pairs]
            z_vj_s = self.vjepa.encode_patches(src_imgs)
            z_vj_t = self.vjepa.encode_patches(tgt_imgs)
            img_source = self.bundle.adapter_v(z_vj_s)
            img_target = self.bundle.adapter_v(z_vj_t)
        else:
            img_source = None
            img_target = None

        return EncodedPairs(
            text_source=text_source,
            text_target=text_target,
            img_source=img_source,
            img_target=img_target,
        )

    # -----------------------------------------------------------------------
    # Cross-modal consistency: mandatory grounding signal IF images provided.
    # -----------------------------------------------------------------------
    def _xmodal_consistency_loss(self) -> torch.Tensor:
        """Enforce: text concept-direction ≈ visual concept-direction.

        For plurality with grounded images, this anchors the operator in
        perception. For text-only concepts (e.g., past tense in this repo),
        we skip this loss and rely entirely on text-side supervision.
        """
        if not self.enc.has_images:
            return torch.tensor(0.0, device=self.enc.text_source.device)
        v_text = (self.enc.text_target - self.enc.text_source).mean(dim=0)   # (D,)
        v_vis  = (self.enc.img_target - self.enc.img_source).mean(dim=0)     # (D,)
        return 1.0 - F.cosine_similarity(
            v_text.unsqueeze(0), v_vis.unsqueeze(0)
        ).squeeze()

    # -----------------------------------------------------------------------
    # One training step
    # -----------------------------------------------------------------------
    def step(self, epoch: int) -> dict:
        cfg = self.cfg

        # Forward operator: predict target-emb from source-emb.
        pred_target = self.fwd(self.enc.text_source)
        L_fwd = F.mse_loss(pred_target, self.enc.text_target)

        # Inverse operator: predict source-emb from target-emb.
        pred_source = self.inv(self.enc.text_target)
        L_inv = F.mse_loss(pred_source, self.enc.text_source)

        # Cross-modal consistency: only fires if we have image pairs.
        L_xmodal = self._xmodal_consistency_loss()

        loss = (
            cfg.w_forward * L_fwd
            + cfg.w_inverse * L_inv
            + cfg.w_xmodal * L_xmodal
        )

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "loss": loss.item(),
            "L_fwd": L_fwd.item(),
            "L_inv": L_inv.item(),
            "L_xmodal": L_xmodal.item() if isinstance(L_xmodal, torch.Tensor) else float(L_xmodal),
        }

    # -----------------------------------------------------------------------
    # Top-level training loop
    # -----------------------------------------------------------------------
    def fit(self) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Stage 2 — '{cfg.concept_name}' operator ===")
        print(f"  device: {cfg.device}   epochs: {cfg.epochs}")
        print(f"  text pairs: {len(self.text_pairs)}   image pairs: {len(self.image_pairs)}")
        if not self.enc.has_images:
            print(f"  (no image pairs → text-only training, L_xmodal disabled)")
        print()

        for epoch in range(cfg.epochs):
            metrics = self.step(epoch)
            if epoch % cfg.log_every == 0:
                print(
                    f"[epoch {epoch:5d}]  loss={metrics['loss']:.4f}  "
                    f"L_fwd={metrics['L_fwd']:.4f}  "
                    f"L_inv={metrics['L_inv']:.4f}  "
                    f"L_xmodal={metrics['L_xmodal']:.4f}"
                )

        self.save(out_dir / "final.pt")
        print(f"\nStage 2 training complete ('{cfg.concept_name}' operator saved).")

    def save(self, path: Path) -> None:
        torch.save({
            "fwd": self.fwd.state_dict(),
            "inv": self.inv.state_dict(),
            "config": self.cfg.__dict__,
        }, path)
