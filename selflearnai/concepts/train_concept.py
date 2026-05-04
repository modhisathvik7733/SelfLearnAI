"""Stage 2 — train ONE concept operator (plurality) end-to-end.

Inputs:
  • Frozen foundations (V-JEPA-2, GTE, CLIP).
  • Frozen Stage-1 adapters (loaded from checkpoint).
  • A small set of (singular_text, plural_text) pairs.
  • A small set of grounded image pairs (image_of_one_X, image_of_many_X).

Trains:
  • Forward plural operator (ConceptOperator).
  • Inverse plural operator (InverseConceptOperator).

Losses (locked, see plan section 2):
  L_text_forward  = MSE(forward(z_sing_text), z_plur_text)
  L_text_inverse  = MSE(inverse(z_plur_text), z_sing_text)
  L_xmodal_consistency  = 1 − cos(v_text_shift, v_visual_shift)
      where v_text_shift   = mean over text pairs   of (z_plur - z_sing)
            v_visual_shift = mean over image pairs  of (z_many - z_one)
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
    out_dir: str = "checkpoints/stage2_plurality"
    device: str = "cuda"
    seed: int = 0


# ---------------------------------------------------------------------------
# Encoded-pair caches (pre-encoded once, reused across epochs — fast)
# ---------------------------------------------------------------------------
@dataclass
class TextPair:
    singular: str
    plural: str


@dataclass
class ImagePair:
    one_path: Path
    many_path: Path
    noun: str        # for diagnostic logging


@dataclass
class EncodedPairs:
    """Pre-encoded shared-space embeddings for every training pair.
    Computed once at the start; then we train the operator over them."""
    text_sing: torch.Tensor    # (N_text, SHARED_DIM)
    text_plur: torch.Tensor    # (N_text, SHARED_DIM)
    img_one: torch.Tensor      # (N_img,  SHARED_DIM)
    img_many: torch.Tensor     # (N_img,  SHARED_DIM)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Stage2Trainer:
    """Trains plurality operator + inverse on the frozen Stage-1 latent space."""

    def __init__(
        self,
        cfg: Stage2Config,
        text_pairs: list[TextPair],
        image_pairs: list[ImagePair],
    ):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # --- Frozen everything (foundations + Stage-1 adapters) ---
        self.gte = FrozenGTE(cfg.gte_name, device=cfg.device)
        self.clip = FrozenCLIP(cfg.clip_name, device=cfg.device)
        self.vjepa = FrozenVJEPA2(cfg.vjepa_name, device=cfg.device)
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
        text_pairs: list[TextPair],
        image_pairs: list[ImagePair],
    ) -> EncodedPairs:
        """Map every (sing, plur) and (one, many) pair into the SHARED_DIM space."""

        # ---- Text side ----
        # Per the star topology, text goes through GTE → adapter_t.
        sing_texts = [p.singular for p in text_pairs]
        plur_texts = [p.plural for p in text_pairs]
        z_gte_s = self.gte.encode(sing_texts)
        z_gte_p = self.gte.encode(plur_texts)
        text_sing = self.bundle.adapter_t(z_gte_s)
        text_plur = self.bundle.adapter_t(z_gte_p)

        # ---- Image side ----
        one_imgs = [Image.open(p.one_path).convert("RGB") for p in image_pairs]
        many_imgs = [Image.open(p.many_path).convert("RGB") for p in image_pairs]
        z_vj_one = self.vjepa.encode_patches(one_imgs)
        z_vj_many = self.vjepa.encode_patches(many_imgs)
        img_one = self.bundle.adapter_v(z_vj_one)
        img_many = self.bundle.adapter_v(z_vj_many)

        return EncodedPairs(
            text_sing=text_sing,
            text_plur=text_plur,
            img_one=img_one,
            img_many=img_many,
        )

    # -----------------------------------------------------------------------
    # Cross-modal consistency: mandatory grounding signal.
    # -----------------------------------------------------------------------
    def _xmodal_consistency_loss(self) -> torch.Tensor:
        """Enforce: text plural-direction ≈ visual plural-direction.

        Without this, the operator may align in text space but be irrelevant
        to vision. Plan section 6a flags this as a true-grounding metric;
        making it a training loss enforces it during optimization, not just
        post-hoc.
        """
        v_text = (self.enc.text_plur - self.enc.text_sing).mean(dim=0)     # (D,)
        v_vis  = (self.enc.img_many - self.enc.img_one).mean(dim=0)        # (D,)
        # 1 − cosine: minimized when directions align.
        return 1.0 - F.cosine_similarity(
            v_text.unsqueeze(0), v_vis.unsqueeze(0)
        ).squeeze()

    # -----------------------------------------------------------------------
    # One training step
    # -----------------------------------------------------------------------
    def step(self, epoch: int) -> dict:
        cfg = self.cfg

        # Forward operator: predict plural-text-emb from singular-text-emb.
        pred_plur = self.fwd(self.enc.text_sing)
        L_fwd = F.mse_loss(pred_plur, self.enc.text_plur)

        # Inverse operator: predict singular-text-emb from plural-text-emb.
        pred_sing = self.inv(self.enc.text_plur)
        L_inv = F.mse_loss(pred_sing, self.enc.text_sing)

        # Cross-modal consistency: text plural-shift must align with visual
        # plural-shift in shared space.
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
            "L_xmodal": L_xmodal.item(),
        }

    # -----------------------------------------------------------------------
    # Top-level training loop
    # -----------------------------------------------------------------------
    def fit(self) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Stage 2 — plurality operator ===")
        print(f"  device: {cfg.device}   epochs: {cfg.epochs}")
        print(f"  text pairs: {len(self.text_pairs)}   image pairs: {len(self.image_pairs)}")
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
        print("\nStage 2 training complete.")

    def save(self, path: Path) -> None:
        torch.save({
            "fwd": self.fwd.state_dict(),
            "inv": self.inv.state_dict(),
            "config": self.cfg.__dict__,
        }, path)
