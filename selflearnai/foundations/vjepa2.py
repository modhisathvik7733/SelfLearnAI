"""V-JEPA-2 visual encoder wrapper. Frozen, never trained.

V-JEPA-2 (Joint Embedding Predictive Architecture, video edition) is the
purest available embodiment of the JEPA philosophy in vision: trained by
predicting masked-region embeddings from visible regions, all in latent
space, NO token / pixel reconstruction. Output dim: 1024 (per-patch).

V-JEPA-2 is video-pretrained. We use it on still images by treating each
image as a single-frame "video clip" — the model accepts T=1 inputs.

Available checkpoints (pick one based on hardware):
    facebook/vjepa2-vitl-fpc16-256-ssv2     ViT-L,  256px, ~300M params
    facebook/vjepa2-vith-fpc16-384-ssv2     ViT-H,  384px, ~600M params
    facebook/vjepa2-vitg-fpc16-384-ssv2     ViT-G,  384px, ~1.2B params

For MVP: default to ViT-L (smallest). The plan's compute budget assumes ViT-L.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image


VJEPA2_DEFAULT = "facebook/vjepa2-vitl-fpc16-256-ssv2"
VJEPA2_NATIVE_DIM = 1024


class FrozenVJEPA2(nn.Module):
    """Frozen V-JEPA-2 encoder.

    Output mode: per-patch embeddings mean-pooled into a (B, 1024) per-image
    representation. The factored visual adapter (Section 1 of the plan) gets
    raw per-patch features instead — see `encode_patches` below for that.
    """

    def __init__(self, model_name: str = VJEPA2_DEFAULT, device: str = "cpu"):
        super().__init__()
        from transformers import AutoModel, AutoVideoProcessor

        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.model.to(device)
        self.native_dim = VJEPA2_NATIVE_DIM

    def train(self, mode: bool = True):
        return super().train(False)

    def _to_video_batch(self, images: Iterable[Image.Image]) -> dict:
        """Treat each PIL image as a single-frame video clip."""
        # Each "clip" is a list of one frame.
        clips = [[img] for img in images]
        batch = self.processor(clips, return_tensors="pt").to(self.device)
        return batch

    @torch.no_grad()
    def encode(self, images: Iterable[Image.Image]) -> torch.Tensor:
        """Mean-pooled per-image embedding. Returns (B, 1024)."""
        batch = self._to_video_batch(images)
        out = self.model(**batch)
        # V-JEPA-2 returns (B, T*N_patches, D) features.
        feats = out.last_hidden_state                                # (B, T*N, D)
        return feats.mean(dim=1)                                      # (B, D=1024)

    @torch.no_grad()
    def encode_patches(self, images: Iterable[Image.Image]) -> torch.Tensor:
        """Per-patch features for the factored visual adapter.

        Returns (B, N_patches, 1024). The factored adapter pools this in three
        different ways (max, mean, attention) — see plan Section 1.
        """
        batch = self._to_video_batch(images)
        out = self.model(**batch)
        return out.last_hidden_state                                  # (B, N, D)
