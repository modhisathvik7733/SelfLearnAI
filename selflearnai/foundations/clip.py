"""CLIP wrapper. Frozen, never trained.

CLIP (Contrastive Language-Image Pretraining) is the cross-modal anchor in
the star topology — see plan section 1. NOT next-token prediction; trained
with contrastive image-text alignment. Output dim: 512 (text and vision).

We use the explicit `WithProjection` classes (rather than the unified
CLIPModel + get_text_features / get_image_features convenience methods)
because their return type — ModelOutput with `.text_embeds` / `.image_embeds`
fields — is stable across transformers versions. The convenience methods'
return type changed between versions and is fragile.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_NATIVE_DIM = 512


class FrozenCLIP(nn.Module):
    """Frozen CLIP ViT-B/32. Both encoders permanently in eval mode.

    Uses CLIPTextModelWithProjection + CLIPVisionModelWithProjection —
    these return `text_embeds` / `image_embeds` fields directly, with the
    projection-head output, no version-fragile pooling logic in our code.
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = "cpu"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_name)
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        for m in (self.text_model, self.vision_model):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        self.device = device
        self.text_model.to(device)
        self.vision_model.to(device)
        self.native_dim = CLIP_NATIVE_DIM

    def train(self, mode: bool = True):
        # Override: foundation is permanently in eval mode.
        return super().train(False)

    @torch.no_grad()
    def encode_text(self, texts: list[str], max_length: int = 77) -> torch.Tensor:
        """(B, 512) CLIP text embeddings (post-projection, pre-normalization)."""
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        out = self.text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return out.text_embeds                                          # (B, 512)

    @torch.no_grad()
    def encode_image(self, images: Iterable[Image.Image]) -> torch.Tensor:
        """(B, 512) CLIP vision embeddings (post-projection, pre-normalization).
        Accepts PIL Images.
        """
        batch = self.image_processor(
            images=list(images),
            return_tensors="pt",
        ).to(self.device)
        out = self.vision_model(pixel_values=batch["pixel_values"])
        return out.image_embeds                                         # (B, 512)
