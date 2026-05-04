"""GTE-base text encoder wrapper. Frozen, never trained.

GTE (General Text Embedding) is a contrastive sentence-embedding model. NOT
trained with next-token prediction — it learns by pulling related sentences
close in embedding space. Output dim: 768.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


GTE_MODEL_NAME = "thenlper/gte-base"
GTE_NATIVE_DIM = 768


class FrozenGTE(nn.Module):
    """Frozen GTE-base. Always in eval mode; gradients always disabled.

    Usage:
        gte = FrozenGTE()
        emb = gte.encode(["the cat sat", "a dog ran"])   # (B, 768)
    """

    def __init__(self, model_name: str = GTE_MODEL_NAME, device: str = "cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.model.to(device)
        self.native_dim = GTE_NATIVE_DIM

    def train(self, mode: bool = True):
        # Override: foundation is permanently in eval mode.
        return super().train(False)

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 128) -> torch.Tensor:
        """Encode a list of sentences into (B, 768) embeddings.

        Uses GTE's standard pooling: average over token embeddings, masked by
        attention mask. Returns float32 tensors on self.device.
        """
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**batch)
        # Mean-pool with attention mask.
        last_hidden = out.last_hidden_state                          # (B, T, 768)
        mask = batch["attention_mask"].unsqueeze(-1).float()         # (B, T, 1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled                                                # (B, 768)
