"""Non-AR transformer decoder with sequence conditioning + Pointer-Generator.

Extracted verbatim (with light cleanup) from
`scripts/stage2a_pointer.py:PointerSeqCondDecoder` (commit bd4c183),
which `scripts/stage2a_pointer_novel.py` (commit d554c6c) used to hit
TRULY_NOVEL_PASS — 80% bit-exact word-fidelity on truly-novel held-out.

The architecture is locked at plan §19.14. Forward pass:

  1. Project encoder activations h: D_enc → hidden_dim per token.
  2. Optional feature dropout on the projected memory.
  3. T_out learnable output position embeddings.
  4. nn.TransformerDecoder layers — bidirectional self-attention on
     output positions (NO causal mask — non-autoregressive) +
     cross-attention from output → memory.
  5. Vocab branch: token_head produces per-position vocab logits.
  6. Pointer branch: separate ptr_q/ptr_k attention over encoder
     positions → softmax → scatter onto vocab via input_token_ids.
  7. Gate: per-position p_gen ∈ [0, 1] via sigmoid.
  8. Final probability: p_gen·P_vocab + (1−p_gen)·P_copy.

Returns (final_log_probs, decoder_hidden, p_gen).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerSeqCondDecoder(nn.Module):
    """Validated by 2a.0e/2a.0f. Default config is the validated one."""

    def __init__(
        self,
        encoder_dim: int = 1024,
        hidden_dim: int = 512,
        t_max: int = 32,
        vocab_size: int = 30522,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_mult: int = 4,
        feat_dropout: float = 0.2,
        attn_dropout: float = 0.1,
        use_position_bias: bool = False,
        position_bias_init: float = 0.5,
    ) -> None:
        """Locked architecture from §19.14. The `use_position_bias` flag
        is the 2a.4 multi-subword-fix extension (defaults to False to
        preserve 2a.3 behavior).

        When True, adds a learnable T_out × T_in matrix added to the
        pointer's attention scores. Initialized with `position_bias_init`
        on the diagonal — a soft identity prior that says "output
        position N likely copies from encoder position N." For
        autoencoder-shaped tasks (target sentence == encoder input,
        which is Phase 2a's setup) this is the right inductive bias
        and helps multi-subword targets where per-position attention
        otherwise gets confused (see plan §19.14 documented limitation).
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.t_max = t_max
        self.vocab_size = vocab_size

        # Project encoder activations into the decoder's hidden space.
        self.cond_proj = nn.Linear(encoder_dim, hidden_dim)
        # Feature-level dropout on conditioning tokens (paper §4.2.2).
        # Lower than 2a.0c's 0.4 so the pointer-attention sees stable
        # cross-attention input.
        self.feat_dropout = nn.Dropout(p=feat_dropout)

        # Output position seeds (learned).
        self.output_pos_emb = nn.Parameter(
            torch.randn(t_max, hidden_dim) * 0.02
        )

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            batch_first=True,
            activation="gelu",
            dropout=attn_dropout,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=n_layers)

        # Vocab head.
        self.token_head = nn.Linear(hidden_dim, vocab_size)

        # Pointer head: separate Q/K projections so the pointer learns
        # its own attention pattern, independent of the transformer's
        # internal cross-attention.
        self.ptr_q = nn.Linear(hidden_dim, hidden_dim)
        self.ptr_k = nn.Linear(hidden_dim, hidden_dim)

        # Optional position-alignment bias for multi-subword fix (2a.4).
        # Soft identity bias on the pointer attention scores.
        if use_position_bias:
            bias = torch.eye(t_max) * position_bias_init
            self.position_bias = nn.Parameter(bias)
        else:
            self.register_parameter("position_bias", None)

        # Per-position p_gen ∈ [0, 1] via sigmoid.
        self.gen_gate = nn.Linear(hidden_dim, 1)

        # MSE projection: decoder hidden → encoder dim, used by the
        # activation-MSE auxiliary loss (paper recipe). Not used in
        # pointer mixture but kept here so the decoder owns its full
        # output surface.
        self.mse_proj = nn.Linear(hidden_dim, encoder_dim)

    def forward(
        self,
        encoder_h: torch.Tensor,             # [B, T_in, D_enc]
        encoder_mask: torch.Tensor,          # [B, T_in], 1 for real tokens
        encoder_token_ids: torch.Tensor,     # [B, T_in], token ids of input
        return_ptr_attn: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Returns (final_log_probs [B, T_out, V], decoder_hidden
        [B, T_out, h], p_gen [B, T_out, 1]).

        If return_ptr_attn=True, also returns ptr_attn [B, T_out, T_in]
        as a 4th tuple element. Used by training-time auxiliary losses
        like subword_chain_loss (Phase 2b.1) that supervise the pointer
        attention directly. Inference paths can leave it False."""
        B = encoder_h.size(0)
        memory = self.feat_dropout(self.cond_proj(encoder_h))    # [B, T_in, h]
        memory_pad_mask = (encoder_mask < 0.5)                   # True where padding

        out_seeds = self.output_pos_emb.unsqueeze(0).expand(B, -1, -1)
        out_hidden = self.transformer(
            tgt=out_seeds,
            memory=memory,
            memory_key_padding_mask=memory_pad_mask,
        )                                                        # [B, T_out, h]

        # ---- Vocab branch ----
        vocab_logits = self.token_head(out_hidden)               # [B, T_out, V]
        vocab_probs = F.softmax(vocab_logits, dim=-1)

        # ---- Pointer branch ----
        q = self.ptr_q(out_hidden)                               # [B, T_out, h]
        k = self.ptr_k(memory)                                   # [B, T_in,  h]
        ptr_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)

        # Add the learnable position-alignment bias if enabled (2a.4).
        # Slice to actual T_out × T_in shape — bias was initialized at
        # t_max × t_max so this just trims to the current sequence.
        if self.position_bias is not None:
            T_out_actual = ptr_scores.size(1)
            T_in_actual = ptr_scores.size(2)
            ptr_scores = ptr_scores + self.position_bias[
                :T_out_actual, :T_in_actual,
            ].unsqueeze(0)

        ptr_scores = ptr_scores.masked_fill(
            memory_pad_mask.unsqueeze(1), float("-inf"),
        )
        ptr_attn = F.softmax(ptr_scores, dim=-1)                 # [B, T_out, T_in]

        # Scatter pointer attention probs to the vocab via
        # encoder_token_ids: copy_probs[b, i, k] = sum_j(ptr_attn[b, i, j])
        # for j s.t. encoder_token_ids[b, j] == k.
        copy_probs = torch.zeros_like(vocab_probs)               # [B, T_out, V]
        idx = encoder_token_ids.unsqueeze(1).expand(-1, vocab_probs.size(1), -1)
        copy_probs = copy_probs.scatter_add(-1, idx, ptr_attn)

        # ---- Gate + mixture ----
        p_gen = torch.sigmoid(self.gen_gate(out_hidden))         # [B, T_out, 1]
        final_probs = p_gen * vocab_probs + (1.0 - p_gen) * copy_probs
        final_log_probs = torch.log(final_probs.clamp_min(1e-12))

        if return_ptr_attn:
            return final_log_probs, out_hidden, p_gen, ptr_attn
        return final_log_probs, out_hidden, p_gen
