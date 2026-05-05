"""Inference sampling for the Pointer-Generator decoder.

Two functions:

  - decode_to_text: greedy argmax decode → tokenizer.decode. Used by
    every eval (cos / grammar / word-fidelity gates).

  - multi_candidate_sample: K stochastic samples (Gumbel noise on the
    log-probs OR temperature on p_gen), picks the best by
    cos(encode(candidate), psi_target). Plan §19.14 sub-task 2a.5
    will validate this provides ≥0.02 cos lift over greedy.

The simple decode_to_text is what 2a.0e/2a.0f used and what 2a.3
(production training) will use for its primary eval.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def decode_to_text(
    log_probs: torch.Tensor,        # [B, T_out, V]
    tokenizer,
    skip_special_tokens: bool = True,
) -> list[str]:
    """Greedy argmax decode → tokenizer.decode → list[str], one per
    batch item. Mirrors the inference path in 2a.0e and 2a.0f.
    """
    gen_ids = log_probs.argmax(dim=-1)              # [B, T_out]
    return [
        tokenizer.decode(row.tolist(), skip_special_tokens=skip_special_tokens)
        for row in gen_ids
    ]


@torch.no_grad()
def multi_candidate_sample(
    decoder,                                            # PointerSeqCondDecoder
    encoder_h: torch.Tensor,                            # [B, T_in, D_enc]
    encoder_mask: torch.Tensor,                         # [B, T_in]
    encoder_token_ids: torch.Tensor,                    # [B, T_in]
    *,
    psi_target: torch.Tensor,                           # [B, D_enc] — pooled goal
    encode_text_fn: Callable[[list[str]], torch.Tensor],
    tokenizer,
    k: int = 5,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[list[str], torch.Tensor, list[list[str]], list[list[float]]]:
    """K stochastic samples per item, pick the best by cos to psi_target.

    Method: add Gumbel noise to the mixture log-probs at temperature
    T → argmax → text. Repeat K times. For each item, return the
    candidate whose re-encoded ψ has the highest cosine to psi_target.

    Plan §19.14 sub-task 2a.5 acceptance: median cos lift ≥ 0.02 with
    K=5 vs K=1 on held-out. If lift < 0.005, the multi-candidate
    machinery isn't pulling its weight.

    Returns:
      best_text:        [B] selected best candidate text per item
      best_cos:         [B] cosine of selected candidate to psi_target
      all_candidates:   [B][K] all K texts per item
      all_cos:          [B][K] all K cosines per item
    """
    if seed is not None:
        torch.manual_seed(seed)

    B = encoder_h.size(0)
    log_probs, _, _ = decoder(encoder_h, encoder_mask, encoder_token_ids)
    # log_probs: [B, T_out, V]

    all_candidates: list[list[str]] = [[] for _ in range(B)]
    all_cos: list[list[float]] = [[] for _ in range(B)]

    for _ in range(k):
        # Gumbel-perturbed argmax: add noise drawn from Gumbel(0,1) and
        # take argmax. Standard trick for sampling from log-probs.
        gumbel = -torch.log(-torch.log(
            torch.rand_like(log_probs).clamp_min(1e-12),
        ).clamp_min(1e-12))
        sampled_ids = (log_probs / temperature + gumbel).argmax(dim=-1)
        for b in range(B):
            text = tokenizer.decode(
                sampled_ids[b].tolist(), skip_special_tokens=True,
            )
            all_candidates[b].append(text)

    # Score every candidate via re-encode. Batch by item to keep memory
    # manageable on long candidate lists; each call to encode_text_fn
    # encodes K candidates for one item.
    best_text: list[str] = []
    best_cos_list: list[float] = []
    for b in range(B):
        cands = all_candidates[b]
        non_empty = [c if c.strip() else " " for c in cands]   # encode_text_fn dislikes empty
        psi_cands = encode_text_fn(non_empty)                  # [K, D_enc]
        cos_b = F.cosine_similarity(
            psi_cands, psi_target[b].unsqueeze(0).expand_as(psi_cands), dim=-1,
        )
        all_cos[b] = [float(x.item()) for x in cos_b]
        idx = int(cos_b.argmax().item())
        best_text.append(cands[idx])
        best_cos_list.append(float(cos_b[idx].item()))

    return best_text, torch.tensor(best_cos_list), all_candidates, all_cos
