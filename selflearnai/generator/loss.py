"""Loss orchestration for Phase 2a Pointer-Generator training.

Three pieces, all extracted from validated scripts:

  - perturb_h(h, mask): per-sample sequence-level perturbation.
    50% of samples get perturbed; 50/50 between Gaussian noise
    (δ=0.7) and 30% token-vector masking. Padding positions are
    never perturbed. Source:
    `scripts/stage2a_seq_conditioning.py:perturb_h` (commit 2a5ce56).

  - mixture_nll(log_probs, target_ids): NLL on the
    Pointer-Generator's mixture log-probabilities. Wrapper for
    F.nll_loss with the mixture's natural shape.

  - mse_activation_loss(decoder_hidden, mse_proj, target_h, mask):
    auxiliary MSE between the decoder's projected hidden states and
    the encoder's per-token activations on the ground-truth text
    (paper §4.2.2 ablation: +58% MAUVE over CE-only).

The training loop composition:
    L = mixture_nll(...) + λ_mse · mse_activation_loss(...)
where λ_mse defaults to 0.5 (validated in 2a.0e/0f).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def perturb_h(
    h: torch.Tensor,                    # [B, T_in, D_enc]
    h_mask: torch.Tensor,                # [B, T_in], 1 for real
    *,
    apply_prob: float = 0.3,
    gaussian_delta: float = 0.7,
    mask_token_rate: float = 0.3,
) -> torch.Tensor:
    """Sequence-level perturbation augmentation.

    For each sample (independently), with probability `apply_prob`
    perturb its encoder activations. When perturbing, 50/50 between:

      MODE A (Gaussian noise on real positions):
        h' = δ·h + sqrt(1−δ²)·ε,  ε ~ N(0, I)

      MODE B (token-vector masking):
        zero `mask_token_rate` fraction of real positions

    Padding positions are never perturbed.

    Defaults match the 2a.0e/0f validated recipe (apply_prob=0.3 was
    lower than 2a.0c's 0.5 so the pointer-attention sees more stable
    inputs during training).
    """
    B, T, _ = h.shape
    device = h.device
    apply = torch.rand(B, device=device) < apply_prob
    if not apply.any():
        return h
    mode_a = torch.rand(B, device=device) < 0.5
    out = h.clone()

    # MODE A: Gaussian noise on real positions of selected samples.
    a_idx = (apply & mode_a).nonzero(as_tuple=True)[0]
    if a_idx.numel() > 0:
        noise = torch.randn_like(out[a_idx])
        m = h_mask[a_idx].unsqueeze(-1)
        new = (
            gaussian_delta * out[a_idx]
            + (1.0 - gaussian_delta ** 2) ** 0.5 * noise
        )
        out[a_idx] = m * new + (1.0 - m) * out[a_idx]

    # MODE B: token-vector masking — zero `mask_token_rate` of real positions.
    b_idx = (apply & ~mode_a).nonzero(as_tuple=True)[0]
    if b_idx.numel() > 0:
        keep = (torch.rand(b_idx.numel(), T, device=device) >= mask_token_rate).float()
        keep = keep * h_mask[b_idx]
        out[b_idx] = (
            out[b_idx] * keep.unsqueeze(-1) * h_mask[b_idx].unsqueeze(-1)
            + out[b_idx] * (1.0 - h_mask[b_idx].unsqueeze(-1))
        )
    return out


def mixture_nll(
    log_probs: torch.Tensor,        # [B, T_out, V]
    target_ids: torch.Tensor,       # [B, T_out]
) -> torch.Tensor:
    """NLL on the Pointer-Generator's mixture log-probabilities.

    The decoder already returns log(p_gen·P_vocab + (1−p_gen)·P_copy);
    this just wraps F.nll_loss with the right reshape.
    """
    return F.nll_loss(
        log_probs.reshape(-1, log_probs.size(-1)),
        target_ids.reshape(-1),
    )


def mse_activation_loss(
    decoder_hidden: torch.Tensor,           # [B, T_out, h]
    mse_proj: torch.nn.Module,              # decoder.mse_proj
    target_h: torch.Tensor,                 # [B, T_out, D_enc]  (or T_in if matched)
    mask: torch.Tensor,                     # [B, T_out], 1 for real
) -> torch.Tensor:
    """MSE between projected decoder hidden states and the encoder's
    per-token activations on the ground-truth text.

    The decoder owns the projection (decoder.mse_proj: hidden_dim →
    encoder_dim) so this function is just the orchestration layer.
    """
    encoder_dim = target_h.size(-1)
    h_pred = mse_proj(decoder_hidden)            # [B, T, D_enc]
    mask_unsq = mask.unsqueeze(-1)
    mse_per = (h_pred - target_h).pow(2)
    return (mse_per * mask_unsq).sum() / (mask_unsq.sum() * encoder_dim).clamp(min=1.0)
