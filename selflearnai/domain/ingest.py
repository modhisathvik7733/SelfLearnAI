"""Universal domain ingestion orchestrator (Stage 3 / sub-task 3.4).

Single function `ingest_domain(...)` composes:

    Stage 3.1's training recipe   (per-domain decoder via Phase 2a recipe)
  + Stage 3.2's energy fits        (Gaussian + MLP density on in-domain ψ)
  + Stage 3.3's domain registry    (versioned on-disk artifacts)

into one round-trip call that turns (train sentences, held-out sentences,
src/tgt word pairs) into a registered, callable domain.

Per plan §19.17 sub-task 3.4 acceptance: a fresh domain can be
ingested in one call and its decoder + energy + provenance live in
the registry's `artifacts/<domain_id>__v1/` directory.

Usage:

    cfg = IngestionConfig(
        domain_id="temporal",
        decoder_steps=3000,
        mlp_steps=1500,
        cos_min=0.85, grammar_pass_rate=0.95, word_fidelity_min=0.70,
    )
    result = ingest_domain(
        cfg,
        train_sents=train_sents,
        train_pair_words=train_pairs,            # [(src, tgt), ...]
        holdout_sents=holdout_sents,
        holdout_pair_words=holdout_pairs,
        encoder_tokenizer=tok,
        encoder_model=mdl,
        encoder_dim=1024,
        device="cuda",
        registry=DomainRegistry("data/registries/domains"),
        out_of_domain_psi=phase2a_psi,            # optional, for energy AUC
        workdir="data/explanations_v3/checkpoints",
    )
    assert result.registry_entry.current().domain_id == "temporal"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .energy import GaussianEnergy, MLPEnergyModel, roc_auc
from .registry import DomainEntry, DomainRegistry


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class IngestionConfig:
    """All knobs for one round-trip ingestion. Defaults match plan
    §19.14 (locked Phase 2a recipe) and §19.17 (Stage 3 gate values)."""
    domain_id: str
    # Decoder architecture (locked from §19.14)
    hidden_dim: int = 512
    t_max: int = 32
    n_layers: int = 4
    n_heads: int = 8
    ffn_mult: int = 4
    feat_dropout: float = 0.2
    attn_dropout: float = 0.1
    # Decoder training
    decoder_steps: int = 10000
    batch_size: int = 32
    lr: float = 2e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    # Loss recipe
    use_mse: bool = True
    mse_weight: float = 0.5
    use_perturb: bool = True
    perturb_prob: float = 0.3
    gaussian_delta: float = 0.7
    mask_token_rate: float = 0.3
    # Energy training
    energy_kind: str = "MLPEnergyModel"   # or "GaussianEnergy"
    mlp_hidden1: int = 256
    mlp_hidden2: int = 64
    mlp_steps: int = 3000
    mlp_batch_size: int = 64
    mlp_lr: float = 1e-3
    mlp_margin: float = 1.0
    # Gates (Stage 3 thresholds, slightly looser than 2a.3)
    cos_min: float = 0.85
    grammar_pass_rate: float = 0.95
    word_fidelity_min: float = 0.70
    auc_min: float = 0.95
    # Logging cadence
    log_every: int = 200


@dataclass
class IngestionResult:
    """What ingest_domain returns once the round-trip completes."""
    domain_id: str
    n_train_sents: int
    n_holdout_sents: int
    decoder_n_params: int
    decoder_loss_history: list[dict[str, Any]] = field(default_factory=list)
    decoder_ckpt_path: str = ""
    energy_kind: str = ""
    energy_ckpt_path: str = ""
    energy_fit_stats: dict[str, Any] = field(default_factory=dict)
    energy_auc: Optional[float] = None
    decoder_eval: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)
    all_gates_pass: bool = False
    registry_entry: Optional[DomainEntry] = None
    verdict: str = ""
    message: str = ""


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def ingest_domain(
    cfg: IngestionConfig,
    *,
    train_sents: list[str],
    train_pair_words: list[tuple[str, str]],
    holdout_sents: list[str],
    holdout_pair_words: list[tuple[str, str]],
    encoder_tokenizer: Any,
    encoder_model: Any,
    encoder_dim: int,
    device: str,
    registry: DomainRegistry,
    workdir: str | Path,
    out_of_domain_psi: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> IngestionResult:
    """Train a per-domain decoder + energy model, evaluate, register.

    Required call sites already have the encoder loaded (we don't load
    it inside the orchestrator — the encoder is the most expensive
    object in the pipeline and callers may share it across multiple
    ingest_domain calls).

    The function returns an IngestionResult AND mutates `registry` by
    adding a new DomainEntry under cfg.domain_id.
    """
    # Imported here so the domain package doesn't pull in generator at
    # import time (the generator package depends on transformers, which
    # we don't want forced on every domain import).
    from selflearnai.generator import (
        PointerSeqCondDecoder,
        perturb_h,
        mixture_nll,
        word_pair_fidelity,
        GenerationVerdict,
    )
    from selflearnai.generator.eval import grammar_proxy, roll_up_gates
    from selflearnai.generator.loss import mse_activation_loss
    try:
        from scripts.stage2a_quick_probe import grammar_grade
    except ImportError:
        def grammar_grade(text: str) -> tuple[int, bool]:
            proxy = grammar_proxy(text)
            return (0 if proxy >= 0.55 else 1, proxy >= 0.55)

    n_train = len(train_sents)
    n_holdout = len(holdout_sents)
    if len(train_pair_words) != n_train:
        raise ValueError(
            f"len(train_pair_words)={len(train_pair_words)} != len(train_sents)={n_train}"
        )
    if len(holdout_pair_words) != n_holdout:
        raise ValueError(
            f"len(holdout_pair_words)={len(holdout_pair_words)} "
            f"!= len(holdout_sents)={n_holdout}"
        )

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[ingest_domain] domain_id={cfg.domain_id!r}  "
              f"n_train={n_train}  n_holdout={n_holdout}")

    # ---- Encode ------------------------------------------------------
    train_h, train_h_mask = _encode_activations(
        train_sents, encoder_tokenizer, encoder_model, device, t_max=cfg.t_max,
    )
    holdout_h, holdout_h_mask = _encode_activations(
        holdout_sents, encoder_tokenizer, encoder_model, device, t_max=cfg.t_max,
    )
    holdout_psi = _encode_pooled(
        holdout_sents, encoder_tokenizer, encoder_model, device, max_length=cfg.t_max,
    )
    train_tok = encoder_tokenizer(
        train_sents, padding="max_length", truncation=True,
        max_length=cfg.t_max, return_tensors="pt",
    ).to(device)
    train_ids = train_tok.input_ids
    holdout_tok = encoder_tokenizer(
        holdout_sents, padding="max_length", truncation=True,
        max_length=cfg.t_max, return_tensors="pt",
    ).to(device)
    holdout_ids = holdout_tok.input_ids

    # ---- Decoder (locked architecture) -------------------------------
    torch.manual_seed(cfg.seed)
    decoder = PointerSeqCondDecoder(
        encoder_dim=encoder_dim, hidden_dim=cfg.hidden_dim, t_max=cfg.t_max,
        vocab_size=encoder_tokenizer.vocab_size,
        n_layers=cfg.n_layers, n_heads=cfg.n_heads, ffn_mult=cfg.ffn_mult,
        feat_dropout=cfg.feat_dropout, attn_dropout=cfg.attn_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    if verbose:
        print(f"[ingest_domain] decoder: {n_params/1e6:.2f}M params, "
              f"training {cfg.decoder_steps} steps")

    opt = torch.optim.AdamW(
        decoder.parameters(), lr=cfg.lr,
        betas=(0.9, 0.95), weight_decay=cfg.weight_decay,
    )

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
        return cfg.lr

    decoder.train()
    loss_history: list[dict[str, Any]] = []
    use_mse = cfg.use_mse and cfg.mse_weight > 0.0
    for step in range(cfg.decoder_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad()
        idx = torch.randint(0, n_train, (cfg.batch_size,), device=device)
        h_batch = train_h[idx]
        h_mask_batch = train_h_mask[idx]
        ids_batch = train_ids[idx]
        h_input = perturb_h(
            h_batch, h_mask_batch,
            apply_prob=(cfg.perturb_prob if cfg.use_perturb else 0.0),
            gaussian_delta=cfg.gaussian_delta,
            mask_token_rate=cfg.mask_token_rate,
        )
        log_probs, hidden_out, p_gen = decoder(h_input, h_mask_batch, ids_batch)
        nll = mixture_nll(log_probs, ids_batch)
        if use_mse:
            mse_loss = mse_activation_loss(
                hidden_out, decoder.mse_proj, h_batch, h_mask_batch,
            )
            loss = nll + cfg.mse_weight * mse_loss
        else:
            mse_loss = torch.tensor(0.0, device=device)
            loss = nll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=cfg.grad_clip)
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.decoder_steps - 1:
            loss_history.append({
                "step": step, "loss": float(loss.item()),
                "nll": float(nll.item()), "mse": float(mse_loss.item()),
                "p_gen_mean": float(p_gen.mean().item()),
            })
            if verbose:
                print(f"  step {step:>5}/{cfg.decoder_steps}  "
                      f"loss={loss.item():.4f}  nll={nll.item():.4f}  "
                      f"mse={mse_loss.item():.4f}  p_gen={p_gen.mean().item():.3f}")

    decoder_ckpt = workdir / f"{cfg.domain_id}_decoder_v1.pt"
    torch.save(decoder.state_dict(), decoder_ckpt)
    if verbose:
        print(f"[ingest_domain] decoder saved → {decoder_ckpt}")

    # ---- Decoder eval (gates) ----------------------------------------
    decoder.eval()
    eval_bs = 32
    all_gen_ids = []
    all_p_gen = []
    with torch.no_grad():
        for s in range(0, n_holdout, eval_bs):
            log_probs, _, p_gen = decoder(
                holdout_h[s:s + eval_bs],
                holdout_h_mask[s:s + eval_bs],
                holdout_ids[s:s + eval_bs],
            )
            all_gen_ids.append(log_probs.argmax(dim=-1))
            all_p_gen.append(p_gen.squeeze(-1))
    gen_ids = torch.cat(all_gen_ids, dim=0)
    p_gen_all = torch.cat(all_p_gen, dim=0)
    pad_id = encoder_tokenizer.pad_token_id

    gen_texts = [
        encoder_tokenizer.decode(gen_ids[i].tolist(), skip_special_tokens=True)
        for i in range(n_holdout)
    ]
    psi_gen = _encode_pooled(
        gen_texts, encoder_tokenizer, encoder_model, device, max_length=cfg.t_max,
    )
    cos_recovered = F.cosine_similarity(psi_gen, holdout_psi, dim=-1)

    verdicts = []
    for i in range(n_holdout):
        gen_text = gen_texts[i]
        target_text = holdout_sents[i]
        src_word, tgt_word = holdout_pair_words[i]
        cos = float(cos_recovered[i].item())
        n_errors, gpass = grammar_grade(gen_text)
        proxy = grammar_proxy(gen_text)
        src_in, tgt_in, both_in = word_pair_fidelity(src_word, tgt_word, gen_text)
        exact = gen_text.strip() == target_text.strip()
        real_mask = (holdout_ids[i] != pad_id).float()
        p_gen_mean = float(
            (p_gen_all[i] * real_mask).sum().item()
            / real_mask.sum().clamp(min=1.0).item()
        )
        verdicts.append(GenerationVerdict(
            target=target_text, generated=gen_text, concept=cfg.domain_id,
            src_word=src_word, tgt_word=tgt_word,
            cos_recovered=cos, grammar_pass=gpass, grammar_n_errors=n_errors,
            grammar_proxy=proxy,
            src_in_gen=src_in, tgt_in_gen=tgt_in, both_in_gen=both_in,
            exact_match=exact, p_gen_mean=p_gen_mean,
        ))

    gates = roll_up_gates(
        verdicts,
        cos_min=cfg.cos_min,
        grammar_pass_rate=cfg.grammar_pass_rate,
        word_fidelity_min=cfg.word_fidelity_min,
    )
    decoder_eval = {
        "median_cos": gates.median_cos,
        "n_cos_pass": gates.n_cos_pass,
        "n_grammar_pass": gates.n_grammar_pass,
        "n_both_in": gates.n_both_in,
        "n_exact_match": gates.n_exact_match,
        "p_gen_overall": gates.p_gen_overall,
    }

    # ---- Energy model (in-domain ψ training) -------------------------
    psi_train_in = _encode_pooled(
        train_sents, encoder_tokenizer, encoder_model, device, max_length=cfg.t_max,
    ).cpu()
    if verbose:
        print(f"[ingest_domain] training energy model "
              f"({cfg.energy_kind}) on {psi_train_in.size(0)} in-domain ψ")

    if cfg.energy_kind == "GaussianEnergy":
        energy = GaussianEnergy(dim=encoder_dim)
        energy_fit = energy.fit(psi_train_in)
        energy_ckpt = workdir / f"{cfg.domain_id}_energy_gauss_v1.pt"
        energy.save(energy_ckpt)
        with torch.no_grad():
            e_in = energy.energy(_encode_pooled(
                holdout_sents, encoder_tokenizer, encoder_model, device,
                max_length=cfg.t_max,
            ).cpu())
            if out_of_domain_psi is not None:
                e_out = energy.energy(out_of_domain_psi.cpu())
    elif cfg.energy_kind == "MLPEnergyModel":
        energy = MLPEnergyModel(
            dim=encoder_dim, hidden1=cfg.mlp_hidden1, hidden2=cfg.mlp_hidden2,
        )
        energy_fit = energy.fit(
            psi_train_in,
            steps=cfg.mlp_steps, batch_size=cfg.mlp_batch_size,
            lr=cfg.mlp_lr, margin=cfg.mlp_margin,
            device=device, seed=cfg.seed,
        )
        energy.eval()
        energy.cpu()
        energy_ckpt = workdir / f"{cfg.domain_id}_energy_mlp_v1.pt"
        energy.save(energy_ckpt)
        with torch.no_grad():
            e_in = energy.energy(_encode_pooled(
                holdout_sents, encoder_tokenizer, encoder_model, device,
                max_length=cfg.t_max,
            ).cpu())
            if out_of_domain_psi is not None:
                e_out = energy.energy(out_of_domain_psi.cpu())
    else:
        raise ValueError(f"unknown energy_kind: {cfg.energy_kind!r}")

    # Sanitize fit stats for JSON-friendly storage in registry provenance
    energy_fit_stats = {
        k: v for k, v in energy_fit.items() if k != "loss_history"
    }

    energy_auc: Optional[float] = None
    if out_of_domain_psi is not None:
        n_in = e_in.numel()
        n_out = e_out.numel()
        y_probe = torch.cat([
            torch.zeros(n_in, dtype=torch.long),
            torch.ones(n_out, dtype=torch.long),
        ])
        scores = torch.cat([e_in, e_out])
        energy_auc = roc_auc(y_probe, scores)
        if verbose:
            print(f"[ingest_domain] energy ROC-AUC vs out-of-domain: "
                  f"{energy_auc:.4f}  (target ≥ {cfg.auc_min})")

    # ---- Register in domain registry ---------------------------------
    provenance = {
        "n_train_sents": n_train,
        "n_holdout_sents": n_holdout,
        "decoder_eval": decoder_eval,
        "decoder_n_params": n_params,
        "decoder_steps": cfg.decoder_steps,
        "energy_eval": {
            "kind": cfg.energy_kind,
            "fit_stats": energy_fit_stats,
            "auc": energy_auc,
            "auc_min": cfg.auc_min,
        },
        "gates": {
            "cos_gate": gates.cos_gate,
            "grammar_gate": gates.grammar_gate,
            "word_fidelity_gate": gates.word_fidelity_gate,
            "all_pass": gates.all_pass,
        },
    }
    registry_entry = registry.register(
        cfg.domain_id,
        decoder_src_path=decoder_ckpt,
        energy_src_path=energy_ckpt,
        energy_kind=cfg.energy_kind,
        provenance=provenance,
    )
    if verbose:
        v = registry_entry.current()
        print(f"[ingest_domain] registered: {registry_entry.domain_id} "
              f"v{v.version}  decoder={v.decoder_path}  "
              f"energy={v.energy_path} ({v.energy_kind})")

    # ---- Verdict -----------------------------------------------------
    decoder_pass = gates.all_pass
    energy_pass = energy_auc is None or energy_auc >= cfg.auc_min
    all_pass = decoder_pass and energy_pass
    if all_pass:
        verdict = "INGEST_PASS"
        message = (
            f"Domain {cfg.domain_id!r} ingested cleanly. Decoder gates "
            f"all pass (median cos {gates.median_cos:.4f}, "
            f"word-fidelity {gates.n_both_in}/{n_holdout}); "
            f"energy {'AUC ' + format(energy_auc, '.4f') if energy_auc is not None else 'fit'} "
            f"meets target. Registered as v{registry_entry.current_version}."
        )
    elif not decoder_pass:
        verdict = "INGEST_DECODER_GATE_FAIL"
        message = (
            f"Decoder gates FAIL: cos={gates.cos_gate} "
            f"grammar={gates.grammar_gate} word_fidelity={gates.word_fidelity_gate}. "
            f"Domain registered (artifacts saved for inspection) but flagged."
        )
    else:
        verdict = "INGEST_ENERGY_AUC_FAIL"
        message = (
            f"Decoder gates pass but energy ROC-AUC ({energy_auc:.4f}) "
            f"< target ({cfg.auc_min}). Domain registered; consider "
            f"escalating to GaussianEnergy or longer MLP training."
        )

    return IngestionResult(
        domain_id=cfg.domain_id,
        n_train_sents=n_train,
        n_holdout_sents=n_holdout,
        decoder_n_params=n_params,
        decoder_loss_history=loss_history,
        decoder_ckpt_path=str(decoder_ckpt),
        energy_kind=cfg.energy_kind,
        energy_ckpt_path=str(energy_ckpt),
        energy_fit_stats=energy_fit_stats,
        energy_auc=energy_auc,
        decoder_eval=decoder_eval,
        gates={
            "cos_gate": gates.cos_gate,
            "grammar_gate": gates.grammar_gate,
            "word_fidelity_gate": gates.word_fidelity_gate,
            "all_pass": gates.all_pass,
        },
        all_gates_pass=all_pass,
        registry_entry=registry_entry,
        verdict=verdict,
        message=message,
    )


# ---------------------------------------------------------------------------
# Encoder helpers (kept inline so this file is the single ingestion entrypoint)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_pooled(sents, tok, mdl, device, max_length=64):
    inputs = tok(sents, padding=True, truncation=True, max_length=max_length,
                 return_tensors="pt").to(device)
    out = mdl(**inputs).last_hidden_state
    mask = inputs.attention_mask.unsqueeze(-1).float()
    return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def _encode_activations(sents, tok, mdl, device, t_max=32):
    inputs = tok(sents, padding="max_length", truncation=True,
                 max_length=t_max, return_tensors="pt").to(device)
    out = mdl(**inputs).last_hidden_state
    return out, inputs.attention_mask.float()
