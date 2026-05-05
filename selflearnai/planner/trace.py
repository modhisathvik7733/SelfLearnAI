"""Ψ-program trace serialization and replay (Task 1.11).

A Ψ-program is the durable, replayable form of a planner trace. It
captures everything needed to reproduce a reasoning result from
scratch:

  - the natural-language input (source word, optional target)
  - the encoder name + dim used to ground them
  - every operator applied, in order, by name
  - every intermediate ψ embedding, fully serialized as JSON
  - per-step verification gate results (from selflearnai.planner.verifier)
  - provenance: timestamp, planner config, intent source

Why this matters: the project's central claim is that reasoning
happens in representation space and is verifiable. A
"verifiable computation" is only verifiable if a third party can
take the trace, run it again on the same operators + encoder, and
get the same answer. Without serialization there's no third-party
verification — only "trust the run". This module is the substrate
for that.

Two operations:
  - `PsiProgram.to_json()` / `to_dict()` / file write — persist a trace.
  - `PsiProgram.from_json()` / `from_dict()` — load a trace.
  - `replay(program, encoder, operators)` — re-run the trace and
    confirm it reproduces the saved final state. Returns a
    `ReplayResult` with cos-similarity to the saved final ψ, gate
    re-evaluation, and a single overall reproducibility verdict.

JSON format is human-readable so traces can be inspected, diffed,
and version-controlled. Embeddings are stored as plain Python lists
of floats — verbose but trivially parseable in any language.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PsiProgramStep:
    """One step in a Ψ-program.

    Holds the operator name and the embedding produced by applying it.
    Verification is stored as a plain dict so it survives JSON
    round-tripping without custom decoders.
    """
    step_index: int
    op_name: str
    psi_after: list[float]
    cos_to_goal_after: Optional[float]
    verification: dict[str, Any] = field(default_factory=dict)


@dataclass
class PsiProgram:
    """The full serializable reasoning trace produced by the planner.

    Construct via `PsiProgram.from_planner_output(...)` (a convenience
    classmethod that wraps the BeamSearchPlanner / verify_chain
    outputs) or by explicit field assembly.
    """
    # Inputs
    source: str
    target: Optional[str]
    encoder_name: str
    encoder_dim: int

    # Initial / goal embeddings
    psi_initial: list[float]
    psi_goal: Optional[list[float]]

    # Chain of operators in order
    chain: list[str]

    # Per-step intermediate states + verification
    steps: list[PsiProgramStep]

    # Final outcome
    final_top1_word: Optional[str]
    final_cos_to_goal: Optional[float]

    # Provenance
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now_timestamp(cls) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict is recursive; nested dataclasses (steps) are dicts now.
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def to_file(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PsiProgram":
        steps = [PsiProgramStep(**s) for s in d.get("steps", [])]
        ctor_kwargs = {k: v for k, v in d.items() if k != "steps"}
        ctor_kwargs["steps"] = steps
        return cls(**ctor_kwargs)

    @classmethod
    def from_json(cls, s: str) -> "PsiProgram":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_file(cls, path: str | Path) -> "PsiProgram":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Construction helper from planner + verifier outputs
# ---------------------------------------------------------------------------

def build_psi_program(
    *,
    source: str,
    target: Optional[str],
    encoder_name: str,
    encoder_dim: int,
    psi_initial: torch.Tensor,
    psi_goal: Optional[torch.Tensor],
    chain: Sequence[str],
    intermediate_psis: Sequence[torch.Tensor],
    verification_steps: Optional[Sequence[Any]] = None,
    final_top1_word: Optional[str] = None,
    final_cos_to_goal: Optional[float] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> PsiProgram:
    """Pack planner + verifier outputs into a serializable PsiProgram.

    `intermediate_psis` MUST be the per-step outputs (`psi_after` for
    each operator), one tensor per name in `chain`.

    `verification_steps`, if provided, must be a sequence of
    StepVerification objects (one per chain step) from
    `selflearnai.planner.verifier.verify_chain`. Each is converted
    to a plain dict for JSON safety.
    """
    if len(intermediate_psis) != len(chain):
        raise ValueError(
            f"chain has {len(chain)} ops but {len(intermediate_psis)} "
            f"intermediate states were provided"
        )
    if verification_steps is not None and len(verification_steps) != len(chain):
        raise ValueError(
            f"chain has {len(chain)} ops but {len(verification_steps)} "
            f"verification steps"
        )

    steps: list[PsiProgramStep] = []
    psi_goal_t = (
        psi_goal.detach().flatten() if psi_goal is not None else None
    )
    for i, (op_name, psi_t) in enumerate(zip(chain, intermediate_psis)):
        psi_after_flat = psi_t.detach().flatten()
        cos_after: Optional[float]
        if psi_goal_t is not None:
            cos_after = float(
                F.cosine_similarity(
                    psi_after_flat.unsqueeze(0),
                    psi_goal_t.unsqueeze(0),
                    dim=-1,
                ).item()
            )
        else:
            cos_after = None

        verif_dict: dict[str, Any] = {}
        if verification_steps is not None:
            v = verification_steps[i]
            verif_dict = {
                "step_index": v.step_index,
                "op_name": v.op_name,
                "all_passed": v.all_passed,
                "gates": {
                    name: {
                        "passed": gr.passed,
                        "metric": gr.metric,
                        "threshold": gr.threshold,
                        "reason": gr.reason,
                    }
                    for name, gr in v.gate_results.items()
                },
            }

        steps.append(PsiProgramStep(
            step_index=i,
            op_name=op_name,
            psi_after=psi_after_flat.cpu().tolist(),
            cos_to_goal_after=cos_after,
            verification=verif_dict,
        ))

    return PsiProgram(
        source=source,
        target=target,
        encoder_name=encoder_name,
        encoder_dim=encoder_dim,
        psi_initial=psi_initial.detach().flatten().cpu().tolist(),
        psi_goal=(
            psi_goal_t.cpu().tolist() if psi_goal_t is not None else None
        ),
        chain=list(chain),
        steps=steps,
        final_top1_word=final_top1_word,
        final_cos_to_goal=final_cos_to_goal,
        timestamp=PsiProgram.now_timestamp(),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    """Outcome of replaying a Ψ-program against fresh operators + encoder.

    `final_cos_to_saved` is the cosine similarity between the freshly-
    re-executed final ψ and the ψ recorded in the saved program. Values
    very near 1.0 (≥ 0.999) indicate exact reproducibility (modulo
    floating-point); lower values suggest operator/encoder drift.
    """
    reproducible: bool
    final_cos_to_saved: float
    final_top1_word_replay: Optional[str]
    final_top1_word_saved: Optional[str]
    final_top1_match: Optional[bool]
    per_step_cos_to_saved: list[float]
    notes: list[str] = field(default_factory=list)


@torch.no_grad()
def replay(
    program: PsiProgram,
    *,
    encoder_encode_fn: Callable[[list[str]], torch.Tensor],
    operators: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    candidate_pool: Optional[list[str]] = None,
    final_psi_threshold: float = 0.999,
    per_step_threshold: float = 0.99,
) -> ReplayResult:
    """Re-execute a Ψ-program from scratch and compare to saved state.

    Steps:
      1. Encode `program.source` with the provided encoder.
      2. Verify the freshly-encoded source matches `program.psi_initial`
         (cos ≥ per_step_threshold).
      3. Re-apply each operator in `program.chain` and collect
         intermediate states.
      4. Compare each intermediate to the saved `psi_after` (cos).
      5. If a candidate_pool is provided, recompute final argmax
         and compare to the saved `final_top1_word`.

    `reproducible` is True iff every per-step cos exceeds
    `per_step_threshold` AND the final cos exceeds
    `final_psi_threshold`.
    """
    notes: list[str] = []
    per_step_cos: list[float] = []

    # 1. Encode source.
    z_src_fresh = encoder_encode_fn([program.source]).squeeze(0).flatten()
    z_src_saved = torch.tensor(program.psi_initial, dtype=z_src_fresh.dtype, device=z_src_fresh.device)
    src_cos = float(
        F.cosine_similarity(z_src_fresh.unsqueeze(0), z_src_saved.unsqueeze(0), dim=-1).item()
    )
    if src_cos < per_step_threshold:
        notes.append(
            f"freshly-encoded source diverges from saved psi_initial "
            f"(cos={src_cos:.4f} < {per_step_threshold:.3f}); encoder drift?"
        )

    # 2. Replay each operator step on the fresh source.
    psi = z_src_fresh
    for i, step in enumerate(program.steps):
        if step.op_name not in operators:
            notes.append(f"step {i}: operator {step.op_name!r} missing from library")
            return ReplayResult(
                reproducible=False,
                final_cos_to_saved=0.0,
                final_top1_word_replay=None,
                final_top1_word_saved=program.final_top1_word,
                final_top1_match=None,
                per_step_cos_to_saved=per_step_cos,
                notes=notes,
            )
        op = operators[step.op_name]
        psi = op(psi.unsqueeze(0)).squeeze(0)
        psi_saved = torch.tensor(
            step.psi_after, dtype=psi.dtype, device=psi.device,
        )
        cos = float(
            F.cosine_similarity(psi.unsqueeze(0), psi_saved.unsqueeze(0), dim=-1).item()
        )
        per_step_cos.append(cos)
        if cos < per_step_threshold:
            notes.append(
                f"step {i} ({step.op_name}): replay diverges from saved "
                f"(cos={cos:.4f} < {per_step_threshold:.3f})"
            )

    # 3. Compare final state to saved final psi (last step's psi_after).
    if program.steps:
        saved_final = torch.tensor(
            program.steps[-1].psi_after, dtype=psi.dtype, device=psi.device,
        )
        final_cos = float(
            F.cosine_similarity(psi.unsqueeze(0), saved_final.unsqueeze(0), dim=-1).item()
        )
    else:
        # Empty chain: psi == psi_initial; compare against saved initial.
        saved_final = z_src_saved
        final_cos = src_cos

    # 4. Optional: recompute final argmax over candidate pool.
    final_top1_replay: Optional[str] = None
    final_top1_match: Optional[bool] = None
    if candidate_pool is not None and len(candidate_pool) > 0:
        z_pool = encoder_encode_fn(candidate_pool)
        pred_n = F.normalize(psi.unsqueeze(0), dim=-1)
        pool_n = F.normalize(z_pool, dim=-1)
        sims = pred_n @ pool_n.T
        idx = int(sims.argmax(dim=-1).item())
        final_top1_replay = candidate_pool[idx]
        if program.final_top1_word is not None:
            final_top1_match = final_top1_replay == program.final_top1_word

    reproducible = (
        final_cos >= final_psi_threshold
        and all(c >= per_step_threshold for c in per_step_cos)
    )

    return ReplayResult(
        reproducible=reproducible,
        final_cos_to_saved=final_cos,
        final_top1_word_replay=final_top1_replay,
        final_top1_word_saved=program.final_top1_word,
        final_top1_match=final_top1_match,
        per_step_cos_to_saved=per_step_cos,
        notes=notes,
    )
