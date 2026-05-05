"""Per-step verification gates for the planner (Task 1.10).

Wires up the architectural commitment from plan §6 ("per-step
verification gates"). Every step in a planner-produced chain runs
through a pipeline of independent gates. Each gate returns a
GateResult with a pass/fail boolean, the metric value, the threshold
used, and a human-readable reason.

This first commit (Task 1.10a) implements three gates that don't
require any new training:

  - TypeGate: operator signatures and chain-composition typing.
    Ensures `output_type(op_i) == input_type(op_{i+1})` along the
    chain. Catches nonsense compositions like `plural ∘ plural`
    starting from a Verb (plural's input type is Noun, not Verb).

  - AxiomGate: operator non-identity. Each operator should
    `cos(before, after) < threshold` — i.e. it must actually do
    something. A no-op operator that returns its input unchanged
    fails this check. Default threshold 0.99.

  - DriftGate: state stays on the encoder manifold. After applying
    an operator, the resulting embedding should still be close to
    at least one known reference word (max cos to reference > 0.5).
    Catches operators that produce off-manifold garbage outputs.

The fourth gate, ConformalGate (per-operator calibrated coverage),
arrives in Task 1.10b — it needs per-operator ConformalOperatorCalibrator
fitting which is plumbing-heavy enough to keep separate.

Failed gates do NOT raise. They surface as `passed=False` in the
returned ChainVerification. The planner / caller decides whether to
halt or continue based on which gate failed and how badly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """Outcome of a single gate check at a single step."""
    passed: bool
    metric: float
    threshold: float
    reason: str


@dataclass
class StepVerification:
    """All gate results for one step in a chain."""
    step_index: int
    op_name: str
    gate_results: dict[str, GateResult]
    all_passed: bool


@dataclass
class ChainVerification:
    """All step verifications for an entire chain."""
    chain: tuple[str, ...]
    steps: list[StepVerification]
    all_passed: bool

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def step_summary(self) -> str:
        """One-line per-step summary for printing."""
        return ", ".join(
            f"{s.op_name}({'✓' if s.all_passed else '✗'})"
            for s in self.steps
        )


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

class Gate:
    """Base class for verification gates."""
    name: str = "Gate"

    def check(
        self,
        step_index: int,
        op_name: str,
        psi_before: torch.Tensor,
        psi_after: torch.Tensor,
        context: dict,
    ) -> GateResult:
        raise NotImplementedError


class TypeGate(Gate):
    """Operator signature + chain composition typing.

    Each operator has a (input_type, output_type) signature. The first
    step's operator must have input_type matching the source's type;
    each subsequent operator must have input_type matching the previous
    operator's output_type. Type mismatches surface as `passed=False`.

    `source_type=None` skips the input check at step 0 (useful when the
    caller doesn't know the source type, e.g. ad-hoc planning).
    """
    name = "type"

    def __init__(
        self,
        signatures: dict[str, tuple[str, str]],
        source_type: Optional[str] = None,
    ):
        self.signatures = dict(signatures)
        self.source_type = source_type

    def check(self, step_index, op_name, psi_before, psi_after, context):
        if op_name not in self.signatures:
            return GateResult(
                passed=False,
                metric=0.0,
                threshold=1.0,
                reason=f"unknown operator {op_name!r}",
            )
        in_type, out_type = self.signatures[op_name]
        if step_index == 0:
            expected_in = self.source_type
        else:
            prev_op = context["chain"][step_index - 1]
            expected_in = (
                self.signatures[prev_op][1]
                if prev_op in self.signatures else None
            )
        passed = expected_in is None or in_type == expected_in
        return GateResult(
            passed=passed,
            metric=1.0 if passed else 0.0,
            threshold=1.0,
            reason=(
                f"in_type={in_type!r}, expected={expected_in!r}, "
                f"out_type={out_type!r}"
            ),
        )


class AxiomGate(Gate):
    """Operator non-identity: cos(before, after) must be below threshold.

    Catches operators that fail to transform their input (a degenerate
    no-op). Default threshold 0.99 — operators must move the embedding
    by at least a small margin in encoder space.
    """
    name = "axiom_non_identity"

    def __init__(self, threshold: float = 0.99):
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold

    def check(self, step_index, op_name, psi_before, psi_after, context):
        cos = float(
            F.cosine_similarity(
                psi_before.unsqueeze(0), psi_after.unsqueeze(0), dim=-1
            ).item()
        )
        passed = cos < self.threshold
        return GateResult(
            passed=passed,
            metric=cos,
            threshold=self.threshold,
            reason=(
                f"cos(before, after)={cos:+.4f}  "
                f"({'<' if passed else '>='} {self.threshold:.2f})"
            ),
        )


class DriftGate(Gate):
    """State stays on the encoder manifold.

    A reference set of "known good" embeddings is provided at construction
    time. After each operator, the gate computes
        max_v cos(ψ_after, reference_v)
    and passes iff this max is above `threshold`. Off-manifold embeddings
    (no nearby reference) fail the check.

    Reference embeddings should cover the domain of expected
    intermediates — e.g., a vocabulary drawn from each concept's
    training-pair source and target words.
    """
    name = "drift"

    def __init__(
        self,
        reference_embeddings: torch.Tensor,
        threshold: float = 0.5,
    ):
        if reference_embeddings.dim() != 2:
            raise ValueError(
                f"reference_embeddings must be 2-D (M, D), got "
                f"shape {tuple(reference_embeddings.shape)}"
            )
        if not -1.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (-1, 1), got {threshold}")
        self.reference_n = F.normalize(reference_embeddings, dim=-1)
        self.threshold = threshold

    def check(self, step_index, op_name, psi_before, psi_after, context):
        psi_n = F.normalize(psi_after.unsqueeze(0), dim=-1)
        sims = psi_n @ self.reference_n.T
        max_cos = float(sims.max().item())
        passed = max_cos > self.threshold
        return GateResult(
            passed=passed,
            metric=max_cos,
            threshold=self.threshold,
            reason=(
                f"max cos to reference = {max_cos:+.4f}  "
                f"({'>' if passed else '<='} {self.threshold:.2f})"
            ),
        )


# ---------------------------------------------------------------------------
# Top-level verifier
# ---------------------------------------------------------------------------

def verify_chain(
    chain: tuple[str, ...] | Sequence[str],
    operators: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    psi_start: torch.Tensor,
    gates: Sequence[Gate],
) -> ChainVerification:
    """Re-execute a chain step by step, running all gates at each step.

    `operators` is the planner's operator library (callable z → z). The
    chain is interpreted as a sequence of operator names from this
    library. Each op is applied in order; after each application all
    gates run on (psi_before, psi_after, context). The full per-step
    + chain-level result is returned.
    """
    steps: list[StepVerification] = []
    psi = psi_start.detach().flatten()
    chain_t = tuple(chain)
    for i, op_name in enumerate(chain_t):
        if op_name not in operators:
            raise ValueError(
                f"chain step {i}: unknown operator {op_name!r}; "
                f"library has {sorted(operators.keys())}"
            )
        op = operators[op_name]
        psi_before = psi
        psi_after = op(psi.unsqueeze(0)).squeeze(0)
        context = {"chain": chain_t, "step_index": i}
        gate_results: dict[str, GateResult] = {}
        for g in gates:
            gate_results[g.name] = g.check(
                i, op_name, psi_before, psi_after, context,
            )
        all_passed = all(r.passed for r in gate_results.values())
        steps.append(
            StepVerification(
                step_index=i,
                op_name=op_name,
                gate_results=gate_results,
                all_passed=all_passed,
            )
        )
        psi = psi_after
    chain_passed = all(s.all_passed for s in steps)
    return ChainVerification(
        chain=chain_t,
        steps=steps,
        all_passed=chain_passed,
    )


# ---------------------------------------------------------------------------
# Default signatures for the 7-concept library
# ---------------------------------------------------------------------------
#
# These are the type signatures we attach to each concept by default.
# They are not "linguistically perfect" but they are CONSISTENT with how
# the operators were trained, which is what type-checking actually
# needs:
#
#   agentive    Verb → Noun           (paint → painter)
#   plural      Noun → NounPlural     (cat   → cats)
#   past_tense  Verb → Verb           (run   → ran; still a Verb)
#   comparative Adj  → Adj            (big   → bigger; still an Adj)
#   superlative Adj  → Adj            (big   → biggest; still an Adj)
#   opposite    Word → Word           (warm  → cool; same domain)
#   young       Noun → Noun           (cat   → kitten; both nouns)
#
# Composition consequences (intentional):
#   - agentive ∘ plural   : Verb → Noun → NounPlural   ✓
#   - plural ∘ agentive   : Noun → NounPlural → ???   (agentive needs Verb)  ✗
#   - past_tense ∘ plural : Verb → Verb → ??? (plural needs Noun)            ✗
#
DEFAULT_TYPE_SIGNATURES: dict[str, tuple[str, str]] = {
    "agentive":    ("Verb", "Noun"),
    "plural":      ("Noun", "NounPlural"),
    "past_tense":  ("Verb", "Verb"),
    "comparative": ("Adj",  "Adj"),
    "superlative": ("Adj",  "Adj"),
    "opposite":    ("Word", "Word"),
    "young":       ("Noun", "Noun"),
}
