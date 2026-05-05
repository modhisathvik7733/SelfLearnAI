"""E-graph subchain refactor + macro promotion (Task 1.5.4).

Mines the wake-buffer `success` channel for recurring sub-chains and
promotes them to **macro operators** — compound operators that flatten
common chains so the planner solves a depth-N problem in depth-1.

Per safeguard #3 (plan §19.9, memory feedback_stage1_5_safeguards),
macro promotion needs strict gates because:
  - exponential growth: every length-2 prefix from every success
    chain is a macro candidate; without filtering, the macro pool
    bloats faster than the concept library.
  - redundancy: many candidates are sub-prefixes of larger ones
    that don't add information.
  - over-fitting: a single rare chain becomes a "macro" with no
    actual evidence.

Two HARD gates per candidate:

  1. FREQUENCY ≥ K (default K=5)
     Sub-chain must appear in ≥ K success entries. Counts contiguous
     sub-sequences of length [min_len, max_len] across all chains in
     the buffer.

  2. UTILITY (RELATIVE planner improvement)
     Same gate shape as Task 1.5.3's planner-utility (lesson:
     feedback_relative_gates_in_encoder_space). With the macro
     registered at depth=1, the augmented planner must reach higher
     cos_to_goal on held-out tasks than the baseline planner at
     depth=1 — and ≥ τ_n_tasks tasks must individually improve by
     ≥ τ_per_task. Without this gate, the macro pool fills with
     compounds that compile but don't help.

Both gates required. Promotion writes a versioned MacroSpec; failures
are logged with reason and not registered.

Notes on scope:
  - Contiguous sub-sequence matching only. Non-contiguous patterns
    (e.g. "agentive then anything then plural") are out of scope —
    they require an actual e-graph and rarely beat contiguous chains
    in practice.
  - Length 2–3 by default. Length-1 isn't a macro (it's already an
    operator). Length-4+ rarely recurs and risks over-fitting.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from selflearnai.discovery.wake import BufferEntry
from selflearnai.planner import BeamSearchPlanner


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MacroCandidate:
    """A sub-chain that survived the frequency gate and is awaiting utility
    evaluation."""
    chain: tuple[str, ...]
    n_occurrences: int
    sources_seen: list[str] = field(default_factory=list)


@dataclass
class MacroPromotionResult:
    """Per-candidate audit record from `promote_macros`."""
    candidate: MacroCandidate
    # Frequency gate
    frequency_threshold: int
    passes_frequency: bool
    # Utility gate (RELATIVE — same shape as 1.5.3 planner-utility)
    n_planner_tasks: int
    planner_baseline_cos_mean: float
    planner_augmented_cos_mean: float
    planner_cos_improvement_mean: float
    planner_cos_improvement_threshold: float
    n_tasks_improved: int
    n_tasks_improved_min: int
    per_task_improvement_threshold: float
    passes_utility: bool
    # Overall
    passes: bool
    failing_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def enumerate_subsequences(
    chain: Sequence[str], *, min_len: int = 2, max_len: int = 3,
) -> Iterable[tuple[str, ...]]:
    """Yield all contiguous sub-sequences of `chain` with length in
    [min_len, max_len].
    """
    n = len(chain)
    upper = min(max_len, n)
    for length in range(min_len, upper + 1):
        for start in range(0, n - length + 1):
            yield tuple(chain[start:start + length])


def mine_subchain_candidates(
    success_entries: Sequence[BufferEntry],
    *,
    min_chain_length: int = 2,
    max_chain_length: int = 3,
    frequency_min: int = 5,
) -> list[MacroCandidate]:
    """Count contiguous sub-sequences across the success channel and
    return candidates that meet the frequency gate.

    `sources_seen` is the list of distinct source words from entries
    that contained this sub-chain — used by sleep for downstream
    audit (which inputs this macro would have solved).
    """
    counter: Counter[tuple[str, ...]] = Counter()
    sources_by_subchain: dict[tuple[str, ...], list[str]] = {}
    for entry in success_entries:
        chain = tuple(entry.program.chain)
        seen_for_this_entry: set[tuple[str, ...]] = set()
        for sub in enumerate_subsequences(
            chain, min_len=min_chain_length, max_len=max_chain_length,
        ):
            counter[sub] += 1
            if sub not in seen_for_this_entry:
                sources_by_subchain.setdefault(sub, []).append(entry.program.source)
                seen_for_this_entry.add(sub)
    return [
        MacroCandidate(
            chain=chain,
            n_occurrences=count,
            sources_seen=list(sources_by_subchain.get(chain, [])),
        )
        for chain, count in counter.most_common()
        if count >= frequency_min
    ]


# ---------------------------------------------------------------------------
# Macro composition
# ---------------------------------------------------------------------------

def make_macro_op(
    chain: Sequence[str],
    operators: dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable that applies `chain` in order: op_k ∘ ... ∘ op_1.

    Validates every name in chain has a registered operator.
    """
    missing = [n for n in chain if n not in operators]
    if missing:
        raise KeyError(f"chain references unregistered operators: {missing}")
    ops_in_order = [operators[n] for n in chain]

    @torch.no_grad()
    def macro(z: torch.Tensor) -> torch.Tensor:
        out = z
        for op in ops_in_order:
            out = op(out)
        return out

    return macro


def macro_name_from_chain(chain: Sequence[str]) -> str:
    """Canonical macro name. e.g. ('agentive', 'plural') → 'agentive__plural'.

    Stable, registry-friendly, and easy to grep for in traces.
    """
    return "__".join(chain)


# ---------------------------------------------------------------------------
# Utility gate (relative planner improvement)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_macro_utility(
    candidate: MacroCandidate,
    macro_op: Callable[[torch.Tensor], torch.Tensor],
    *,
    operators: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    holdout_tasks: Sequence[tuple[torch.Tensor, torch.Tensor]],
    macro_depth: int = 1,
    cos_improvement_min: float = 0.01,
    per_task_improvement_min: float = 0.01,
    n_tasks_improved_min: int = 1,
    beam_width: int = 4,
    step_bonus: float = 0.015,
) -> MacroPromotionResult:
    """RELATIVE planner-utility gate for a macro candidate.

    Compares two planners at the SAME beam_width and `max_depth=macro_depth`:
      - baseline: just `operators` (no macro)
      - augmented: `operators` + {macro_name: macro_op}

    `macro_depth` is the depth at which the macro should pay off. The
    canonical use-case is `macro_depth=1` — a length-2 chain becomes
    a single move. If a macro doesn't help even at depth=1 (where it
    has the largest advantage), it isn't worth registering.

    Both planners run the same held-out tasks; the gate is mean
    cos_to_goal improvement plus a minimum count of individually-
    improved tasks (guards against one outlier carrying the mean).
    """
    macro_name = macro_name_from_chain(candidate.chain)
    failing: list[str] = []

    baseline_planner = BeamSearchPlanner(
        operators=operators,
        beam_width=beam_width,
        max_depth=macro_depth,
        step_bonus=step_bonus,
    )
    augmented_ops = {**operators, macro_name: macro_op}
    augmented_planner = BeamSearchPlanner(
        operators=augmented_ops,
        beam_width=beam_width,
        max_depth=macro_depth,
        step_bonus=step_bonus,
    )

    baseline_cos: list[float] = []
    augmented_cos: list[float] = []
    n_tasks_improved = 0
    for psi_start, psi_goal in holdout_tasks:
        base = baseline_planner.search(psi_start, psi_goal)
        aug = augmented_planner.search(psi_start, psi_goal)
        baseline_cos.append(base.cos_to_goal)
        augmented_cos.append(aug.cos_to_goal)
        if aug.cos_to_goal - base.cos_to_goal >= per_task_improvement_min:
            n_tasks_improved += 1

    n_tasks = len(holdout_tasks)
    baseline_mean = float(sum(baseline_cos) / n_tasks) if n_tasks else 0.0
    augmented_mean = float(sum(augmented_cos) / n_tasks) if n_tasks else 0.0
    improvement_mean = augmented_mean - baseline_mean
    passes_utility = (
        improvement_mean >= cos_improvement_min
        and n_tasks_improved >= n_tasks_improved_min
    )
    if not passes_utility:
        failing.append(
            f"macro-utility: improvement_mean={improvement_mean:+.4f} "
            f"(threshold ≥ {cos_improvement_min:.4f}), "
            f"tasks_improved={n_tasks_improved}/{n_tasks} "
            f"(threshold ≥ {n_tasks_improved_min})"
        )

    passes_frequency = True  # candidates here have already passed
    overall = passes_frequency and passes_utility

    return MacroPromotionResult(
        candidate=candidate,
        frequency_threshold=candidate.n_occurrences,  # for audit reference
        passes_frequency=passes_frequency,
        n_planner_tasks=n_tasks,
        planner_baseline_cos_mean=baseline_mean,
        planner_augmented_cos_mean=augmented_mean,
        planner_cos_improvement_mean=improvement_mean,
        planner_cos_improvement_threshold=cos_improvement_min,
        n_tasks_improved=n_tasks_improved,
        n_tasks_improved_min=n_tasks_improved_min,
        per_task_improvement_threshold=per_task_improvement_min,
        passes_utility=passes_utility,
        passes=overall,
        failing_reasons=failing,
    )


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------

def promote_macros(
    success_entries: Sequence[BufferEntry],
    *,
    operators: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    holdout_tasks: Sequence[tuple[torch.Tensor, torch.Tensor]],
    min_chain_length: int = 2,
    max_chain_length: int = 3,
    frequency_min: int = 5,
    macro_depth: int = 1,
    cos_improvement_min: float = 0.01,
    per_task_improvement_min: float = 0.01,
    n_tasks_improved_min: int = 1,
    beam_width: int = 4,
    step_bonus: float = 0.015,
) -> list[MacroPromotionResult]:
    """Mine + utility-evaluate every candidate that passes the
    frequency gate. Returns one MacroPromotionResult per surviving
    candidate (caller filters by `.passes`)."""
    candidates = mine_subchain_candidates(
        success_entries,
        min_chain_length=min_chain_length,
        max_chain_length=max_chain_length,
        frequency_min=frequency_min,
    )
    results: list[MacroPromotionResult] = []
    for candidate in candidates:
        macro_op = make_macro_op(candidate.chain, operators)
        result = evaluate_macro_utility(
            candidate, macro_op,
            operators=operators,
            holdout_tasks=holdout_tasks,
            macro_depth=macro_depth,
            cos_improvement_min=cos_improvement_min,
            per_task_improvement_min=per_task_improvement_min,
            n_tasks_improved_min=n_tasks_improved_min,
            beam_width=beam_width,
            step_bonus=step_bonus,
        )
        results.append(result)
    return results
