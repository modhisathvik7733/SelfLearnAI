"""Wake replay buffer (Task 1.5.1).

The wake phase records every Ψ-program produced by the live planner
into a tagged replay buffer that the sleep phase later mines for new
concept candidates and compound macros.

Per safeguard #2 (plan §19.9, memory feedback_stage1_5_safeguards),
"unexplained" is broader than failed chains — failed-only would miss
real discovery signals. Four channels:

  - `success`        — planner reached the goal with a verified chain.
                       Mined by sleep for compound-macro candidates
                       (frequent successful sub-chains).
  - `failed`         — planner couldn't reach the goal under its
                       current operator library. The strongest
                       discovery signal.
  - `inefficient`    — planner reached the goal but at a depth higher
                       than the minimum that worked for that
                       (source, target) pair. Suggests a missing
                       compound macro.
  - `low_confidence` — planner reached the goal but conformal coverage
                       was wider than threshold (chain produced an
                       answer the system isn't sure of). Suggests an
                       operator with under-fit calibration, or a
                       missing concept that would shorten the chain.

Each entry on disk is one JSON file under
  <root_dir>/<channel>/<source>__<target_or_none>__<timestamp>.json

so the buffer can be read concurrently by reader/writer processes
without locking. Files are append-only; sleep cycles read snapshots.

Design choices:
  - Wraps `PsiProgram` (the existing serialization from Task 1.11) plus
    a small envelope ({channel, routing_reason, routing_metadata,
    appended_at}). No parallel data structure.
  - Pure stdlib + the existing PsiProgram; no torch dependency in
    this module (all tensor handling happens at the planner edge).
  - Channel routing is a pure function (`route_program`) that takes
    the planner's run-time signals and decides the channel. The
    buffer just records the decision.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from selflearnai.planner import PsiProgram


Channel = Literal["success", "failed", "inefficient", "low_confidence"]
CHANNELS: tuple[Channel, ...] = ("success", "failed", "inefficient", "low_confidence")


# ---------------------------------------------------------------------------
# Buffer entry (PsiProgram + routing envelope)
# ---------------------------------------------------------------------------

@dataclass
class BufferEntry:
    """One wake-buffer record.

    `program` is the full Ψ-program the planner produced (including
    every intermediate ψ + per-step verification). `channel`,
    `routing_reason`, and `routing_metadata` come from `route_program`
    and explain why this entry landed on this channel — useful for
    sleep's mining + for post-hoc audit.
    """
    program: PsiProgram
    channel: Channel
    routing_reason: str
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    appended_at: str = field(default_factory=lambda: _now_iso())
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "channel": self.channel,
            "routing_reason": self.routing_reason,
            "routing_metadata": self.routing_metadata,
            "appended_at": self.appended_at,
            "program": self.program.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BufferEntry":
        return cls(
            program=PsiProgram.from_dict(d["program"]),
            channel=d["channel"],
            routing_reason=d["routing_reason"],
            routing_metadata=dict(d.get("routing_metadata", {})),
            appended_at=d.get("appended_at", _now_iso()),
            entry_id=d.get("entry_id", uuid.uuid4().hex[:12]),
        )


# ---------------------------------------------------------------------------
# Channel routing
# ---------------------------------------------------------------------------

def route_program(
    program: PsiProgram,
    *,
    reached_goal: bool,
    min_known_depth: Optional[int] = None,
    coverage_width: Optional[float] = None,
    coverage_width_threshold: float = 0.20,
) -> tuple[Channel, str, dict[str, Any]]:
    """Decide which wake channel an outcome belongs on.

    Inputs are the run-time signals the planner already has at the end
    of a solve attempt:
      - `reached_goal`: did the chain terminate at the goal embedding
        (top-1 word match, or planner verdict)?
      - `min_known_depth`: the shortest chain that has previously
        worked for this (source, target). If the current solve used a
        longer chain, the entry is "inefficient" — a candidate compound
        macro is hiding in the difference.
      - `coverage_width`: conformal coverage width on the final answer
        (e.g. 1 - max(softmax) over the candidate pool, or the
        ConformalGate's prediction-set width). If the answer is
        correct but the system is unsure of it, that's
        `low_confidence`.
      - `coverage_width_threshold`: above this width the outcome is
        flagged low-confidence.

    Routing precedence (first match wins, deliberately):
      1. NOT reached_goal               → "failed"
      2. min_known_depth defined AND
         len(chain) > min_known_depth   → "inefficient"
      3. coverage_width >= threshold    → "low_confidence"
      4. otherwise                      → "success"

    Returns (channel, routing_reason, routing_metadata).
    """
    chain_len = len(program.chain)
    meta: dict[str, Any] = {"chain_len": chain_len}
    if not reached_goal:
        meta["reached_goal"] = False
        return (
            "failed",
            "planner did not reach goal under current operator library",
            meta,
        )
    meta["reached_goal"] = True
    if min_known_depth is not None and chain_len > min_known_depth:
        meta["min_known_depth"] = min_known_depth
        meta["depth_overshoot"] = chain_len - min_known_depth
        return (
            "inefficient",
            (
                f"chain length {chain_len} > min known depth {min_known_depth} "
                f"(missing compound macro?)"
            ),
            meta,
        )
    if coverage_width is not None and coverage_width >= coverage_width_threshold:
        meta["coverage_width"] = coverage_width
        meta["coverage_width_threshold"] = coverage_width_threshold
        return (
            "low_confidence",
            (
                f"goal reached but coverage width {coverage_width:.3f} "
                f">= threshold {coverage_width_threshold:.3f}"
            ),
            meta,
        )
    if coverage_width is not None:
        meta["coverage_width"] = coverage_width
    return ("success", "goal reached and confidence within threshold", meta)


# ---------------------------------------------------------------------------
# On-disk buffer
# ---------------------------------------------------------------------------

class WakeBuffer:
    """Append-only on-disk replay buffer with one subdirectory per channel.

    Layout:
      <root>/
        success/        <entry_id>.json
        failed/         <entry_id>.json
        inefficient/    <entry_id>.json
        low_confidence/ <entry_id>.json

    Each <entry_id>.json contains the full `BufferEntry.to_dict()`.
    Filenames are entry_ids (12-hex-char uuid prefix), so writes are
    collision-free and lexically sortable by insertion order is not
    guaranteed — readers should sort by `appended_at` if they care.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for ch in CHANNELS:
            (self.root / ch).mkdir(parents=True, exist_ok=True)

    # ---- Write -------------------------------------------------------

    def append(
        self,
        program: PsiProgram,
        *,
        channel: Channel,
        routing_reason: str,
        routing_metadata: Optional[dict[str, Any]] = None,
    ) -> BufferEntry:
        """Persist one entry to its channel subdirectory."""
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")
        entry = BufferEntry(
            program=program,
            channel=channel,
            routing_reason=routing_reason,
            routing_metadata=dict(routing_metadata or {}),
        )
        path = self.root / channel / f"{entry.entry_id}.json"
        with open(path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)
        return entry

    def append_routed(
        self,
        program: PsiProgram,
        *,
        reached_goal: bool,
        min_known_depth: Optional[int] = None,
        coverage_width: Optional[float] = None,
        coverage_width_threshold: float = 0.20,
    ) -> BufferEntry:
        """Convenience: route + append in one call."""
        channel, reason, meta = route_program(
            program,
            reached_goal=reached_goal,
            min_known_depth=min_known_depth,
            coverage_width=coverage_width,
            coverage_width_threshold=coverage_width_threshold,
        )
        return self.append(
            program,
            channel=channel,
            routing_reason=reason,
            routing_metadata=meta,
        )

    # ---- Read --------------------------------------------------------

    def read(self, channel: Optional[Channel] = None) -> list[BufferEntry]:
        """Read all entries from one channel, or all channels if None.

        Sorted by `appended_at` ascending for stable downstream order.
        """
        if channel is not None:
            channels = (channel,)
        else:
            channels = CHANNELS
        out: list[BufferEntry] = []
        for ch in channels:
            ch_dir = self.root / ch
            if not ch_dir.exists():
                continue
            for p in sorted(ch_dir.glob("*.json")):
                with open(p) as f:
                    out.append(BufferEntry.from_dict(json.load(f)))
        out.sort(key=lambda e: e.appended_at)
        return out

    def size(self, channel: Channel) -> int:
        ch_dir = self.root / channel
        if not ch_dir.exists():
            return 0
        return sum(1 for _ in ch_dir.glob("*.json"))

    def size_all(self) -> dict[Channel, int]:
        return {ch: self.size(ch) for ch in CHANNELS}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
