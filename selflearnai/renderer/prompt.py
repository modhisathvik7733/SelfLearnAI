"""Structured prompt builder for the LM renderer.

The brain produces a `StructuredIntent` — what the user asked + what
the brain decided is true (retrieved facts, operator outputs, etc.).
The PromptBuilder converts that into a prompt that strongly constrains
the LM to render ONLY what the brain said.

Three intent kinds supported initially:
  - factual_q     — answer using retrieved facts
  - transformation — render an operator's output
  - refusal_render — render a brain-decided refusal in fluent language

Each kind has its own prompt template optimized for groundedness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


IntentKind = Literal["factual_q", "transformation", "refusal_render", "raw"]


@dataclass
class StructuredIntent:
    """What the brain produces, before rendering.

    Fields are deliberately structured so the LM has no room to invent.
    """
    kind: IntentKind
    user_query: str
    # For factual_q:
    retrieved_facts: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    # For transformation:
    transformation_input: Optional[str] = None
    transformation_output: Optional[str] = None
    transformation_chain: list[str] = field(default_factory=list)
    # For refusal:
    refusal_reason: Optional[str] = None
    # For raw (a brain-derived ψ_target the LM must express literally):
    raw_target_text: Optional[str] = None
    # Always:
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptBuilder:
    """Convert StructuredIntent → constrained prompt for the LM.

    Each prompt is engineered to STRONGLY ground the LM:
      - explicit "use ONLY these facts" instruction
      - explicit refusal instruction when no facts
      - short answer length to discourage drift
      - chat-template formatting via the LM's tokenizer
    """

    SYSTEM_GROUNDED = (
        "You are a careful assistant. You answer questions using ONLY the facts provided. "
        "If the facts do not contain the answer, you say exactly: "
        "\"I don't have information about that.\" "
        "You never invent or extrapolate facts. "
        "You answer in 1-2 short sentences."
    )

    SYSTEM_TRANSFORMATION = (
        "You are a careful assistant that renders a structured transformation result "
        "as a natural sentence. You are given INPUT, OUTPUT, and CHAIN. "
        "Your job: write one short fluent sentence that expresses the transformation. "
        "You never add facts not in INPUT/OUTPUT. You never explain — just render."
    )

    SYSTEM_REFUSAL = (
        "You are a careful assistant. The system has decided to refuse the user's request "
        "for the given reason. Render the refusal as one short polite sentence. "
        "Do NOT attempt to answer the question; only acknowledge the refusal."
    )

    SYSTEM_RAW = (
        "You are a careful assistant. The brain has produced a target answer text. "
        "Output that target text as your response, possibly polished for fluency, "
        "but you may NOT add facts beyond the target. Keep it short."
    )

    def __init__(self, lm_renderer) -> None:
        """`lm_renderer` is needed for chat-template formatting."""
        self.lm = lm_renderer

    def build(self, intent: StructuredIntent) -> str:
        if intent.kind == "factual_q":
            return self._build_factual(intent)
        if intent.kind == "transformation":
            return self._build_transformation(intent)
        if intent.kind == "refusal_render":
            return self._build_refusal(intent)
        if intent.kind == "raw":
            return self._build_raw(intent)
        raise ValueError(f"unknown intent kind: {intent.kind!r}")

    def _build_factual(self, intent: StructuredIntent) -> str:
        if not intent.retrieved_facts:
            user = (
                f"QUESTION: {intent.user_query}\n\n"
                f"No facts were retrieved. "
                f"Answer with the exact refusal sentence above."
            )
            return self.lm.chat_format(self.SYSTEM_GROUNDED, user)
        facts_block = "\n".join(
            f"  {i+1}. {fact}"
            for i, fact in enumerate(intent.retrieved_facts)
        )
        user = (
            f"FACTS:\n{facts_block}\n\n"
            f"QUESTION: {intent.user_query}\n\n"
            f"ANSWER (1-2 short sentences, using only the facts above):"
        )
        return self.lm.chat_format(self.SYSTEM_GROUNDED, user)

    def _build_transformation(self, intent: StructuredIntent) -> str:
        chain_str = " → ".join(intent.transformation_chain) or "(direct)"
        user = (
            f"USER QUERY: {intent.user_query}\n"
            f"INPUT: {intent.transformation_input}\n"
            f"OUTPUT: {intent.transformation_output}\n"
            f"CHAIN: {chain_str}\n\n"
            f"Render the result as one short natural sentence:"
        )
        return self.lm.chat_format(self.SYSTEM_TRANSFORMATION, user)

    def _build_refusal(self, intent: StructuredIntent) -> str:
        user = (
            f"USER QUERY: {intent.user_query}\n"
            f"REFUSAL REASON: {intent.refusal_reason or 'unknown'}\n\n"
            f"Render the refusal as one short polite sentence:"
        )
        return self.lm.chat_format(self.SYSTEM_REFUSAL, user)

    def _build_raw(self, intent: StructuredIntent) -> str:
        user = (
            f"USER QUERY: {intent.user_query}\n"
            f"TARGET TEXT: {intent.raw_target_text}\n\n"
            f"Output the target text as your response (you may polish phrasing):"
        )
        return self.lm.chat_format(self.SYSTEM_RAW, user)
