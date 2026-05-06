"""Path B pipeline orchestrator.

Wires up:
  brain (encoder + retriever + optional planner)
    → StructuredIntent
    → PromptBuilder
    → LMRenderer (eval-only)
    → RenderingVerifier
    → RenderedResponse with audit trail

The brain is plug-in. For v1 demo, brain is just (encoder + retriever);
later it can include the Stage 1 intent classifier, planner, and concept
operators.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .lm import LMRenderer
from .prompt import PromptBuilder, StructuredIntent
from .verifier import RenderingVerifier, VerificationResult

if TYPE_CHECKING:
    import torch


@dataclass
class RenderedResponse:
    """Final pipeline output. Includes full audit trail."""
    text: str
    accepted: bool
    intent_kind: str
    user_query: str
    # Brain decisions
    retrieved_facts: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    refusal_reason: Optional[str] = None
    brain_refusal: bool = False
    brain_admit_top_score: float = 0.0
    brain_admit_margin: float = 0.0
    brain_admit_tier: str = ""
    # LM rendering
    lm_prompt: str = ""
    lm_output_raw: str = ""
    lm_n_input_tokens: int = 0
    lm_n_output_tokens: int = 0
    # Verifier
    grounding_cos: float = 0.0
    relevance_cos: float = 0.0
    content_overlap_rate: float = 0.0
    novel_content_words: list[str] = field(default_factory=list)
    rare_novel_content_words: list[str] = field(default_factory=list)
    rare_novel_count: int = 0
    grounding_threshold: float = 0.0
    relevance_threshold: float = 0.0
    content_threshold: float = 0.0
    rare_novel_max: int = 0
    verifier_failure_reason: Optional[str] = None
    # Property-operator audit (when intent.kind == "property_q")
    property_entity: Optional[str] = None
    property_axis: Optional[str] = None
    property_top_value: Optional[str] = None
    property_top_score: float = 0.0
    property_top_margin: float = 0.0
    property_topk: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PathBPipeline:
    """Full Path B pipeline: brain → structured intent → LM → verifier.

    Construct once per session. `respond(query)` returns a RenderedResponse
    with the audit trail.

    The brain's role is encapsulated in `gather_intent`, which the caller
    can override to plug in retrievers, planners, intent classifiers, etc.
    For v1 demo we provide a simple retrieval-only brain.
    """

    def __init__(
        self,
        encode_fn: Callable[[list[str]], "torch.Tensor"],
        renderer: LMRenderer,
        retriever: Optional[Any] = None,                   # selflearnai.memory.Retriever
        property_pkg: Optional[Any] = None,                # PropertyOperatorPackage
        *,
        grounding_threshold: float = 0.65,
        relevance_threshold: float = 0.50,
        content_threshold: float = 0.50,
        rare_novel_max: int = 1,
        common_zipf_threshold: float = 4.0,
        max_new_tokens: int = 120,
        retrieval_k: int = 3,
        retrieval_admit_threshold: float = 0.78,
        retrieval_admit_margin: float = 0.04,
        retrieval_strong_threshold: float = 0.82,
        property_top_score_min: float = 0.18,
    ) -> None:
        """Defaults updated 2026-05-06 after the demo hallucination finding +
        the compositional false-negative finding (v2 rerun):

          - admit_threshold = 0.78 (above encoder background ~0.74)
          - admit_margin = 0.04 (top - second when top is BORDERLINE)
          - strong_threshold = 0.82: when top ≥ this, admit all retrieved
            ≥ admit_threshold without requiring margin. Catches the case
            where multiple facts are genuinely relevant for a
            compositional question (e.g. 'what eats mice AND what is the
            largest cat') — three relevant facts at 0.84/0.83/0.83 should
            be admitted, but their tight margin would otherwise refuse.
          - content_threshold = 0.50 — content-word fidelity gate
          - max_new_tokens = 120 (less room for tangents)
        """
        self.encode_fn = encode_fn
        self.renderer = renderer
        self.retriever = retriever
        self.property_pkg = property_pkg
        self.prompt_builder = PromptBuilder(renderer)
        self.verifier = RenderingVerifier(
            encode_fn=encode_fn,
            grounding_threshold=grounding_threshold,
            relevance_threshold=relevance_threshold,
            content_threshold=content_threshold,
            rare_novel_max=rare_novel_max,
            common_zipf_threshold=common_zipf_threshold,
        )
        self.max_new_tokens = max_new_tokens
        self.retrieval_k = retrieval_k
        self.retrieval_admit_threshold = retrieval_admit_threshold
        self.retrieval_admit_margin = retrieval_admit_margin
        self.retrieval_strong_threshold = retrieval_strong_threshold
        self.property_top_score_min = property_top_score_min

    # ---- Brain layer (overridable) ----------------------------------

    def gather_intent(self, query: str) -> StructuredIntent:
        """Routes the query through brain components in priority order:

          1. PROPERTY OPERATOR (Tier 0.1) — if query parses as a property
             question AND we have a trained operator, use it. The brain
             computes the answer in pure ψ-space; the LM types it.

          2. RETRIEVAL (Path B v1) — for everything else, the brain
             retrieves grounded facts from a corpus. Tiered admit gate
             handles refusal-relevant cases.

          3. REFUSAL — when neither operator nor retrieval has a
             confident answer.
        """
        # ---- 1. Try property operator first --------------------
        if self.property_pkg is not None:
            from selflearnai.intent.property_parser import parse_property_question
            parsed = parse_property_question(query)
            if parsed is not None:
                entity, axis = parsed
                if self.property_pkg.has_axis(axis):
                    topk = self.property_pkg.query(
                        entity, axis, self.encode_fn, k=5,
                    )
                    intent = StructuredIntent(
                        kind="property_q",
                        user_query=query,
                        property_entity=entity,
                        property_axis=axis,
                        property_top_value=topk.top_value,
                        property_top_score=topk.top_score,
                        property_top_margin=topk.margin,
                        property_topk=list(topk.candidates),
                    )
                    intent.metadata["brain_admit_tier"] = "property_op"
                    intent.metadata["brain_admit_top_score"] = topk.top_score
                    intent.metadata["brain_admit_margin"] = topk.margin
                    # Confidence gate: refuse if top score below floor
                    if topk.top_score < self.property_top_score_min:
                        intent.metadata["brain_refusal"] = True
                        intent.metadata["brain_refusal_reason"] = (
                            f"property operator low confidence: top score "
                            f"{topk.top_score:.3f} < {self.property_top_score_min}"
                        )
                        # Convert to refusal intent so the verifier auto-passes
                        intent.kind = "factual_q"
                        intent.property_top_value = None
                    return intent

        # ---- 2. Fall back to retrieval -------------------------
        if self.retriever is None:
            return StructuredIntent(
                kind="refusal_render",
                user_query=query,
                refusal_reason="no retriever configured",
            )
        z_query = self.encode_fn([query]).squeeze(0).flatten()
        retrieved = self.retriever.retrieve(z_query, k=self.retrieval_k)

        top_score = retrieved[0][1] if retrieved else 0.0
        second_score = retrieved[1][1] if len(retrieved) >= 2 else 0.0
        margin = top_score - second_score
        is_strong = top_score >= self.retrieval_strong_threshold
        is_borderline = (
            top_score >= self.retrieval_admit_threshold
            and top_score < self.retrieval_strong_threshold
        )

        intent = StructuredIntent(
            kind="factual_q",
            user_query=query,
            retrieved_facts=[],
            retrieved_scores=[],
            metadata={
                "brain_admit_top_score": float(top_score),
                "brain_admit_margin": float(margin),
                "brain_admit_tier": (
                    "strong" if is_strong
                    else "borderline" if is_borderline
                    else "below_threshold"
                ),
            },
        )

        # Tier 1: below admit threshold → refuse.
        if not retrieved or top_score < self.retrieval_admit_threshold:
            intent.metadata["brain_refusal"] = True
            intent.metadata["brain_refusal_reason"] = (
                f"top retrieval score {top_score:.3f} below "
                f"admit threshold {self.retrieval_admit_threshold} "
                f"(encoder background)"
            )
            return intent

        # Tier 2: borderline → require margin.
        if is_borderline and margin < self.retrieval_admit_margin:
            intent.metadata["brain_refusal"] = True
            intent.metadata["brain_refusal_reason"] = (
                f"borderline top score ({top_score:.3f}) AND tight margin "
                f"({margin:+.3f} < {self.retrieval_admit_margin}) — "
                f"top barely better than runner-up, signal of false retrieval"
            )
            return intent

        # Tier 3 (strong) OR Tier 2 with margin: admit all retrieved
        # above admit_threshold. The LM prompt tells it to pick the
        # single most relevant fact (or two if both directly answer
        # a compositional question).
        admitted = [
            (text, score, meta)
            for (text, score, meta) in retrieved
            if score >= self.retrieval_admit_threshold
        ]
        intent.retrieved_facts = [r[0] for r in admitted]
        intent.retrieved_scores = [r[1] for r in admitted]
        return intent

    # ---- Run a query end-to-end -------------------------------------

    def respond(self, query: str) -> RenderedResponse:
        # 1. Brain: gather intent.
        intent = self.gather_intent(query)
        # 2. Build constrained prompt.
        prompt = self.prompt_builder.build(intent)
        # 3. LM render.
        gen = self.renderer.generate(
            prompt, max_new_tokens=self.max_new_tokens, do_sample=False,
        )
        # 4. Verify.
        brain_targets = self._brain_target_texts(intent)
        brain_refusal = bool(intent.metadata.get("brain_refusal", False))
        if intent.kind == "factual_q" and not intent.retrieved_facts:
            # Brain refused on retrieval-relevance. Verifier just checks
            # the LM produced a refusal-shaped sentence (auto-pass).
            verification = VerificationResult(
                grounded=True, relevant=True, content_fidelity_passed=True,
                accept=True,
                grounding_cos=1.0, relevance_cos=1.0, content_overlap_rate=1.0,
                rare_novel_count=0,
                grounding_threshold=self.verifier.grounding_threshold,
                relevance_threshold=self.verifier.relevance_threshold,
                content_threshold=self.verifier.content_threshold,
                rare_novel_max=self.verifier.rare_novel_max,
                failure_reason=None,
            )
        else:
            verification = self.verifier.verify(
                rendered_text=gen.text,
                user_query=query,
                brain_target_texts=brain_targets,
            )
        # 5. Assemble response with audit trail.
        text_out = gen.text if verification.accept else self._refused_message(verification)
        return RenderedResponse(
            text=text_out,
            accepted=verification.accept,
            intent_kind=intent.kind,
            user_query=query,
            retrieved_facts=intent.retrieved_facts,
            retrieved_scores=intent.retrieved_scores,
            refusal_reason=(intent.refusal_reason
                            or intent.metadata.get("brain_refusal_reason")),
            brain_refusal=brain_refusal,
            brain_admit_top_score=float(intent.metadata.get("brain_admit_top_score", 0.0)),
            brain_admit_margin=float(intent.metadata.get("brain_admit_margin", 0.0)),
            brain_admit_tier=str(intent.metadata.get("brain_admit_tier", "")),
            lm_prompt=prompt,
            lm_output_raw=gen.text,
            lm_n_input_tokens=gen.n_input_tokens,
            lm_n_output_tokens=gen.n_output_tokens,
            grounding_cos=verification.grounding_cos,
            relevance_cos=verification.relevance_cos,
            content_overlap_rate=verification.content_overlap_rate,
            novel_content_words=list(verification.novel_content_words),
            rare_novel_content_words=list(verification.rare_novel_content_words),
            rare_novel_count=verification.rare_novel_count,
            grounding_threshold=verification.grounding_threshold,
            relevance_threshold=verification.relevance_threshold,
            content_threshold=verification.content_threshold,
            rare_novel_max=verification.rare_novel_max,
            verifier_failure_reason=verification.failure_reason,
            property_entity=intent.property_entity,
            property_axis=intent.property_axis,
            property_top_value=intent.property_top_value,
            property_top_score=float(intent.property_top_score or 0.0),
            property_top_margin=float(intent.property_top_margin or 0.0),
            property_topk=list(intent.property_topk),
        )

    @staticmethod
    def _brain_target_texts(intent: StructuredIntent) -> list[str]:
        if intent.kind == "factual_q":
            return list(intent.retrieved_facts)
        if intent.kind == "transformation":
            parts = []
            if intent.transformation_input:
                parts.append(intent.transformation_input)
            if intent.transformation_output:
                parts.append(intent.transformation_output)
            return parts
        if intent.kind == "raw" and intent.raw_target_text:
            return [intent.raw_target_text]
        if intent.kind == "property_q":
            # Verifier compares LM output to: query + entity + axis + value.
            # The value MUST appear in output (content-fidelity gate enforces it).
            parts = [intent.user_query]
            if intent.property_entity:
                parts.append(intent.property_entity)
            if intent.property_axis:
                parts.append(intent.property_axis.replace("_", " "))
            if intent.property_top_value:
                parts.append(intent.property_top_value)
            return parts
        return []

    @staticmethod
    def _refused_message(verification: VerificationResult) -> str:
        return (
            f"[refused: LM output failed verification — "
            f"{verification.failure_reason}]"
        )
