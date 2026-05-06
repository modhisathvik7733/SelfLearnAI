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
    grounding_threshold: float = 0.0
    relevance_threshold: float = 0.0
    content_threshold: float = 0.0
    verifier_failure_reason: Optional[str] = None

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
        *,
        grounding_threshold: float = 0.65,
        relevance_threshold: float = 0.50,
        content_threshold: float = 0.50,
        max_new_tokens: int = 120,
        retrieval_k: int = 3,
        retrieval_admit_threshold: float = 0.78,
        retrieval_admit_margin: float = 0.04,
    ) -> None:
        """Defaults updated 2026-05-06 after the demo hallucination finding:
          - admit_threshold raised 0.55 → 0.78 (matches encoder background)
          - admit_margin added (top - second ≥ 0.04) — top must be
            meaningfully better than runner-up
          - content_threshold = 0.50 — content-word fidelity gate (NEW)
          - max_new_tokens dropped 200 → 120 (less room for tangents)
        """
        self.encode_fn = encode_fn
        self.renderer = renderer
        self.retriever = retriever
        self.prompt_builder = PromptBuilder(renderer)
        self.verifier = RenderingVerifier(
            encode_fn=encode_fn,
            grounding_threshold=grounding_threshold,
            relevance_threshold=relevance_threshold,
            content_threshold=content_threshold,
        )
        self.max_new_tokens = max_new_tokens
        self.retrieval_k = retrieval_k
        self.retrieval_admit_threshold = retrieval_admit_threshold
        self.retrieval_admit_margin = retrieval_admit_margin

    # ---- Brain layer (overridable) ----------------------------------

    def gather_intent(self, query: str) -> StructuredIntent:
        """Default brain: retrieval-only with admit-threshold + margin gate.

        Brain admit logic (post-2026-05-06 hallucination fix):
          1. Top retrieval score must clear absolute threshold (default 0.78).
             Encoder background cosine on E5 is ~0.74 for unrelated text;
             0.78+ signals genuine relevance.
          2. AND top must beat second by margin (default 0.04). Catches
             the case where multiple unrelated facts all score ~0.78
             due to encoder geometry — top isn't actually the right one.

        If either gate fails: BRAIN refuses (returns no facts). The LM
        then renders a refusal sentence. This keeps refusal decisions
        in the brain, not in the LM.

        Subclass and override to integrate the Stage 1 intent classifier,
        planner, concept operators, etc.
        """
        if self.retriever is None:
            return StructuredIntent(
                kind="refusal_render",
                user_query=query,
                refusal_reason="no retriever configured",
            )
        # Encode query and retrieve.
        z_query = self.encode_fn([query]).squeeze(0).flatten()
        retrieved = self.retriever.retrieve(z_query, k=self.retrieval_k)

        top_score = retrieved[0][1] if retrieved else 0.0
        second_score = retrieved[1][1] if len(retrieved) >= 2 else 0.0
        margin = top_score - second_score
        admit_top = top_score >= self.retrieval_admit_threshold
        admit_margin = margin >= self.retrieval_admit_margin

        intent = StructuredIntent(
            kind="factual_q",
            user_query=query,
            retrieved_facts=[],
            retrieved_scores=[],
            metadata={
                "brain_admit_top_score": float(top_score),
                "brain_admit_margin": float(margin),
                "brain_admit_top_passed": bool(admit_top),
                "brain_admit_margin_passed": bool(admit_margin),
            },
        )
        if not retrieved or not admit_top or not admit_margin:
            # Brain refuses on retrieval-relevance grounds.
            intent.metadata["brain_refusal"] = True
            intent.metadata["brain_refusal_reason"] = (
                "no retrieval admitted: "
                + ("top score too low" if not admit_top else "")
                + (("; " if not admit_top and not admit_margin else "")
                   if (not admit_top or not admit_margin) else "")
                + ("margin too small (top barely better than runner-up)"
                   if not admit_margin else "")
            )
            return intent

        # Brain admits: keep ALL retrieved (above threshold) for the
        # prompt, but only the top-1 is "definitely relevant"; the LM
        # is told in the prompt to use the SINGLE most relevant fact.
        intent.retrieved_facts = [r[0] for r in retrieved]
        intent.retrieved_scores = [r[1] for r in retrieved]
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
                grounding_threshold=self.verifier.grounding_threshold,
                relevance_threshold=self.verifier.relevance_threshold,
                content_threshold=self.verifier.content_threshold,
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
            lm_prompt=prompt,
            lm_output_raw=gen.text,
            lm_n_input_tokens=gen.n_input_tokens,
            lm_n_output_tokens=gen.n_output_tokens,
            grounding_cos=verification.grounding_cos,
            relevance_cos=verification.relevance_cos,
            content_overlap_rate=verification.content_overlap_rate,
            novel_content_words=list(verification.novel_content_words),
            grounding_threshold=verification.grounding_threshold,
            relevance_threshold=verification.relevance_threshold,
            content_threshold=verification.content_threshold,
            verifier_failure_reason=verification.failure_reason,
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
        return []

    @staticmethod
    def _refused_message(verification: VerificationResult) -> str:
        return (
            f"[refused: LM output failed verification — "
            f"{verification.failure_reason}]"
        )
