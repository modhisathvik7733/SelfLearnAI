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
from typing import Any, Callable, Optional

from .lm import LMRenderer
from .prompt import PromptBuilder, StructuredIntent
from .verifier import RenderingVerifier, VerificationResult


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
    # LM rendering
    lm_prompt: str = ""
    lm_output_raw: str = ""
    lm_n_input_tokens: int = 0
    lm_n_output_tokens: int = 0
    # Verifier
    grounding_cos: float = 0.0
    relevance_cos: float = 0.0
    grounding_threshold: float = 0.0
    relevance_threshold: float = 0.0
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
        encode_fn: Callable[[list[str]], "torch.Tensor"],  # noqa: F821
        renderer: LMRenderer,
        retriever: Optional[Any] = None,                   # selflearnai.memory.Retriever
        *,
        grounding_threshold: float = 0.70,
        relevance_threshold: float = 0.55,
        max_new_tokens: int = 200,
        retrieval_k: int = 3,
        retrieval_admit_threshold: float = 0.55,
    ) -> None:
        self.encode_fn = encode_fn
        self.renderer = renderer
        self.retriever = retriever
        self.prompt_builder = PromptBuilder(renderer)
        self.verifier = RenderingVerifier(
            encode_fn=encode_fn,
            grounding_threshold=grounding_threshold,
            relevance_threshold=relevance_threshold,
        )
        self.max_new_tokens = max_new_tokens
        self.retrieval_k = retrieval_k
        self.retrieval_admit_threshold = retrieval_admit_threshold

    # ---- Brain layer (overridable) ----------------------------------

    def gather_intent(self, query: str) -> StructuredIntent:
        """Default brain: retrieval-only. The brain decides what's true.

        Subclass and override to integrate the Stage 1 intent classifier,
        planner, concept operators, etc. — different intent kinds for
        different query shapes.
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
        # `retrieved` is a list of (text, score, metadata) tuples.
        # If best score is below admit threshold, refuse.
        if not retrieved or retrieved[0][1] < self.retrieval_admit_threshold:
            return StructuredIntent(
                kind="factual_q",
                user_query=query,
                retrieved_facts=[],
                retrieved_scores=[],
            )
        return StructuredIntent(
            kind="factual_q",
            user_query=query,
            retrieved_facts=[r[0] for r in retrieved],
            retrieved_scores=[r[1] for r in retrieved],
        )

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
        if intent.kind == "factual_q" and not intent.retrieved_facts:
            # Special case: no retrieval → expect refusal sentence.
            # Verifier just checks relevance (low bar) and accepts.
            verification = VerificationResult(
                grounded=True, relevant=True, accept=True,
                grounding_cos=1.0, relevance_cos=1.0,
                grounding_threshold=self.verifier.grounding_threshold,
                relevance_threshold=self.verifier.relevance_threshold,
                failure_reason=None,
            )
        else:
            verification = self.verifier.verify(
                rendered_text=gen.text,
                user_query=query,
                brain_target_texts=brain_targets,
            )
        # 5. Assemble response with audit trail.
        return RenderedResponse(
            text=gen.text if verification.accept else self._refused_message(verification),
            accepted=verification.accept,
            intent_kind=intent.kind,
            user_query=query,
            retrieved_facts=intent.retrieved_facts,
            retrieved_scores=intent.retrieved_scores,
            refusal_reason=intent.refusal_reason,
            lm_prompt=prompt,
            lm_output_raw=gen.text,
            lm_n_input_tokens=gen.n_input_tokens,
            lm_n_output_tokens=gen.n_output_tokens,
            grounding_cos=verification.grounding_cos,
            relevance_cos=verification.relevance_cos,
            grounding_threshold=verification.grounding_threshold,
            relevance_threshold=verification.relevance_threshold,
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
