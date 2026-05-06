"""selflearnai.renderer — Path B surface-rendering layer.

After the project rule revision (2026-05-06, see memory
`feedback_no_llm_in_architecture`), small LMs are permitted at the
FINAL surface-rendering layer ONLY. The brain (ψ-space + concept
operators + planner + retriever + intent classifier + conformal
calibration) makes ALL decisions; the LM's job is fluency only.

Pipeline shape:

    user query
       │
       ▼
    encoder (E5)               ← brain
       │
       ▼
    intent / planner / retriever  ← brain decides what's true
       │
       ▼
    StructuredIntent
       (query, intent_kind, retrieved_facts, operator_output, ...)
       │
       ▼
    LMRenderer  (constrained prompt → fluent text)
       │
       ▼
    RenderingVerifier
       (cos(encode(rendered), ψ_intent) >= τ; refuse if drift)
       │
       ▼
    RenderedResponse(text, verified, audit_trail)

Public modules:
  - lm:        LMRenderer (HuggingFace small-LM wrapper, eval-only)
  - prompt:    PromptBuilder (structured-input → constrained prompt)
  - verifier:  RenderingVerifier (conformal-style output gate)
  - pipeline:  PathBPipeline (orchestrator)

Strict invariants:
  - LM is loaded in eval mode; NEVER trained/fine-tuned in this repo.
  - LM never decides facts; the prompt is structured to force grounded
    rendering only ("answer using ONLY the provided facts").
  - Every response carries an audit trail showing the brain's decisions.
  - Verifier gate refuses when LM output drifts from brain intent.
"""
from .lm import LMRenderer
from .prompt import PromptBuilder, StructuredIntent
from .verifier import RenderingVerifier, VerificationResult
from .pipeline import PathBPipeline, RenderedResponse

__all__ = [
    "LMRenderer",
    "PromptBuilder",
    "StructuredIntent",
    "RenderingVerifier",
    "VerificationResult",
    "PathBPipeline",
    "RenderedResponse",
]
