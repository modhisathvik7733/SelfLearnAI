"""SelfLearnAI — real-scale concept-learning system.

Layer 1 only: meaning engine (concepts, grounding, operators in latent space).
Layer 2 (expression / generation) is explicitly out of scope.

See `/Users/chintu/.claude/plans/you-are-a-senior-jazzy-shannon.md` for the
system blueprint, design principles, and metric battery.
"""

# ---------------------------------------------------------------------------
# Quiet down HuggingFace transformers' chatty model-loading reports.
# Without this, every `from_pretrained` call dumps 25-line "UNEXPECTED keys"
# tables for the half of the checkpoint we deliberately don't load (e.g.,
# CLIPTextModelWithProjection ignoring the vision tower). They're noise.
#
# Set this BEFORE any transformers import — env var is the safest knob.
# ---------------------------------------------------------------------------
import os as _os
_os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")  # keep progress bars
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
    # Some transformers builds also have an explicit "load report" logger.
    import logging as _logging
    for _name in (
        "transformers",
        "transformers.modeling_utils",
        "transformers.configuration_utils",
    ):
        _logging.getLogger(_name).setLevel(_logging.ERROR)
except Exception:
    # If transformers isn't installed yet at import time, skip silently;
    # downstream imports will fail loudly with their own errors.
    pass


# Shared dim is the central architectural constant. Locked at 384 by review;
# do not change without rerunning all metric baselines.
SHARED_DIM = 384
