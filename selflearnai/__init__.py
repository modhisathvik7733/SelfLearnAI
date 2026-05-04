"""SelfLearnAI — real-scale concept-learning system.

Layer 1 only: meaning engine (concepts, grounding, operators in latent space).
Layer 2 (expression / generation) is explicitly out of scope.

See `/Users/chintu/.claude/plans/you-are-a-senior-jazzy-shannon.md` for the
system blueprint, design principles, and metric battery.
"""

# Shared dim is the central architectural constant. Locked at 384 by review;
# do not change without rerunning all metric baselines.
SHARED_DIM = 384
