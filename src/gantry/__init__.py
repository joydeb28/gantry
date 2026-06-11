"""Reusable pattern-weaver harness for agentic AI systems."""

from .models import Outcome, Plan, PolicyDecision, Signal, Task, Verification, AgentRecipe
from .patterns.pipeline import PipelineWeaver

# Backward compatibility mapping
ClaimWeaver = PipelineWeaver

__all__ = [
    "AgentRecipe",
    "ClaimWeaver",
    "PipelineWeaver",
    "Outcome",
    "Plan",
    "PolicyDecision",
    "Signal",
    "Task",
    "Verification",
]
