"""Reusable pattern-weaver harness for agentic AI systems."""

from .models import (
    AgentRecipe,
    Evidence,
    Outcome,
    Plan,
    PolicyDecision,
    PlannerAgentProtocol,
    PolicyAgentProtocol,
    Signal,
    SignalAgentProtocol,
    Task,
    Verification,
    VerifierAgentProtocol,
)
from .patterns.pipeline import PipelineWeaver

# Backward compatibility mapping
ClaimWeaver = PipelineWeaver

__all__ = [
    # Data models
    "AgentRecipe",
    "Evidence",
    "Outcome",
    "Plan",
    "PolicyDecision",
    "Signal",
    "Task",
    "Verification",
    # Agent role Protocols — implement these to type-annotate custom agents
    "SignalAgentProtocol",
    "PolicyAgentProtocol",
    "PlannerAgentProtocol",
    "VerifierAgentProtocol",
    # Weavers
    "ClaimWeaver",
    "PipelineWeaver",
]
