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

# Production harness — metrics, memory, orchestration, autonomy
from .memory import ConversationBuffer, InMemoryBackend
from .metrics import LoggingEmitter, MetricsEmitter, NoOpEmitter
from .orchestration import RetryOrchestrator
from .generic_agents import ClarificationAgent
from .retrieval import KBWatcher
from .triggers import ScheduledTrigger, PollingTrigger, WebhookTrigger, TriggerRunner
from .llm import DynamicStepPlanner

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
    # Production: Memory
    "ConversationBuffer",
    "InMemoryBackend",
    # Production: Metrics
    "MetricsEmitter",
    "NoOpEmitter",
    "LoggingEmitter",
    # Production: Orchestration
    "RetryOrchestrator",
    # Production: Agents
    "ClarificationAgent",
    # Production: Retrieval
    "KBWatcher",
    # Production: Autonomy / Triggers
    "ScheduledTrigger",
    "PollingTrigger",
    "WebhookTrigger",
    "TriggerRunner",
    # Production: Dynamic planning
    "DynamicStepPlanner",
]
