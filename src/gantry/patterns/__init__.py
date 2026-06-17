"""Agentic Harness — Pattern Library.

Each pattern class implements the same interface:

    weaver = SomePatternWeaver(...)
    outcome = weaver.run(task)

Available patterns:
- PipelineWeaver              : Sequential claim-weaving chain (support, crm, research, coding)
- OrchestratorWeaver          : Orchestrator + N sub-agents fan-out (guardrail)
- ParallelOrchestratorWeaver  : Generic parallel fan-out with custom aggregation (fraud)
- RouterWeaver                : Classify → dispatch to specialist agent (it helpdesk)
- ReflectionWeaver            : Draft → Critic loop until approved (legal)
- HumanInTheLoopWeaver        : Pause at checkpoint, resume on approval (hr)
- PlanExecuteWeaver           : Multi-step plan, each step verified (finance)
"""

from .pipeline import PipelineWeaver
from .orchestrator import OrchestratorWeaver
from .parallel_orchestrator import ParallelOrchestratorWeaver
from .router import RouterWeaver, SpecialistAgent
from .reflection import ReflectionWeaver, CriticAgent, Critique
from .hitl import HumanInTheLoopWeaver, CHECKPOINT_ACTION
from .plan_execute import PlanExecuteWeaver, ExecutionStep

__all__ = [
    "PipelineWeaver",
    "OrchestratorWeaver",
    "ParallelOrchestratorWeaver",
    "RouterWeaver",
    "SpecialistAgent",
    "ReflectionWeaver",
    "CriticAgent",
    "Critique",
    "HumanInTheLoopWeaver",
    "CHECKPOINT_ACTION",
    "PlanExecuteWeaver",
    "ExecutionStep",
]
