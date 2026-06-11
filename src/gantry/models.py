"""Agentic Harness — Pydantic v2 data models.

All data that flows through the system is typed, validated, and immutable.
Pydantic v2 provides:
- Runtime validation with clear error messages
- model_dump() / model_dump_json() for serialisation
- JSON schema generation (used by LangChain with_structured_output)
- Field-level constraints (ge/le on confidence, etc.)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Task(BaseModel):
    """The input to any agent run. Loaded from a task JSON file."""

    model_config = ConfigDict(frozen=True)

    id: str
    use_case: str
    title: str
    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Combined text used for retrieval and signal extraction."""
        return f"{self.title}\n{self.body}"


class Signal(BaseModel):
    """Detected intent, risk level, and tags extracted from the task."""

    model_config = ConfigDict(frozen=True)

    intent: str
    risk: int = Field(ge=0, le=3)
    missing_fields: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


class Evidence(BaseModel):
    """A single retrieved knowledge-base document with a relevance score."""

    model_config = ConfigDict(frozen=True)

    source: str
    title: str
    text: str
    score: float = Field(default=0.0, ge=0.0)


class PolicyDecision(BaseModel):
    """Which actions are permitted or blocked for this signal."""

    model_config = ConfigDict(frozen=True)

    allowed_actions: tuple[str, ...]
    blocked_actions: tuple[str, ...] = ()
    reason: str = ""


class Plan(BaseModel):
    """The proposed action and customer-facing response from the planner.

    Used directly as the structured output schema for LangChain
    ``with_structured_output(Plan)``.
    """

    model_config = ConfigDict(frozen=True)

    action: str = Field(description="The action to take (must be in allowed_actions)")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score from 0.0 (none) to 1.0 (certain)",
    )
    response: str = Field(description="Customer-facing response text")
    internal_note: str = Field(default="", description="Internal reasoning / audit note")
    citations: tuple[str, ...] = Field(default=(), description="Source document titles cited")

    @model_validator(mode="after")
    def strip_whitespace(self) -> "Plan":
        # Pydantic frozen models are immutable; use object.__setattr__ for post-init cleanup
        object.__setattr__(self, "response", self.response.strip())
        object.__setattr__(self, "internal_note", self.internal_note.strip())
        return self


class Verification(BaseModel):
    """Gate result from the verifier agent."""

    model_config = ConfigDict(frozen=True)

    approved: bool
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    findings: tuple[str, ...] = ()


class Outcome(BaseModel):
    """Complete result of a weaver run — includes full audit trail."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    use_case: str
    signal: Signal
    evidence: tuple[Evidence, ...]
    policy: PolicyDecision
    draft: Plan
    verification: Verification
    final_action: str
    response: str
    internal_note: str = ""
    audit_trail: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sub-agent finding types (used by Orchestrator and Fraud patterns)
# ---------------------------------------------------------------------------

class SubAgentFinding(BaseModel):
    """Typed result from a guardrail sub-agent."""

    model_config = ConfigDict(frozen=True)

    name: str
    triggered: bool
    reason: str
    risk_delta: int = Field(default=0, ge=0)


class FraudFinding(BaseModel):
    """Typed result from a fraud-detection sub-agent."""

    model_config = ConfigDict(frozen=True)

    name: str
    triggered: bool
    fraud_type: str
    reason: str
    risk_score: int = Field(default=0, ge=0, le=3)
    recommended_action: str


# ---------------------------------------------------------------------------
# Agent Recipe
# ---------------------------------------------------------------------------

class AgentRecipe(BaseModel):
    """Configuration bundle defining the agents for a pipeline use case."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    use_case: str
    signal_agent: Any
    policy_agent: Any
    planner_agent: Any
    verifier_agent: Any
    fallback_action: str = "escalate"
    fallback_response: str = "This needs review before automation continues."

