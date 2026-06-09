"""Orchestrator + Sub-agents Pattern.

A central orchestrator dispatches the incoming task to N specialist
sub-agents simultaneously. Each sub-agent inspects one dimension of
the task and returns a typed SubAgentFinding. The orchestrator then
aggregates all findings into a single Signal and runs the shared
Policy → Plan → Verify pipeline.

Pattern flow::

    Task
      -> OrchestratorWeaver
           ├── PIISubAgent          -> SubAgentFinding
           ├── SafetySubAgent       -> SubAgentFinding
           ├── ExternalSendSubAgent -> SubAgentFinding
           └── ToneSubAgent         -> SubAgentFinding
      -> Aggregate findings -> Signal
      -> Policy -> Plan -> Verify -> Outcome

Use case: guardrails (PII, safety, external-send, hostile-tone checks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..generic_agents import BasicVerifierAgent, RulePolicyAgent, TemplatePlannerAgent
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import TinyRetriever


@dataclass(frozen=True)
class SubAgentFinding:
    """Typed result from a single sub-agent."""

    name: str
    triggered: bool
    reason: str
    risk_delta: int = 0


@runtime_checkable
class SubAgent(Protocol):
    """Protocol that any sub-agent must satisfy."""

    name: str

    def run(self, task: Task) -> SubAgentFinding:
        ...


@dataclass(frozen=True)
class PIISubAgent:
    """Detects personally identifiable information."""

    name: str = "pii_detector"
    keywords: tuple[str, ...] = (
        "ssn",
        "credit card",
        "passport",
        "phone number",
        "date of birth",
        "bank account",
        "social security",
    )

    def run(self, task: Task) -> SubAgentFinding:
        text = task.text.lower()
        matches = [k for k in self.keywords if k in text]
        triggered = bool(matches)
        return SubAgentFinding(
            name=self.name,
            triggered=triggered,
            reason=f"PII detected: {', '.join(matches)}" if matches else "No PII detected.",
            risk_delta=2 if triggered else 0,
        )


@dataclass(frozen=True)
class SafetySubAgent:
    """Detects unsafe prompt patterns and jailbreak attempts."""

    name: str = "safety_checker"
    keywords: tuple[str, ...] = (
        "ignore previous",
        "jailbreak",
        "bypass policy",
        "ignore instructions",
        "act as",
        "pretend you are",
        "disregard",
    )

    def run(self, task: Task) -> SubAgentFinding:
        text = task.text.lower()
        matches = [k for k in self.keywords if k in text]
        triggered = bool(matches)
        return SubAgentFinding(
            name=self.name,
            triggered=triggered,
            reason=f"Unsafe prompt pattern detected: {', '.join(matches)}" if matches else "No unsafe patterns.",
            risk_delta=3 if triggered else 0,
        )


@dataclass(frozen=True)
class ExternalSendSubAgent:
    """Detects attempts to send data outside the system."""

    name: str = "external_send_detector"
    keywords: tuple[str, ...] = (
        "email customer",
        "post publicly",
        "send outside",
        "forward to",
        "share with",
        "send to external",
    )

    def run(self, task: Task) -> SubAgentFinding:
        text = task.text.lower()
        matches = [k for k in self.keywords if k in text]
        triggered = bool(matches)
        return SubAgentFinding(
            name=self.name,
            triggered=triggered,
            reason=f"External data send attempt: {', '.join(matches)}" if matches else "No external send detected.",
            risk_delta=2 if triggered else 0,
        )


@dataclass(frozen=True)
class ToneSubAgent:
    """Detects hostile or threatening language."""

    name: str = "tone_guard"
    keywords: tuple[str, ...] = (
        "threat",
        "sue",
        "lawyer",
        "hate",
        "destroy",
        "attack",
        "burn",
    )

    def run(self, task: Task) -> SubAgentFinding:
        text = task.text.lower()
        matches = [k for k in self.keywords if k in text]
        triggered = bool(matches)
        return SubAgentFinding(
            name=self.name,
            triggered=triggered,
            reason=f"Hostile tone detected: {', '.join(matches)}" if matches else "Tone acceptable.",
            risk_delta=1 if triggered else 0,
        )


@dataclass
class OrchestratorWeaver:
    """Orchestrator + Sub-agents pattern.

    A central orchestrator fans out to N specialist sub-agents,
    collects typed SubAgentFindings, aggregates them into a Signal,
    then runs the shared Policy + Planner + Verifier pipeline.

    To add a new guardrail dimension, implement a sub-agent class with
    a `run(task) -> SubAgentFinding` method and include it in `sub_agents`.
    """

    sub_agents: tuple[SubAgent, ...]
    policy_agent: RulePolicyAgent
    planner_agent: TemplatePlannerAgent
    verifier_agent: BasicVerifierAgent
    retriever: TinyRetriever
    fallback_action: str = "escalate"
    fallback_response: str = "Guardrail verification failed; route to review."

    # Maps sub-agent name -> intent name for Signal construction
    _INTENT_MAP: dict[str, str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self._INTENT_MAP is None:
            self._INTENT_MAP = {
                "pii_detector": "pii",
                "safety_checker": "unsafe_prompt",
                "external_send_detector": "external_send",
                "tone_guard": "unsafe_prompt",
            }

    def run(self, task: Task) -> Outcome:
        audit: list[str] = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:orchestrator",
        ]

        # 1. Fan out to all sub-agents
        findings: list[SubAgentFinding] = []
        for agent in self.sub_agents:
            finding = agent.run(task)
            findings.append(finding)
            audit.append(
                f"sub_agent:{finding.name}:triggered={finding.triggered}"
                f":risk_delta={finding.risk_delta}:{finding.reason}"
            )

        # 2. Aggregate findings into a single Signal
        triggered = [f for f in findings if f.triggered]
        total_risk = min(3, 1 + sum(f.risk_delta for f in triggered))

        if triggered:
            intent = self._INTENT_MAP.get(triggered[0].name, "general")
            tags = tuple(f.name for f in triggered)
        else:
            intent = "safe"
            tags = ()

        signal = Signal(intent=intent, risk=total_risk, tags=tags)
        audit.append(
            f"signal:intent={intent}:risk={total_risk}"
            f":sub_agents_triggered={len(triggered)}/{len(self.sub_agents)}"
        )

        # 3. Retrieve evidence
        evidence = self.retriever.search(task.text)
        audit.append(f"retrieval:documents={len(evidence)}")

        # 4. Policy
        policy = self.policy_agent.run(task, signal)
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")

        # 5. Planner
        draft = self.planner_agent.run(task, signal, evidence, policy)
        audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")

        # 6. Verifier
        verification = self.verifier_agent.run(draft, policy, evidence)
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")

        # 7. Finalise
        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            note = f"Verifier blocked: {'; '.join(verification.findings)}"
            audit.append(f"fallback:{self.fallback_action}")

        return Outcome(
            task_id=task.id,
            use_case=task.use_case,
            signal=signal,
            evidence=evidence,
            policy=policy,
            draft=draft,
            verification=verification,
            final_action=final_action,
            response=response,
            internal_note=note,
            audit_trail=audit,
        )
