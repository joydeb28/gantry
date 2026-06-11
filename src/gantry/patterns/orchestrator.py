"""Orchestrator + Sub-agents Pattern — LangGraph StateGraph implementation.

A central orchestrator dispatches the incoming task to N specialist sub-agents
in parallel using the LangGraph Send API. Findings are aggregated, and then the
shared Policy -> Plan -> Verify pipeline runs.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, Protocol, TypedDict, runtime_checkable
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from ..generic_agents import BasicVerifierAgent, RulePolicyAgent, TemplatePlannerAgent
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification, SubAgentFinding
from ..retrieval import KnowledgeBaseRetriever


@runtime_checkable
class SubAgent(Protocol):
    """Protocol that every sub-agent must satisfy."""

    name: str

    def run(self, task: Task) -> SubAgentFinding:
        ...


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


class SubAgentState(TypedDict):
    task: Task
    agent_idx: int


class OrchestratorState(TypedDict):
    task: Task
    findings: Annotated[list[SubAgentFinding], operator.add]
    signal: Optional[Signal]
    evidence: tuple[Evidence, ...]
    policy: Optional[PolicyDecision]
    draft: Optional[Plan]
    verification: Optional[Verification]
    outcome: Optional[Outcome]
    audit_trail: list[str]


class OrchestratorWeaver:
    """Orchestrator + Sub-agents pattern using LangGraph StateGraph."""

    def __init__(
        self,
        sub_agents: tuple[SubAgent, ...],
        policy_agent: RulePolicyAgent,
        planner_agent: TemplatePlannerAgent,
        verifier_agent: BasicVerifierAgent,
        retriever: KnowledgeBaseRetriever,
        fallback_action: str = "escalate",
        fallback_response: str = "Guardrail verification failed; route to review.",
    ) -> None:
        self.sub_agents = sub_agents
        self.policy_agent = policy_agent
        self.planner_agent = planner_agent
        self.verifier_agent = verifier_agent
        self.retriever = retriever
        self.fallback_action = fallback_action
        self.fallback_response = fallback_response

        self._INTENT_MAP = {
            "pii_detector": "pii",
            "safety_checker": "unsafe_prompt",
            "external_send_detector": "external_send",
            "tone_guard": "unsafe_prompt",
        }

        builder = StateGraph(OrchestratorState)
        builder.add_node("run_sub_agent", self._run_sub_agent_node)
        builder.add_node("aggregate", self._aggregate_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("policy", self._policy_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("verify", self._verify_node)
        builder.add_node("finalize", self._finalize_node)

        # Start by dispatching sub-agents in parallel
        builder.add_conditional_edges(START, self._dispatch_sub_agents, ["run_sub_agent"])
        builder.add_edge("run_sub_agent", "aggregate")
        builder.add_edge("aggregate", "retrieve")
        builder.add_edge("retrieve", "policy")
        builder.add_edge("policy", "plan")
        builder.add_edge("plan", "verify")
        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)

        self.graph = builder.compile()

    def _dispatch_sub_agents(self, state: OrchestratorState) -> list[Send]:
        return [
            Send("run_sub_agent", {"task": state["task"], "agent_idx": i})
            for i in range(len(self.sub_agents))
        ]

    def _run_sub_agent_node(self, state: SubAgentState) -> dict:
        agent = self.sub_agents[state["agent_idx"]]
        finding = agent.run(state["task"])
        return {"findings": [finding]}

    def _aggregate_node(self, state: OrchestratorState) -> dict:
        findings = state["findings"]
        triggered = [f for f in findings if f.triggered]
        total_risk = min(3, 1 + sum(f.risk_delta for f in triggered))

        if triggered:
            intent = self._INTENT_MAP.get(triggered[0].name, "general")
            tags = tuple(f.name for f in triggered)
        else:
            intent = "safe"
            tags = ()

        signal = Signal(intent=intent, risk=total_risk, tags=tags)
        audit = list(state.get("audit_trail") or [])

        # Match exact old audit formatting
        for f in findings:
            audit.append(
                f"sub_agent:{f.name}:triggered={f.triggered}"
                f":risk_delta={f.risk_delta}:{f.reason}"
            )
        audit.append(
            f"signal:intent={intent}:risk={total_risk}"
            f":sub_agents_triggered={len(triggered)}/{len(self.sub_agents)}"
        )
        return {"signal": signal, "audit_trail": audit}

    def _retrieve_node(self, state: OrchestratorState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"retrieval:documents={len(evidence)}")
        return {"evidence": evidence, "audit_trail": audit}

    def _policy_node(self, state: OrchestratorState) -> dict:
        task = state["task"]
        signal = state["signal"]
        policy = self.policy_agent.run(task, signal)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")
        return {"policy": policy, "audit_trail": audit}

    def _plan_node(self, state: OrchestratorState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = self.planner_agent.run(task, signal, evidence, policy)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")
        return {"draft": draft, "audit_trail": audit}

    def _verify_node(self, state: OrchestratorState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.verifier_agent.run(draft, policy, evidence)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")
        return {"verification": verification, "audit_trail": audit}

    def _finalize_node(self, state: OrchestratorState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = state["draft"]
        verification = state["verification"]
        audit = list(state.get("audit_trail") or [])

        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            note = f"Verifier blocked: {'; '.join(verification.findings)}"
            audit.append(f"fallback:{self.fallback_action}")

        outcome = Outcome(
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
        return {"outcome": outcome}

    def run(self, task: Task) -> Outcome:
        initial_audit = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:orchestrator",
        ]
        result = self.graph.invoke({"task": task, "audit_trail": initial_audit, "findings": []})
        return result["outcome"]
