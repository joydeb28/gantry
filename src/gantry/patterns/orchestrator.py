"""Orchestrator + Sub-agents Pattern — LangGraph StateGraph implementation.

A central orchestrator dispatches the incoming task to N specialist sub-agents
in parallel using the LangGraph Send API. Findings are aggregated using an
intent classification map, and then the shared Policy -> Plan -> Verify pipeline runs.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, Protocol, TypedDict, runtime_checkable
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from ..generic_agents import BasicVerifierAgent, RulePolicyAgent, TemplatePlannerAgent, safe_node
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification, SubAgentFinding
from ..retrieval import KnowledgeBaseRetriever


@runtime_checkable
class SubAgent(Protocol):
    """Protocol that every sub-agent must satisfy."""

    name: str

    def run(self, task: Task) -> SubAgentFinding:
        ...


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
    audit_trail: Annotated[list[str], operator.add]  # LangGraph merges via operator.add


class OrchestratorWeaver:
    """Orchestrator + Sub-agents pattern using LangGraph StateGraph.

    Parameters:
        sub_agents:       Tuple of sub-agents to run in parallel.
        policy_agent:     Gating policy agent.
        planner_agent:    Draft planning agent.
        verifier_agent:   Safety verification agent.
        retriever:        Vector retrieval client.
        intent_map:       Dict mapping sub-agent names to intent names (e.g. {"pii_detector": "pii"}).
        default_intent:   Intent when no sub-agent is triggered. Default: "safe".
        fallback_action:  Action to take if verification fails. Default: "escalate".
    """

    def __init__(
        self,
        sub_agents: tuple[SubAgent, ...],
        policy_agent: RulePolicyAgent,
        planner_agent: TemplatePlannerAgent,
        verifier_agent: BasicVerifierAgent,
        retriever: KnowledgeBaseRetriever,
        intent_map: dict[str, str],
        default_intent: str = "safe",
        fallback_action: str = "escalate",
        fallback_response: str = "Guardrail verification failed; route to review.",
    ) -> None:
        self.sub_agents = sub_agents
        self.policy_agent = policy_agent
        self.planner_agent = planner_agent
        self.verifier_agent = verifier_agent
        self.retriever = retriever
        self.intent_map = intent_map
        self.default_intent = default_intent
        self.fallback_action = fallback_action
        self.fallback_response = fallback_response

        builder = StateGraph(OrchestratorState)
        builder.add_node("run_sub_agent", safe_node(self._run_sub_agent_node, {"findings": []}))
        builder.add_node("aggregate", self._aggregate_node)
        builder.add_node("retrieve",  safe_node(self._retrieve_node, {"evidence": (), "audit_trail": ["retrieve:error"]}))
        builder.add_node("policy",    safe_node(self._policy_node,   {"audit_trail": ["policy:error"]}))
        builder.add_node("plan",      safe_node(self._plan_node,     {"audit_trail": ["plan:error"]}))
        builder.add_node("verify",    safe_node(self._verify_node,   {"audit_trail": ["verify:error"]}))
        builder.add_node(
            "finalize",
            safe_node(self._finalize_node, {"outcome": None, "audit_trail": ["finalize:error"]}),
        )

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
            intent = self.intent_map.get(triggered[0].name, "general")
            tags = tuple(f.name for f in triggered)
        else:
            intent = self.default_intent
            tags = ()

        signal = Signal(intent=intent, risk=total_risk, tags=tags)
        full_audit = list(state.get("audit_trail") or [])
        new_entries: list[str] = []

        # Match exact audit formatting
        for f in findings:
            new_entries.append(
                f"sub_agent:{f.name}:triggered={f.triggered}"
                f":risk_delta={f.risk_delta}:{f.reason}"
            )
        new_entries.append(
            f"signal:intent={intent}:risk={total_risk}"
            f":sub_agents_triggered={len(triggered)}/{len(self.sub_agents)}"
        )
        return {"signal": signal, "audit_trail": new_entries}

    def _retrieve_node(self, state: OrchestratorState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        return {"evidence": evidence, "audit_trail": [f"retrieval:documents={len(evidence)}"]}

    def _policy_node(self, state: OrchestratorState) -> dict:
        task = state["task"]
        signal = state["signal"]
        policy = self.policy_agent.run(task, signal)
        return {"policy": policy, "audit_trail": [f"policy:allowed={','.join(policy.allowed_actions)}"]}

    def _plan_node(self, state: OrchestratorState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = self.planner_agent.run(task, signal, evidence, policy)
        return {"draft": draft, "audit_trail": [f"draft:action={draft.action}:confidence={draft.confidence}"]}

    def _verify_node(self, state: OrchestratorState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.verifier_agent.run(draft, policy, evidence)
        return {"verification": verification, "audit_trail": [f"verify:approved={verification.approved}:score={verification.score}"]}

    def _finalize_node(self, state: OrchestratorState) -> dict:
        task = state["task"]
        signal = state.get("signal")
        evidence = state.get("evidence") or ()
        policy = state.get("policy")
        draft = state.get("draft")
        verification = state.get("verification")
        full_audit = list(state.get("audit_trail") or [])
        new_entries: list[str] = []

        # Guard: upstream safe_node failure may have left None in state.
        if policy is None or draft is None or verification is None:
            missing = [k for k, v in [("policy", policy), ("draft", draft), ("verification", verification)] if v is None]
            new_entries.append(f"finalize:upstream_failure:missing={','.join(missing)}:escalating")
            outcome = Outcome(
                task_id=task.id,
                use_case=task.use_case,
                signal=signal or Signal(intent="unknown", risk=3),
                evidence=evidence,
                policy=policy or PolicyDecision(
                    allowed_actions=(self.fallback_action,),
                    reason="upstream node failure",
                ),
                draft=draft or Plan(
                    action=self.fallback_action,
                    confidence=0.0,
                    response=self.fallback_response,
                    internal_note="upstream node failure",
                ),
                verification=verification or Verification(
                    approved=False, score=0.0,
                    findings=("upstream node failed — see audit_trail",),
                ),
                final_action=self.fallback_action,
                response=self.fallback_response,
                internal_note=f"Upstream node failure: {', '.join(missing)} was None.",
                audit_trail=full_audit + new_entries,
            )
            return {"outcome": outcome, "audit_trail": new_entries}

        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            note = f"Verifier blocked: {'; '.join(verification.findings)}"
            new_entries.append(f"fallback:{self.fallback_action}")

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
            audit_trail=full_audit + new_entries,
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
