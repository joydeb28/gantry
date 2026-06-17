"""Parallel Orchestrator Pattern — LangGraph StateGraph implementation.

Fans out to N sub-agents in parallel using the LangGraph Send API.
Findings are fanned back in and aggregated using a custom domain policy.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, Optional, Protocol, TypedDict, runtime_checkable
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from ..generic_agents import BasicVerifierAgent, RulePolicyAgent, TemplatePlannerAgent, safe_node
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import KnowledgeBaseRetriever


@runtime_checkable
class ParallelSubAgent(Protocol):
    """Protocol that every parallel sub-agent must satisfy."""

    name: str

    def run(self, task: Task) -> Any:
        ...


class ParallelOrchestratorState(TypedDict):
    task: Task
    findings: Annotated[list[Any], operator.add]
    signal: Optional[Signal]
    evidence: tuple[Evidence, ...]
    policy: Optional[PolicyDecision]
    draft: Optional[Plan]
    verification: Optional[Verification]
    outcome: Optional[Outcome]
    audit_trail: list[str]


class ParallelSubAgentState(TypedDict):
    task: Task
    agent_idx: int


class ParallelOrchestratorWeaver:
    """Generic Parallel Orchestrator pattern using LangGraph StateGraph.

    Parameters:
        sub_agents:         Tuple of sub-agents implementing the ParallelSubAgent protocol.
        policy_agent:       Gating policy agent.
        planner_agent:      Draft planning agent.
        verifier_agent:     Safety verification agent.
        retriever:          Vector retrieval client.
        aggregation_policy: Callable that maps fanned-in findings to:
                            (intent: str, risk_level: int, tags: list[str], audit_logs: list[str])
        plan_enrichment_fn: Optional callable to modify the draft Plan based on findings.
        initial_audit_fn:   Optional callable to generate custom initial audit trail logs.
    """

    def __init__(
        self,
        sub_agents: tuple[ParallelSubAgent, ...],
        policy_agent: RulePolicyAgent,
        planner_agent: TemplatePlannerAgent,
        verifier_agent: BasicVerifierAgent,
        retriever: KnowledgeBaseRetriever,
        aggregation_policy: Callable[[list[Any]], tuple[str, int, list[str], list[str]]],
        plan_enrichment_fn: Optional[Callable[[Plan, list[Any]], Plan]] = None,
        fallback_action: str = "escalate",
        fallback_response: str = "Verification failed; routing to review.",
        initial_audit_fn: Optional[Callable[[Task], list[str]]] = None,
    ) -> None:
        self.sub_agents = sub_agents
        self.policy_agent = policy_agent
        self.planner_agent = planner_agent
        self.verifier_agent = verifier_agent
        self.retriever = retriever
        self.aggregation_policy = aggregation_policy
        self.plan_enrichment_fn = plan_enrichment_fn
        self.fallback_action = fallback_action
        self.fallback_response = fallback_response
        self.initial_audit_fn = initial_audit_fn

        builder = StateGraph(ParallelOrchestratorState)
        builder.add_node("run_sub_agent", safe_node(self._run_sub_agent_node, {"findings": []}))
        builder.add_node("aggregate", self._aggregate_node)
        builder.add_node("retrieve",  safe_node(self._retrieve_node, {"evidence": (), "audit_trail": ["retrieve:error"]}))
        builder.add_node("policy",    safe_node(self._policy_node,   {"audit_trail": ["policy:error"]}))
        builder.add_node("plan",      safe_node(self._plan_node,     {"audit_trail": ["plan:error"]}))
        builder.add_node("verify",    safe_node(self._verify_node,   {"audit_trail": ["verify:error"]}))
        builder.add_node("finalize",  self._finalize_node)

        builder.add_conditional_edges(START, self._dispatch_sub_agents, ["run_sub_agent"])
        builder.add_edge("run_sub_agent", "aggregate")
        builder.add_edge("aggregate", "retrieve")
        builder.add_edge("retrieve", "policy")
        builder.add_edge("policy", "plan")
        builder.add_edge("plan", "verify")
        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)

        self.graph = builder.compile()

    def _dispatch_sub_agents(self, state: ParallelOrchestratorState) -> list[Send]:
        return [
            Send("run_sub_agent", {"task": state["task"], "agent_idx": i})
            for i in range(len(self.sub_agents))
        ]

    def _run_sub_agent_node(self, state: ParallelSubAgentState) -> dict:
        agent = self.sub_agents[state["agent_idx"]]
        finding = agent.run(state["task"])
        return {"findings": [finding]}

    def _aggregate_node(self, state: ParallelOrchestratorState) -> dict:
        findings = state["findings"]
        intent, risk, tags, audit_logs = self.aggregation_policy(findings)

        audit = list(state.get("audit_trail") or [])
        audit.extend(audit_logs)

        signal = Signal(intent=intent, risk=risk, tags=tuple(tags))
        audit.append(f"signal:intent={intent}:risk={risk}")
        return {"signal": signal, "audit_trail": audit}

    def _retrieve_node(self, state: ParallelOrchestratorState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"retrieval:documents={len(evidence)}")
        return {"evidence": evidence, "audit_trail": audit}

    def _policy_node(self, state: ParallelOrchestratorState) -> dict:
        task = state["task"]
        signal = state["signal"]
        policy = self.policy_agent.run(task, signal)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")
        return {"policy": policy, "audit_trail": audit}

    def _plan_node(self, state: ParallelOrchestratorState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = self.planner_agent.run(task, signal, evidence, policy)

        if self.plan_enrichment_fn:
            draft = self.plan_enrichment_fn(draft, state["findings"])

        audit = list(state.get("audit_trail") or [])
        audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")
        return {"draft": draft, "audit_trail": audit}

    def _verify_node(self, state: ParallelOrchestratorState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.verifier_agent.run(draft, policy, evidence)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")
        return {"verification": verification, "audit_trail": audit}

    def _finalize_node(self, state: ParallelOrchestratorState) -> dict:
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
        initial_audit = []
        if self.initial_audit_fn:
            initial_audit = self.initial_audit_fn(task)
        else:
            initial_audit = [
                f"task:{task.id}:received",
                f"use_case:{task.use_case}",
                "pattern:parallel_orchestrator",
            ]
        result = self.graph.invoke({"task": task, "audit_trail": initial_audit, "findings": []})
        return result["outcome"]
