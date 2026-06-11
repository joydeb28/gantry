"""Pipeline Pattern — LangGraph StateGraph implementation.

The pipeline pattern is the default sequential chain:
    Task -> Signal -> Evidence -> Policy -> Plan -> Verification -> Outcome

Each step has exactly one job and passes a typed claim to the next step.
"""

from __future__ import annotations

from typing import Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from ..models import Task, Signal, Evidence, PolicyDecision, Plan, Verification, Outcome, AgentRecipe
from ..retrieval import KnowledgeBaseRetriever


class PipelineState(TypedDict):
    task: Task
    signal: Optional[Signal]
    evidence: tuple[Evidence, ...]
    policy: Optional[PolicyDecision]
    draft: Optional[Plan]
    verification: Optional[Verification]
    outcome: Optional[Outcome]
    audit_trail: list[str]


class PipelineWeaver:
    """Pipeline Pattern using LangGraph StateGraph."""

    def __init__(self, recipe: AgentRecipe, retriever: KnowledgeBaseRetriever) -> None:
        self.recipe = recipe
        self.retriever = retriever

        builder = StateGraph(PipelineState)
        builder.add_node("signal", self._signal_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("policy", self._policy_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("verify", self._verify_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "signal")
        builder.add_edge("signal", "retrieve")
        builder.add_edge("retrieve", "policy")
        builder.add_edge("policy", "plan")
        builder.add_edge("plan", "verify")
        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)

        self.graph = builder.compile()

    def _signal_node(self, state: PipelineState) -> dict:
        task = state["task"]
        signal = self.recipe.signal_agent.run(task)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"signal:intent={signal.intent}:risk={signal.risk}")
        return {"signal": signal, "audit_trail": audit}

    def _retrieve_node(self, state: PipelineState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"retrieval:documents={len(evidence)}")
        return {"evidence": evidence, "audit_trail": audit}

    def _policy_node(self, state: PipelineState) -> dict:
        task = state["task"]
        signal = state["signal"]
        policy = self.recipe.policy_agent.run(task, signal)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")
        return {"policy": policy, "audit_trail": audit}

    def _plan_node(self, state: PipelineState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = self.recipe.planner_agent.run(task, signal, evidence, policy)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")
        return {"draft": draft, "audit_trail": audit}

    def _verify_node(self, state: PipelineState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.recipe.verifier_agent.run(draft, policy, evidence)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")
        return {"verification": verification, "audit_trail": audit}

    def _finalize_node(self, state: PipelineState) -> dict:
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
            final_action = self.recipe.fallback_action
            response = self.recipe.fallback_response
            note = f"Verifier blocked: {'; '.join(verification.findings)}"
            audit.append(f"fallback:{self.recipe.fallback_action}")

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
            "pattern:pipeline",
        ]
        result = self.graph.invoke({"task": task, "audit_trail": initial_audit})
        return result["outcome"]
