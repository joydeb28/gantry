"""Pipeline Pattern — LangGraph StateGraph implementation.

The pipeline pattern is the default sequential chain:
    Task -> Signal -> Evidence -> Policy -> Plan -> Verification -> Outcome

Each step has exactly one job and passes a typed claim to the next step.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional
from langgraph.graph import StateGraph, START, END

from ..models import Task, Signal, Evidence, PolicyDecision, Plan, Verification, Outcome, AgentRecipe
from ..retrieval import KnowledgeBaseRetriever
from ..generic_agents import safe_node


class PipelineState(TypedDict):
    task: Task
    signal: Optional[Signal]
    evidence: tuple[Evidence, ...]
    policy: Optional[PolicyDecision]
    draft: Optional[Plan]
    verification: Optional[Verification]
    outcome: Optional[Outcome]
    audit_trail: Annotated[list[str], operator.add]  # LangGraph merges via operator.add


class PipelineWeaver:
    """Pipeline Pattern using LangGraph StateGraph."""

    def __init__(self, recipe: AgentRecipe, retriever: KnowledgeBaseRetriever) -> None:
        self.recipe = recipe
        self.retriever = retriever

        builder = StateGraph(PipelineState)
        builder.add_node("signal",   safe_node(self._signal_node,   {"audit_trail": ["signal:error"]}))
        builder.add_node("retrieve", safe_node(self._retrieve_node, {"evidence": (), "audit_trail": ["retrieve:error"]}))
        builder.add_node("policy",   safe_node(self._policy_node,   {"audit_trail": ["policy:error"]}))
        builder.add_node("plan",     safe_node(self._plan_node,     {"audit_trail": ["plan:error"]}))
        builder.add_node("verify",   safe_node(self._verify_node,   {"audit_trail": ["verify:error"]}))
        builder.add_node(
            "finalize",
            safe_node(self._finalize_node, self._fallback_outcome_dict()),
        )

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
        return {
            "signal": signal,
            "audit_trail": [f"signal:intent={signal.intent}:risk={signal.risk}"],
        }

    def _retrieve_node(self, state: PipelineState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        return {
            "evidence": evidence,
            "audit_trail": [f"retrieval:documents={len(evidence)}"],
        }

    def _policy_node(self, state: PipelineState) -> dict:
        task = state["task"]
        signal = state["signal"]
        policy = self.recipe.policy_agent.run(task, signal)
        return {
            "policy": policy,
            "audit_trail": [f"policy:allowed={','.join(policy.allowed_actions)}"],
        }

    def _plan_node(self, state: PipelineState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = self.recipe.planner_agent.run(task, signal, evidence, policy)
        return {
            "draft": draft,
            "audit_trail": [f"draft:action={draft.action}:confidence={draft.confidence}"],
        }

    def _verify_node(self, state: PipelineState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.recipe.verifier_agent.run(draft, policy, evidence)
        return {
            "verification": verification,
            "audit_trail": [f"verify:approved={verification.approved}:score={verification.score}"],
        }

    def _fallback_outcome_dict(self) -> dict:
        """Return a safe-node fallback dict used when _finalize_node itself fails."""
        return {"outcome": None, "audit_trail": ["finalize:error"]}

    def _finalize_node(self, state: PipelineState) -> dict:
        task = state["task"]
        signal = state.get("signal")
        evidence = state.get("evidence") or ()
        policy = state.get("policy")
        draft = state.get("draft")
        verification = state.get("verification")
        # Full trail accumulated by LangGraph reducer across all previous nodes
        full_audit = list(state.get("audit_trail") or [])
        new_entries: list[str] = []

        # Guard: upstream safe_node failure may have left None in state.
        # Degrade gracefully to an escalation Outcome rather than crashing.
        if policy is None or draft is None or verification is None:
            missing = [k for k, v in [("policy", policy), ("draft", draft), ("verification", verification)] if v is None]
            new_entries.append(f"finalize:upstream_failure:missing={','.join(missing)}:escalating")
            outcome = Outcome(
                task_id=task.id,
                use_case=task.use_case,
                signal=signal or Signal(intent="unknown", risk=3),
                evidence=evidence,
                policy=policy or PolicyDecision(
                    allowed_actions=(self.recipe.fallback_action,),
                    reason="upstream node failure",
                ),
                draft=draft or Plan(
                    action=self.recipe.fallback_action,
                    confidence=0.0,
                    response=self.recipe.fallback_response,
                    internal_note="upstream node failure",
                ),
                verification=verification or Verification(
                    approved=False, score=0.0,
                    findings=("upstream node failed — see audit_trail",),
                ),
                final_action=self.recipe.fallback_action,
                response=self.recipe.fallback_response,
                internal_note=f"Upstream node failure: {', '.join(missing)} was None.",
                audit_trail=full_audit + new_entries,
            )
            return {"outcome": outcome, "audit_trail": new_entries}

        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.recipe.fallback_action
            response = self.recipe.fallback_response
            note = f"Verifier blocked: {'; '.join(verification.findings)}"
            new_entries.append(f"fallback:{self.recipe.fallback_action}")

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
        return {"outcome": outcome, "audit_trail": new_entries}

    def run(self, task: Task) -> Outcome:
        initial_audit = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:pipeline",
        ]
        result = self.graph.invoke({"task": task, "audit_trail": initial_audit})
        return result["outcome"]
