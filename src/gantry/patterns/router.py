"""Router / Dispatcher Pattern — LangGraph StateGraph implementation.

A router agent classifies the incoming task into a route and dispatches
it to the appropriate specialist agent. Each specialist runs its own
mini pipeline (policy + plan), keeping domain logic isolated.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

from ..generic_agents import BasicVerifierAgent, RulePolicyAgent, TemplatePlannerAgent, safe_node
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import KnowledgeBaseRetriever


class SpecialistAgent:
    """A domain-specific agent bundle used by the RouterWeaver."""

    def __init__(self, name: str, policy_agent: RulePolicyAgent, planner_agent: TemplatePlannerAgent) -> None:
        self.name = name
        self.policy_agent = policy_agent
        self.planner_agent = planner_agent


class RouterState(TypedDict):
    task: Task
    route: str
    signal: Optional[Signal]
    evidence: tuple[Evidence, ...]
    policy: Optional[PolicyDecision]
    draft: Optional[Plan]
    verification: Optional[Verification]
    outcome: Optional[Outcome]
    audit_trail: list[str]


class RouterWeaver:
    """Router / Dispatcher pattern using LangGraph StateGraph."""

    def __init__(
        self,
        router_keywords: dict[str, tuple[str, ...]],
        routes: dict[str, SpecialistAgent],
        default_route: str,
        verifier_agent: BasicVerifierAgent,
        retriever: KnowledgeBaseRetriever,
        fallback_action: str = "escalate",
        fallback_response: str = "No specialist available for this request; escalating.",
    ) -> None:
        self.router_keywords = router_keywords
        self.routes = routes
        self.default_route = default_route
        self.verifier_agent = verifier_agent
        self.retriever = retriever
        self.fallback_action = fallback_action
        self.fallback_response = fallback_response

        builder = StateGraph(RouterState)
        builder.add_node("router",   safe_node(self._router_node,   {"route": self.default_route, "audit_trail": ["router:error"]}))
        builder.add_node("retrieve", safe_node(self._retrieve_node, {"evidence": (), "audit_trail": ["retrieve:error"]}))

        # Add dynamic specialist nodes
        for name in self.routes.keys():
            builder.add_node(f"specialist_{name}", safe_node(self._make_specialist_node(name), {"audit_trail": [f"specialist_{name}:error"]}))

        builder.add_node("verify",   safe_node(self._verify_node,   {"audit_trail": ["verify:error"]}))
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "router")
        builder.add_edge("router", "retrieve")

        # Routing logic
        builder.add_conditional_edges(
            "retrieve",
            self._route_decision,
            {name: f"specialist_{name}" for name in self.routes.keys()}
        )

        for name in self.routes.keys():
            builder.add_edge(f"specialist_{name}", "verify")

        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)

        self.graph = builder.compile()

    def _router_node(self, state: RouterState) -> dict:
        task = state["task"]
        text = task.text.lower()
        route = self.default_route
        for intent, keywords in self.router_keywords.items():
            if any(k in text for k in keywords):
                route = intent
                break

        audit = list(state.get("audit_trail") or [])
        audit.append(f"router:classified_route={route}")

        signal = Signal(intent=route, risk=1)
        return {"route": route, "signal": signal, "audit_trail": audit}

    def _retrieve_node(self, state: RouterState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"retrieval:documents={len(evidence)}")
        return {"evidence": evidence, "audit_trail": audit}

    def _route_decision(self, state: RouterState) -> str:
        return state["route"]

    def _make_specialist_node(self, route_name: str) -> Any:
        def specialist_node(state: RouterState) -> dict:
            task = state["task"]
            signal = state["signal"]
            evidence = state["evidence"]
            specialist = self.routes[route_name]

            audit = list(state.get("audit_trail") or [])
            audit.append(f"dispatch:specialist={specialist.name}")

            policy = specialist.policy_agent.run(task, signal)
            audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")

            draft = specialist.planner_agent.run(task, signal, evidence, policy)
            audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")

            return {"policy": policy, "draft": draft, "audit_trail": audit}
        return specialist_node

    def _verify_node(self, state: RouterState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.verifier_agent.run(draft, policy, evidence)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")
        return {"verification": verification, "audit_trail": audit}

    def _finalize_node(self, state: RouterState) -> dict:
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
            "pattern:router",
        ]
        result = self.graph.invoke({"task": task, "audit_trail": initial_audit})
        return result["outcome"]
