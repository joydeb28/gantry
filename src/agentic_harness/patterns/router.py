"""Router / Dispatcher Pattern.

A router agent classifies the incoming task into a route and dispatches
it to the appropriate specialist agent. Each specialist runs its own
mini pipeline (policy + plan + verify), keeping domain logic isolated.

Pattern flow::

    Task
      -> RouterWeaver
           -> route classification (keyword match)
           -> dispatch to SpecialistAgent
                -> Policy -> Plan -> Verify
      -> Outcome

To add a new specialist: create a SpecialistAgent with its own
policy_agent and planner_agent, then add a route entry and keywords.

Use case: IT helpdesk (access requests, incidents, hardware).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..generic_agents import BasicVerifierAgent, RulePolicyAgent, TemplatePlannerAgent
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import TinyRetriever


@dataclass(frozen=True)
class SpecialistAgent:
    """A domain-specific agent bundle used by the RouterWeaver."""

    name: str
    policy_agent: RulePolicyAgent
    planner_agent: TemplatePlannerAgent


@dataclass
class RouterWeaver:
    """Router / Dispatcher pattern.

    A router agent classifies the incoming task and dispatches to
    the appropriate specialist agent. Each specialist runs its own
    policy + plan pipeline, keeping domain logic cleanly separated.
    """

    router_keywords: dict[str, tuple[str, ...]]  # route -> keywords
    routes: dict[str, SpecialistAgent]           # route -> specialist
    default_route: str
    verifier_agent: BasicVerifierAgent
    retriever: TinyRetriever
    fallback_action: str = "escalate"
    fallback_response: str = "No specialist available for this request; escalating."

    def run(self, task: Task) -> Outcome:
        audit: list[str] = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:router",
        ]

        # 1. Route classification
        text = task.text.lower()
        route = self.default_route
        for intent, keywords in self.router_keywords.items():
            if any(k in text for k in keywords):
                route = intent
                break
        audit.append(f"router:classified_route={route}")

        # 2. Build Signal from route
        signal = Signal(intent=route, risk=1)

        # 3. Retrieve evidence
        evidence = self.retriever.search(task.text)
        audit.append(f"retrieval:documents={len(evidence)}")

        # 4. Dispatch to specialist
        specialist = self.routes.get(route) or self.routes[self.default_route]
        audit.append(f"dispatch:specialist={specialist.name}")

        # 5. Specialist: policy + plan
        policy = specialist.policy_agent.run(task, signal)
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")

        draft = specialist.planner_agent.run(task, signal, evidence, policy)
        audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")

        # 6. Verify
        verification = self.verifier_agent.run(draft, policy, evidence)
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")

        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            note = f"Verifier blocked: {'; '.join(verification.findings)}"

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
