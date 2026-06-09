from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .generic_agents import BasicVerifierAgent, KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent
from .models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from .retrieval import TinyRetriever


@runtime_checkable
class SignalAgent(Protocol):
    def run(self, task: Task) -> Signal: ...


@runtime_checkable
class PolicyAgent(Protocol):
    def run(self, task: Task, signal: Signal) -> PolicyDecision: ...


@runtime_checkable
class PlannerAgent(Protocol):
    def run(self, task: Task, signal: Signal, evidence: tuple[Evidence, ...], policy: PolicyDecision) -> Plan: ...


@runtime_checkable
class VerifierAgent(Protocol):
    def run(self, draft: Plan, policy: PolicyDecision, evidence: tuple[Evidence, ...]) -> Verification: ...


@dataclass(frozen=True)
class AgentRecipe:
    use_case: str
    signal_agent: SignalAgent
    policy_agent: PolicyAgent
    planner_agent: PlannerAgent
    verifier_agent: VerifierAgent
    fallback_action: str = "escalate"
    fallback_response: str = "This needs review before automation continues."


@dataclass
class ClaimWeaver:
    """Pipeline Pattern — sequential typed-claim chain.

    Task -> Signal -> Evidence -> Policy -> Plan -> Verification -> Outcome

    This is the simplest agentic pattern and the foundation all others build on.
    Use cases: customer support, CRM, research, coding.
    """

    recipe: AgentRecipe
    retriever: TinyRetriever

    def run(self, task: Task) -> Outcome:
        audit: list[str] = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:pipeline",
        ]

        signal = self.recipe.signal_agent.run(task)
        audit.append(f"signal:intent={signal.intent}:risk={signal.risk}")

        evidence = self.retriever.search(task.text)
        audit.append(f"retrieval:documents={len(evidence)}")

        policy = self.recipe.policy_agent.run(task, signal)
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")

        draft = self.recipe.planner_agent.run(task, signal, evidence, policy)
        audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")

        verification = self.recipe.verifier_agent.run(draft, policy, evidence)
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")

        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.recipe.fallback_action
            response = self.recipe.fallback_response
            note = f"Verifier blocked: {'; '.join(verification.findings)}"
            audit.append(f"fallback:{self.recipe.fallback_action}")

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
