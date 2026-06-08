from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from .retrieval import TinyRetriever


class SignalAgent(Protocol):
    def run(self, task: Task) -> Signal:
        ...


class PolicyAgent(Protocol):
    def run(self, task: Task, signal: Signal) -> PolicyDecision:
        ...


class PlannerAgent(Protocol):
    def run(
        self,
        task: Task,
        signal: Signal,
        evidence: tuple[Evidence, ...],
        policy: PolicyDecision,
    ) -> Plan:
        ...


class VerifierAgent(Protocol):
    def run(self, draft: Plan, policy: PolicyDecision, evidence: tuple[Evidence, ...]) -> Verification:
        ...


@dataclass(frozen=True)
class AgentRecipe:
    use_case: str
    signal_agent: SignalAgent
    policy_agent: PolicyAgent
    planner_agent: PlannerAgent
    verifier_agent: VerifierAgent
    fallback_action: str = "escalate"
    fallback_response: str = "This needs review before automation continues."


class ClaimWeaver:
    """Generic claim-weaving orchestration.

    Agents emit small typed claims. The workflow composes them only after policy
    and verification agree, producing an audit trail that works across domains.
    """

    def __init__(self, recipe: AgentRecipe, retriever: TinyRetriever):
        self.recipe = recipe
        self.retriever = retriever

    def run(self, task: Task) -> Outcome:
        audit = [f"task:{task.id}:received", f"use_case:{task.use_case}"]

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

        final = draft
        if not verification.approved:
            final = Plan(
                action=self.recipe.fallback_action,
                confidence=1.0,
                response=self.recipe.fallback_response,
                internal_note="Verifier blocked draft: " + "; ".join(verification.findings),
                citations=draft.citations,
            )
            audit.append(f"fallback:{self.recipe.fallback_action}")

        return Outcome(
            task_id=task.id,
            use_case=task.use_case,
            signal=signal,
            evidence=evidence,
            policy=policy,
            draft=draft,
            verification=verification,
            final_action=final.action,
            response=final.response,
            internal_note=final.internal_note,
            audit_trail=audit,
        )
