"""Reflection / Critic Loop Pattern.

A drafter agent produces a plan. A critic agent scores it against
policy, confidence, and citation requirements. If the score is below
the approval threshold and there are turns remaining, the drafter
revises the plan incorporating the critic's findings. The loop
continues until the critic approves or max_turns is reached.

Pattern flow::

    Task -> Signal -> Evidence -> Policy
      -> Turn 1: Drafter -> Plan
                 Critic  -> Critique (score, findings)
      -> Turn 2: Drafter revises -> Plan
                 Critic          -> Critique
      -> ... (up to max_turns)
      -> Final Verifier -> Outcome

To customise the critic, extend CriticAgent or replace it with your
own class implementing run(draft, policy, evidence) -> Critique.

Use case: legal document review (high-stakes, must cite evidence).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..generic_agents import BasicVerifierAgent, KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import TinyRetriever


@dataclass(frozen=True)
class Critique:
    """Structured feedback from the CriticAgent."""

    score: float          # 0.0 (reject) to 1.0 (perfect)
    findings: tuple[str, ...]
    approved: bool


class CriticAgent:
    """Reviews a draft plan and returns a Critique.

    Checks performed:
    - Action must be in policy.allowed_actions
    - Confidence must meet the minimum threshold
    - Citations required if require_citations=True
    - High-impact actions must have supporting evidence

    Extend or replace this class to add domain-specific checks,
    e.g. clause validation for legal documents.
    """

    def __init__(self, min_confidence: float = 0.75, require_citations: bool = True) -> None:
        self.min_confidence = min_confidence
        self.require_citations = require_citations

    def run(
        self,
        draft: Plan,
        policy: PolicyDecision,
        evidence: tuple[Evidence, ...],
    ) -> Critique:
        findings: list[str] = []

        if draft.action not in policy.allowed_actions:
            findings.append(f"Action '{draft.action}' is not in allowed actions.")
        if draft.confidence < self.min_confidence:
            findings.append(
                f"Confidence {draft.confidence:.2f} is below minimum {self.min_confidence}."
            )
        if self.require_citations and not draft.citations:
            findings.append("No citations provided; legal review requires source evidence.")
        if draft.action not in {"escalate", "ask_for_info", "log_only"} and not evidence:
            findings.append("No evidence retrieved for a substantive action.")

        score = max(0.0, 1.0 - 0.3 * len(findings))
        return Critique(score=score, findings=tuple(findings), approved=not findings)


@dataclass
class ReflectionWeaver:
    """Reflection / Critic Loop pattern.

    A drafter agent produces a plan. A critic agent scores it.
    If the score is below threshold and turns remain, the drafter
    revises the plan using the critique findings as feedback.
    The audit trail records every draft/critique turn.
    """

    signal_agent: KeywordSignalAgent
    policy_agent: RulePolicyAgent
    drafter_agent: TemplatePlannerAgent
    critic_agent: CriticAgent
    verifier_agent: BasicVerifierAgent
    retriever: TinyRetriever
    max_turns: int = 3
    approval_threshold: float = 0.7
    fallback_action: str = "escalate"
    fallback_response: str = "Could not produce an approved plan after reflection; escalating."

    def run(self, task: Task) -> Outcome:
        audit: list[str] = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:reflection",
        ]

        # 1. Signal + Evidence + Policy (same as pipeline)
        signal = self.signal_agent.run(task)
        audit.append(f"signal:intent={signal.intent}:risk={signal.risk}")

        evidence = self.retriever.search(task.text)
        audit.append(f"retrieval:documents={len(evidence)}")

        policy = self.policy_agent.run(task, signal)
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")

        # 2. Reflection loop: Draft -> Critique -> (Revise?)
        draft = self.drafter_agent.run(task, signal, evidence, policy)

        for turn in range(1, self.max_turns + 1):
            critique = self.critic_agent.run(draft, policy, evidence)
            audit.append(
                f"turn:{turn}:action={draft.action}:confidence={draft.confidence:.2f}"
                f":critique_score={critique.score:.2f}:approved={critique.approved}"
            )

            if critique.approved or critique.score >= self.approval_threshold:
                break

            if turn < self.max_turns:
                # Revision: annotate the plan with critique feedback
                critique_note = "; ".join(critique.findings)
                revised_action = (
                    draft.action if draft.action in policy.allowed_actions else "escalate"
                )
                draft = Plan(
                    action=revised_action,
                    confidence=min(1.0, draft.confidence + 0.06 * turn),
                    response=draft.response,
                    internal_note=(
                        f"{draft.internal_note} | revision:{turn} | critique:{critique_note}"
                    ),
                    citations=draft.citations or tuple(e.title for e in evidence[:2]),
                )

        # 3. Final verifier gate
        verification = self.verifier_agent.run(draft, policy, evidence)
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")

        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            note = (
                f"Reflection failed after {self.max_turns} turns. "
                f"Findings: {'; '.join(verification.findings)}"
            )

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
