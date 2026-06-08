from __future__ import annotations

from .agents import ResolutionAgent, TriageAgent, VerifierAgent
from .models import ActionPlan, ActionType, Resolution, Ticket
from .policy import PolicyEngine
from .retrieval import TinyRetriever


class SupportWeaver:
    """Claim-weaving workflow for support resolution.

    Each agent emits a small typed claim. The workflow composes claims only after
    policy and verification agree, creating an inspectable audit trail.
    """

    def __init__(
        self,
        retriever: TinyRetriever,
        triage_agent: TriageAgent | None = None,
        policy_engine: PolicyEngine | None = None,
        resolution_agent: ResolutionAgent | None = None,
        verifier_agent: VerifierAgent | None = None,
    ):
        self.retriever = retriever
        self.triage_agent = triage_agent or TriageAgent()
        self.policy_engine = policy_engine or PolicyEngine()
        self.resolution_agent = resolution_agent or ResolutionAgent()
        self.verifier_agent = verifier_agent or VerifierAgent()

    def resolve(self, ticket: Ticket) -> Resolution:
        audit = [f"ticket:{ticket.id}:received"]
        triage = self.triage_agent.run(ticket)
        audit.append(f"triage:intent={triage.intent}:urgency={triage.urgency}")

        evidence = self.retriever.search(ticket.text)
        audit.append(f"retrieval:documents={len(evidence)}")

        policy = self.policy_engine.evaluate(ticket, triage.intent)
        audit.append(f"policy:allowed={','.join(action.value for action in policy.allowed_actions)}")

        draft = self.resolution_agent.run(ticket, triage, evidence, policy)
        audit.append(f"draft:action={draft.action.value}:confidence={draft.confidence}")

        verification = self.verifier_agent.run(draft, policy, evidence)
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")

        final = draft
        if not verification.approved:
            final = ActionPlan(
                action=ActionType.escalate,
                confidence=1.0,
                customer_reply="Thanks for the details. I am sending this to a support specialist for review.",
                internal_note="Verifier blocked automation: " + "; ".join(verification.findings),
                citations=draft.citations,
            )
            audit.append("fallback:escalated")

        return Resolution(
            ticket_id=ticket.id,
            triage=triage,
            evidence=evidence,
            policy=policy,
            draft=draft,
            verification=verification,
            final_action=final.action,
            customer_reply=final.customer_reply,
            internal_note=final.internal_note,
            audit_trail=audit,
        )
