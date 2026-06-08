from __future__ import annotations

from .models import ActionPlan, ActionType, Evidence, PolicyClaim, Sentiment, Ticket, TriageClaim, Verification


class TriageAgent:
    def run(self, ticket: Ticket) -> TriageClaim:
        text = ticket.text.lower()
        intent = "general_question"
        if any(word in text for word in ("refund", "money back", "charged")):
            intent = "refund_request"
        if any(word in text for word in ("broken", "damaged", "cracked", "defective")):
            intent = "damaged_item"
        if any(word in text for word in ("login", "password", "account")):
            intent = "account_access"

        sentiment = Sentiment.calm
        if any(word in text for word in ("angry", "frustrated", "upset", "terrible")):
            sentiment = Sentiment.frustrated
        if any(word in text for word in ("urgent", "asap", "immediately", "today")):
            sentiment = Sentiment.urgent

        urgency = 1 + int(sentiment != Sentiment.calm) + int(ticket.customer_tier.lower() == "vip")
        missing = ()
        if intent in {"refund_request", "damaged_item"} and ticket.order_value_usd <= 0:
            missing = ("order_value_usd",)
        return TriageClaim(intent=intent, sentiment=sentiment, urgency=min(urgency, 3), missing_fields=missing)


class ResolutionAgent:
    def run(self, ticket: Ticket, triage: TriageClaim, evidence: tuple[Evidence, ...], policy: PolicyClaim) -> ActionPlan:
        if triage.missing_fields:
            return ActionPlan(
                action=ActionType.ask_clarifying_question,
                confidence=0.72,
                customer_reply="Thanks for reaching out. Could you share the missing order details so I can resolve this quickly?",
                internal_note=f"Missing fields: {', '.join(triage.missing_fields)}",
            )

        if triage.intent == "damaged_item" and ActionType.replace in policy.allowed_actions:
            action = ActionType.replace
            reply = "I am sorry the item arrived damaged. I can arrange a replacement right away under our 30-day damage policy."
        elif triage.intent == "refund_request" and ActionType.refund in policy.allowed_actions:
            action = ActionType.refund
            reply = "I can help with that refund. Your order appears eligible under our 30-day refund policy."
        elif triage.intent in {"damaged_item", "refund_request"}:
            action = ActionType.escalate
            reply = "I am sending this to a support specialist because this request needs human approval under our policy."
        elif triage.urgency >= 3:
            action = ActionType.escalate
            reply = "I am routing this to a specialist now so we can handle it with the right priority."
        else:
            action = ActionType.answer
            reply = self._answer_from_evidence(evidence)

        citations = tuple(item.title for item in evidence[:2])
        return ActionPlan(
            action=action,
            confidence=0.84 if citations else 0.62,
            customer_reply=reply,
            internal_note=f"Intent={triage.intent}; policy={policy.reason}",
            citations=citations,
        )

    def _answer_from_evidence(self, evidence: tuple[Evidence, ...]) -> str:
        if not evidence:
            return "Thanks for the details. I could not find a matching support article, so I am escalating this for review."
        return f"Here is what I found: {evidence[0].title}. If this does not solve it, I can route the ticket to a specialist."


class VerifierAgent:
    def run(self, draft: ActionPlan, policy: PolicyClaim, evidence: tuple[Evidence, ...]) -> Verification:
        findings: list[str] = []
        if draft.action not in policy.allowed_actions:
            findings.append(f"Action '{draft.action.value}' is not allowed by policy.")
        if draft.action in {ActionType.answer, ActionType.refund, ActionType.replace} and not evidence:
            findings.append("No evidence was retrieved for the proposed action.")
        if draft.confidence < 0.7:
            findings.append("Draft confidence is below threshold.")

        score = 1.0 - (0.25 * len(findings))
        return Verification(approved=not findings, score=max(score, 0.0), findings=tuple(findings))
