from __future__ import annotations

from .models import ActionType, PolicyClaim, Ticket


class PolicyEngine:
    """Simple policy-as-code guardrail for support workflows."""

    def evaluate(self, ticket: Ticket, intent: str) -> PolicyClaim:
        allowed = {ActionType.answer, ActionType.ask_clarifying_question, ActionType.escalate}
        blocked: set[ActionType] = set()
        reasons: list[str] = []

        if intent in {"refund_request", "damaged_item"}:
            if ticket.days_since_purchase <= 30 and ticket.order_value_usd <= 250:
                allowed.add(ActionType.refund)
                allowed.add(ActionType.replace)
                reasons.append("Self-serve refund/replacement is allowed within 30 days up to $250.")
            else:
                blocked.update({ActionType.refund, ActionType.replace})
                reasons.append("Refund/replacement requires human approval outside the 30-day or $250 limit.")

        if ticket.customer_tier.lower() == "vip":
            allowed.add(ActionType.escalate)
            reasons.append("VIP customers may be escalated for concierge handling.")

        return PolicyClaim(
            allowed_actions=tuple(sorted(allowed, key=lambda action: action.value)),
            blocked_actions=tuple(sorted(blocked, key=lambda action: action.value)),
            reason=" ".join(reasons) or "No special restrictions matched.",
        )
