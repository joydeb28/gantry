from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Sentiment(str, Enum):
    calm = "calm"
    frustrated = "frustrated"
    urgent = "urgent"


class ActionType(str, Enum):
    answer = "answer"
    ask_clarifying_question = "ask_clarifying_question"
    refund = "refund"
    replace = "replace"
    escalate = "escalate"


@dataclass(frozen=True)
class Ticket:
    id: str
    subject: str
    body: str
    customer_tier: str = "standard"
    order_value_usd: float = 0.0
    days_since_purchase: int = 0

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}"


@dataclass(frozen=True)
class Evidence:
    source: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class TriageClaim:
    intent: str
    sentiment: Sentiment
    urgency: int
    missing_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyClaim:
    allowed_actions: tuple[ActionType, ...]
    blocked_actions: tuple[ActionType, ...]
    reason: str


@dataclass(frozen=True)
class ActionPlan:
    action: ActionType
    confidence: float
    customer_reply: str
    internal_note: str
    citations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verification:
    approved: bool
    score: float
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Resolution:
    ticket_id: str
    triage: TriageClaim
    evidence: tuple[Evidence, ...]
    policy: PolicyClaim
    draft: ActionPlan
    verification: Verification
    final_action: ActionType
    customer_reply: str
    internal_note: str
    audit_trail: list[str] = field(default_factory=list)
