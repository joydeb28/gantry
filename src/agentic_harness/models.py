from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    use_case: str
    title: str
    body: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}"


@dataclass(frozen=True)
class Evidence:
    source: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class Signal:
    intent: str
    risk: int
    missing_fields: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyDecision:
    allowed_actions: tuple[str, ...]
    blocked_actions: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Plan:
    action: str
    confidence: float
    response: str
    internal_note: str
    citations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Verification:
    approved: bool
    score: float
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outcome:
    task_id: str
    use_case: str
    signal: Signal
    evidence: tuple[Evidence, ...]
    policy: PolicyDecision
    draft: Plan
    verification: Verification
    final_action: str
    response: str
    internal_note: str
    audit_trail: list[str] = field(default_factory=list)
