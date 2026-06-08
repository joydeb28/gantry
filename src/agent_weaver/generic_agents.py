from __future__ import annotations

from dataclasses import dataclass

from .models import Evidence, Plan, PolicyDecision, Signal, Task, Verification


@dataclass(frozen=True)
class KeywordSignalAgent:
    intents: dict[str, tuple[str, ...]]
    high_risk_words: tuple[str, ...] = ()
    missing_fields_by_intent: dict[str, tuple[str, ...]] | None = None
    default_intent: str = "general"

    def run(self, task: Task) -> Signal:
        text = task.text.lower()
        intent = self.default_intent
        for candidate, keywords in self.intents.items():
            if any(keyword in text for keyword in keywords):
                intent = candidate
                break

        risk = 1 + int(any(word in text for word in self.high_risk_words))
        missing = []
        for field in (self.missing_fields_by_intent or {}).get(intent, ()):
            if field not in task.metadata or task.metadata[field] in ("", 0, False):
                missing.append(field)

        tags = tuple(keyword for keyword in self.high_risk_words if keyword in text)
        return Signal(intent=intent, risk=min(risk, 3), missing_fields=tuple(missing), tags=tags)


@dataclass(frozen=True)
class RulePolicyAgent:
    base_actions: tuple[str, ...]
    intent_actions: dict[str, tuple[str, ...]]
    blocked_when: dict[str, tuple[str, ...]] | None = None

    def run(self, task: Task, signal: Signal) -> PolicyDecision:
        allowed = set(self.base_actions)
        allowed.update(self.intent_actions.get(signal.intent, ()))
        blocked: set[str] = set()
        reasons: list[str] = []

        for action, conditions in (self.blocked_when or {}).items():
            if all(self._condition_matches(task, condition) for condition in conditions):
                allowed.discard(action)
                blocked.add(action)
                reasons.append(f"{action} blocked by {', '.join(conditions)}.")

        return PolicyDecision(
            allowed_actions=tuple(sorted(allowed)),
            blocked_actions=tuple(sorted(blocked)),
            reason=" ".join(reasons) or "Policy allowed the selected automation path.",
        )

    def _condition_matches(self, task: Task, condition: str) -> bool:
        if condition == "high_value":
            return float(task.metadata.get("value_usd", 0)) > 250
        if condition == "outside_30_days":
            return int(task.metadata.get("days_since_event", 0)) > 30
        if condition == "external_send":
            return bool(task.metadata.get("external_send", False))
        if condition == "production_change":
            return bool(task.metadata.get("production_change", False))
        if condition == "no_sources":
            return int(task.metadata.get("source_count", 1)) == 0
        return False


@dataclass(frozen=True)
class TemplatePlannerAgent:
    templates: dict[str, tuple[str, str]]
    default_action: str
    default_response: str

    def run(
        self,
        task: Task,
        signal: Signal,
        evidence: tuple[Evidence, ...],
        policy: PolicyDecision,
    ) -> Plan:
        if signal.missing_fields:
            return Plan(
                action="ask_for_info",
                confidence=0.75,
                response=f"Please provide: {', '.join(signal.missing_fields)}.",
                internal_note=f"Missing fields for intent={signal.intent}",
            )

        action, response = self.templates.get(signal.intent, (self.default_action, self.default_response))
        if action not in policy.allowed_actions:
            action = "escalate"
            response = "This request needs review before I can take action."

        citations = tuple(item.title for item in evidence[:2])
        confidence = 0.86 if citations else 0.64
        return Plan(
            action=action,
            confidence=confidence,
            response=response,
            internal_note=f"intent={signal.intent}; policy={policy.reason}",
            citations=citations,
        )


class BasicVerifierAgent:
    def run(self, draft: Plan, policy: PolicyDecision, evidence: tuple[Evidence, ...]) -> Verification:
        findings: list[str] = []
        if draft.action not in policy.allowed_actions:
            findings.append(f"Action '{draft.action}' is not allowed.")
        if draft.action not in {"escalate", "ask_for_info", "log_only"} and not evidence:
            findings.append("No evidence retrieved for automated action.")
        if draft.confidence < 0.7:
            findings.append("Draft confidence below threshold.")

        score = max(0.0, 1.0 - 0.25 * len(findings))
        return Verification(approved=not findings, score=score, findings=tuple(findings))
