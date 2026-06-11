"""Generic agent components used across all LangGraph patterns.

Each class is a configurable, stateless agent that performs one step of
the agentic pipeline. They are used as node functions inside LangGraph
StateGraphs — each node calls one agent and returns the result as a
state update dict.

Agents:
    KeywordSignalAgent   : Keyword-based intent + risk extraction
    RulePolicyAgent      : Rule-based action gating
    TemplatePlannerAgent : Template-based plan generation (no LLM)
    BasicVerifierAgent   : Policy + confidence + evidence gate

LLM-powered planning is in llm.py (LangChainPlanner).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import Evidence, Plan, PolicyDecision, Signal, Task, Verification


# ---------------------------------------------------------------------------
# Keyword Signal Agent
# ---------------------------------------------------------------------------

@dataclass
class KeywordSignalAgent:
    """Classify a task into an intent using keyword matching.

    Detects high-risk language, missing required metadata fields, and
    derives a risk level (0–3) from the task text.

    Args:
        intents:                 ``{intent_name: (keyword, ...)}`` mapping.
        high_risk_words:         Words that bump risk to 2 (at minimum).
        missing_fields_by_intent: ``{intent: (required_field, ...)}`` — fields
                                   that must be present in task.metadata.
    """

    intents: dict[str, tuple[str, ...]]
    high_risk_words: tuple[str, ...] = ()
    missing_fields_by_intent: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def run(self, task: Task) -> Signal:
        text = task.text.lower()

        # Intent classification — first match wins
        intent = "general"
        for name, keywords in self.intents.items():
            if any(k in text for k in keywords):
                intent = name
                break

        # Risk level
        high_risk_hit = any(w in text for w in self.high_risk_words)
        risk = 2 if high_risk_hit else 1

        # Missing required metadata fields
        required = self.missing_fields_by_intent.get(intent, ())
        missing = tuple(f for f in required if not task.metadata.get(f))
        if missing:
            risk = max(risk, 2)

        return Signal(intent=intent, risk=risk, missing_fields=missing)


# ---------------------------------------------------------------------------
# Rule Policy Agent
# ---------------------------------------------------------------------------

@dataclass
class RulePolicyAgent:
    """Gate which actions are allowed for the current signal.

    Combines base actions with intent-specific actions, then removes
    any that are blocked by conditions derived from task metadata.

    Args:
        base_actions:  Actions always available regardless of intent.
        intent_actions: ``{intent: (additional_action, ...)}`` mapping.
        blocked_when:   ``{action: (condition_name, ...)}`` — conditions
                         that block an action when true.
    """

    base_actions: tuple[str, ...]
    intent_actions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    blocked_when: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # Built-in condition evaluators
    _CONDITIONS: dict[str, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._CONDITIONS = {
            "high_value":        lambda t: float(t.metadata.get("value_usd", 0)) > 250,
            "outside_30_days":   lambda t: int(t.metadata.get("days_since_event", 0)) > 30,
            "production_change": lambda t: bool(t.metadata.get("production_change", False)),
            "no_sources":        lambda t: int(t.metadata.get("source_count", 1)) == 0,
            "external_send":     lambda t: bool(t.metadata.get("external_send", False)),
        }

    def run(self, task: Task, signal: Signal) -> PolicyDecision:
        allowed = set(self.base_actions)
        allowed.update(self.intent_actions.get(signal.intent, ()))

        blocked: list[str] = []
        reason_parts: list[str] = []
        for action, conditions in self.blocked_when.items():
            for cond in conditions:
                evaluator = self._CONDITIONS.get(cond)
                if evaluator and evaluator(task):
                    blocked.append(action)
                    allowed.discard(action)
                    reason_parts.append(f"{action} blocked: {cond}")
                    break

        return PolicyDecision(
            allowed_actions=tuple(sorted(allowed)),
            blocked_actions=tuple(sorted(set(blocked))),
            reason="; ".join(reason_parts) if reason_parts else "standard policy applied",
        )


# ---------------------------------------------------------------------------
# Template Planner Agent
# ---------------------------------------------------------------------------

@dataclass
class TemplatePlannerAgent:
    """Rule-based plan generator using an intent → (action, response) table.

    Used in LangGraph nodes when an LLM is not configured. Falls back to
    ``default_action`` / ``default_response`` when the intent has no template.

    For LLM-powered planning, use ``LangChainPlanner`` from ``llm.py``.

    Args:
        templates:        ``{intent: (action, response)}`` mapping.
        default_action:   Fallback action when intent is not in templates.
        default_response: Fallback response string.
    """

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
        action, response = self.templates.get(
            signal.intent,
            (self.default_action, self.default_response),
        )

        if action not in policy.allowed_actions:
            action = "escalate"
            response = "This request needs review before I can take action."

        citations = tuple(e.title for e in evidence[:2]) if evidence else ()
        confidence = 0.88 if citations else 0.72

        return Plan(
            action=action,
            confidence=confidence,
            response=response,
            internal_note=(
                f"intent={signal.intent} risk={signal.risk} "
                f"evidence_docs={len(evidence)}"
            ),
            citations=citations,
        )


# ---------------------------------------------------------------------------
# Basic Verifier Agent
# ---------------------------------------------------------------------------

@dataclass
class BasicVerifierAgent:
    """Gate that validates a draft plan before it becomes the final action.

    Rejects the plan (approved=False) if:
    - The action is not in policy.allowed_actions
    - The action is substantive but no KB evidence was retrieved
    - The confidence is below the minimum threshold

    Args:
        min_confidence: Minimum confidence to approve. Default: 0.7.
    """

    min_confidence: float = 0.70

    # Actions that are safe to approve even without evidence
    _SAFE_ACTIONS = frozenset({
        "escalate", "ask_for_info", "log_only", "pending_approval",
        "log_note", "answer",
    })

    def run(
        self,
        draft: Plan,
        policy: PolicyDecision,
        evidence: tuple[Evidence, ...],
    ) -> Verification:
        findings: list[str] = []

        if draft.action not in policy.allowed_actions:
            findings.append(
                f"Action '{draft.action}' not in allowed: {policy.allowed_actions}"
            )

        if draft.action not in self._SAFE_ACTIONS and not evidence:
            findings.append("No KB evidence retrieved for a substantive action.")

        if draft.confidence < self.min_confidence:
            findings.append(
                f"Confidence {draft.confidence:.2f} < minimum {self.min_confidence:.2f}"
            )

        approved = not findings
        score = max(0.0, 1.0 - 0.33 * len(findings))
        return Verification(approved=approved, score=score, findings=tuple(findings))
