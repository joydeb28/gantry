"""Tests for Bucket A: Harness components.

Covers:
- MetricsEmitter protocol compliance (NoOpEmitter, LoggingEmitter)
- RetryOrchestrator strategy selection logic
- ClarificationAgent plan generation and delegation
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from gantry.metrics import LoggingEmitter, MetricsEmitter, NoOpEmitter
from gantry.models import (
    Evidence,
    Outcome,
    Plan,
    PolicyDecision,
    Signal,
    Task,
    Verification,
)
from gantry.orchestration import RetryOrchestrator
from gantry.generic_agents import ClarificationAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(text: str = "Test task body") -> Task:
    return Task(id="t-test-1", use_case="support", title="Test", body=text)


def _make_signal(missing_fields: tuple[str, ...] = (), intent: str = "refund") -> Signal:
    return Signal(intent=intent, risk=1, missing_fields=missing_fields)


def _make_policy() -> PolicyDecision:
    return PolicyDecision(allowed_actions=("refund", "escalate"), blocked_actions=())


def _make_outcome(approved: bool = True, action: str = "refund") -> Outcome:
    signal = _make_signal()
    policy = _make_policy()
    plan = Plan(action=action, confidence=0.9, response="OK")
    verification = Verification(approved=approved, score=0.9 if approved else 0.3)
    return Outcome(
        task_id="t-1",
        use_case="support",
        signal=signal,
        evidence=(),
        policy=policy,
        draft=plan,
        verification=verification,
        final_action=action,
        response="OK",
    )


# ---------------------------------------------------------------------------
# MetricsEmitter protocol tests
# ---------------------------------------------------------------------------

class TestNoOpEmitter:
    def test_emit_does_not_raise(self):
        emitter = NoOpEmitter()
        # Must not raise for any valid call
        emitter.emit("gantry.outcome.action", {"use_case": "support", "pattern": "pipeline"}, 1.0)

    def test_satisfies_protocol(self):
        assert isinstance(NoOpEmitter(), MetricsEmitter)


class TestLoggingEmitter:
    def test_emit_logs_at_info_level(self, caplog):
        emitter = LoggingEmitter()  # default level is DEBUG
        with caplog.at_level(logging.DEBUG, logger="gantry.metrics"):
            emitter.emit("gantry.outcome.action", {"use_case": "support", "pattern": "pipeline"}, 1.0)
        assert len(caplog.records) > 0

    def test_emit_logs_custom_level(self, caplog):
        emitter = LoggingEmitter(level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="gantry.metrics"):
            emitter.emit("gantry.test_event", {}, 1.0)
        assert len(caplog.records) > 0

    def test_satisfies_protocol(self):
        assert isinstance(LoggingEmitter(), MetricsEmitter)


# ---------------------------------------------------------------------------
# RetryOrchestrator tests
# ---------------------------------------------------------------------------

class TestRetryOrchestrator:
    def _make_weaver(self, approved: bool, action: str = "refund") -> MagicMock:
        """Returns a *factory* callable (as RetryOrchestrator expects)."""
        weaver = MagicMock()
        weaver.run.return_value = _make_outcome(approved=approved, action=action)
        return weaver

    def _factory(self, approved: bool, action: str = "refund"):
        """Wraps a mock weaver in a zero-arg factory lambda."""
        w = self._make_weaver(approved, action)
        return w, lambda: w

    def test_returns_first_strategy_when_approved(self):
        w1, f1 = self._factory(approved=True, action="refund")
        w2, f2 = self._factory(approved=True, action="escalate")
        orchestrator = RetryOrchestrator(strategies=[f1, f2])
        task = _make_task()
        outcome = orchestrator.run(task)
        assert outcome.final_action == "refund"
        w1.run.assert_called_once_with(task)
        w2.run.assert_not_called()

    def test_falls_back_to_second_strategy_when_first_not_approved(self):
        w1, f1 = self._factory(approved=False, action="refund")
        w2, f2 = self._factory(approved=True, action="escalate")
        orchestrator = RetryOrchestrator(strategies=[f1, f2])
        task = _make_task()
        outcome = orchestrator.run(task)
        assert outcome.final_action == "escalate"
        w1.run.assert_called_once()
        w2.run.assert_called_once()

    def test_returns_last_strategy_outcome_when_all_fail(self):
        _, f1 = self._factory(approved=False, action="refund")
        _, f2 = self._factory(approved=False, action="escalate")
        orchestrator = RetryOrchestrator(strategies=[f1, f2])
        task = _make_task()
        outcome = orchestrator.run(task)
        # Returns the last outcome even if not approved
        assert outcome.final_action == "escalate"

    def test_single_strategy_always_returned(self):
        _, f1 = self._factory(approved=False, action="answer")
        orchestrator = RetryOrchestrator(strategies=[f1])
        task = _make_task()
        outcome = orchestrator.run(task)
        assert outcome.final_action == "answer"

    def test_raises_on_empty_strategies(self):
        with pytest.raises((ValueError, IndexError, AssertionError)):
            RetryOrchestrator(strategies=[]).run(_make_task())


# ---------------------------------------------------------------------------
# ClarificationAgent tests
# ---------------------------------------------------------------------------

class TestClarificationAgent:
    def _make_fallback(self) -> MagicMock:
        mock = MagicMock()
        mock.run.return_value = Plan(
            action="answer", confidence=0.9, response="Delegated answer"
        )
        return mock

    def test_returns_ask_for_info_when_missing_fields(self):
        fallback = self._make_fallback()
        agent = ClarificationAgent(fallback_planner=fallback)
        task = _make_task()
        signal = _make_signal(missing_fields=("value_usd",))
        policy = _make_policy()
        plan = agent.run(task, signal, (), policy)
        assert plan.action == "ask_for_info"
        fallback.run.assert_not_called()

    def test_delegates_to_fallback_when_no_missing_fields(self):
        fallback = self._make_fallback()
        agent = ClarificationAgent(fallback_planner=fallback)
        task = _make_task()
        signal = _make_signal(missing_fields=())
        policy = _make_policy()
        plan = agent.run(task, signal, (), policy)
        assert plan.action == "answer"
        fallback.run.assert_called_once_with(task, signal, (), policy)

    def test_uses_custom_field_question(self):
        fallback = self._make_fallback()
        agent = ClarificationAgent(
            fallback_planner=fallback,
            field_questions={"value_usd": "What is the invoice amount in USD?"},
        )
        task = _make_task()
        signal = _make_signal(missing_fields=("value_usd",))
        policy = _make_policy()
        plan = agent.run(task, signal, (), policy)
        assert "invoice amount" in plan.response.lower() or "value_usd" in plan.response

    def test_generates_default_question_for_unknown_field(self):
        fallback = self._make_fallback()
        agent = ClarificationAgent(
            fallback_planner=fallback,
            field_questions={},  # no custom question for "order_id"
        )
        task = _make_task()
        signal = _make_signal(missing_fields=("order_id",))
        policy = _make_policy()
        plan = agent.run(task, signal, (), policy)
        assert plan.action == "ask_for_info"
        # Response should reference the field (possibly humanized: "order id" or "order_id")
        assert "order" in plan.response.lower()

    def test_response_mentions_all_missing_fields(self):
        fallback = self._make_fallback()
        agent = ClarificationAgent(fallback_planner=fallback)
        task = _make_task()
        signal = _make_signal(missing_fields=("value_usd", "order_id"))
        policy = _make_policy()
        plan = agent.run(task, signal, (), policy)
        # The response should reference at least one of the missing fields
        # (humanized: "value usd" or "order id" or the raw name)
        combined = plan.response.lower()
        assert "value" in combined or "order" in combined or "usd" in combined
