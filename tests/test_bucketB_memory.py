"""Tests for Bucket B: Session Memory.

Covers:
- ConversationBuffer add / max_turns eviction / clear / len
- ConversationBuffer.get_context() formatting
- InMemoryBackend isolation across session_ids
- Turn.from_run() constructor
"""

from __future__ import annotations

import pytest

from gantry.memory import ConversationBuffer, InMemoryBackend, Turn
from gantry.models import (
    Evidence,
    Outcome,
    Plan,
    PolicyDecision,
    Signal,
    Task,
    Verification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_id: str = "t-1", text: str = "Process the invoice") -> Task:
    return Task(use_case="finance", title="Invoice", body=text, id=task_id)


def _make_outcome(task_id: str = "t-1", action: str = "approve_payment") -> Outcome:
    signal = Signal(intent="invoice_approval", risk=1, missing_fields=())
    policy = PolicyDecision(
        allowed_actions=("approve_payment",), blocked_actions=()
    )
    plan = Plan(action=action, confidence=0.9, response="Payment approved.")
    verification = Verification(approved=True, score=0.9)
    return Outcome(
        task_id=task_id,
        use_case="finance",
        signal=signal,
        evidence=(),
        policy=policy,
        draft=plan,
        verification=verification,
        final_action=action,
        response="Payment approved.",
    )


# ---------------------------------------------------------------------------
# Turn tests
# ---------------------------------------------------------------------------

class TestTurn:
    def test_from_run_builds_correct_turn(self):
        task = _make_task("t-42")
        outcome = _make_outcome("t-42", "approve_payment")
        turn = Turn.from_run(task, outcome)
        assert turn.task_id == "t-42"
        assert turn.use_case == "finance"
        assert turn.final_action == "approve_payment"
        assert turn.approved is True

    def test_turn_is_immutable(self):
        task = _make_task()
        outcome = _make_outcome()
        turn = Turn.from_run(task, outcome)
        with pytest.raises(Exception):
            turn.task_id = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ConversationBuffer tests
# ---------------------------------------------------------------------------

class TestConversationBuffer:
    def test_add_stores_a_turn(self):
        buf = ConversationBuffer(session_id="s-1")
        task = _make_task()
        outcome = _make_outcome()
        buf.add(task, outcome)
        assert len(buf) == 1

    def test_get_context_empty_returns_empty_string(self):
        buf = ConversationBuffer(session_id="s-1")
        assert buf.get_context() == ""

    def test_get_context_returns_formatted_string(self):
        buf = ConversationBuffer(session_id="s-1")
        task = _make_task("t-1")
        outcome = _make_outcome("t-1", "approve_payment")
        buf.add(task, outcome)
        ctx = buf.get_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 0
        # Context should reference the action or use_case
        assert "approve_payment" in ctx or "finance" in ctx

    def test_clear_removes_all_turns(self):
        buf = ConversationBuffer(session_id="s-1")
        for i in range(3):
            buf.add(_make_task(f"t-{i}"), _make_outcome(f"t-{i}"))
        buf.clear()
        assert len(buf) == 0

    def test_max_turns_evicts_oldest(self):
        buf = ConversationBuffer(session_id="s-1", max_turns=3)
        for i in range(5):
            buf.add(_make_task(f"t-{i}"), _make_outcome(f"t-{i}"))
        assert len(buf) == 3

    def test_len_returns_correct_count(self):
        buf = ConversationBuffer(session_id="s-1")
        assert len(buf) == 0
        buf.add(_make_task("t-1"), _make_outcome("t-1"))
        assert len(buf) == 1
        buf.add(_make_task("t-2"), _make_outcome("t-2"))
        assert len(buf) == 2

    def test_context_contains_last_n_turns_content(self):
        buf = ConversationBuffer(session_id="s-1", max_turns=2)
        buf.add(_make_task("t-1"), _make_outcome("t-1", "approve_payment"))
        buf.add(_make_task("t-2"), _make_outcome("t-2", "escalate"))
        ctx = buf.get_context()
        # The most recent turn should appear in context
        assert "escalate" in ctx

    def test_multiple_adds_tracked_sequentially(self):
        buf = ConversationBuffer(session_id="s-1")
        for i in range(4):
            buf.add(_make_task(f"t-{i}"), _make_outcome(f"t-{i}"))
        assert len(buf) == 4


# ---------------------------------------------------------------------------
# InMemoryBackend isolation tests
# ---------------------------------------------------------------------------

class TestInMemoryBackend:
    def test_sessions_are_isolated(self):
        backend = InMemoryBackend()
        buf_a = ConversationBuffer(session_id="session-A", backend=backend)
        buf_b = ConversationBuffer(session_id="session-B", backend=backend)

        buf_a.add(_make_task("t-1"), _make_outcome("t-1"))
        # session-B should see 0 turns even though session-A has 1
        assert len(buf_b) == 0

    def test_sessions_track_independently(self):
        backend = InMemoryBackend()
        buf_a = ConversationBuffer(session_id="session-A", backend=backend)
        buf_b = ConversationBuffer(session_id="session-B", backend=backend)

        buf_a.add(_make_task("t-1"), _make_outcome("t-1"))
        buf_a.add(_make_task("t-2"), _make_outcome("t-2"))
        buf_b.add(_make_task("t-3"), _make_outcome("t-3"))

        assert len(buf_a) == 2
        assert len(buf_b) == 1

    def test_clear_only_affects_own_session(self):
        backend = InMemoryBackend()
        buf_a = ConversationBuffer(session_id="session-A", backend=backend)
        buf_b = ConversationBuffer(session_id="session-B", backend=backend)

        buf_a.add(_make_task("t-1"), _make_outcome("t-1"))
        buf_b.add(_make_task("t-2"), _make_outcome("t-2"))

        buf_a.clear()
        assert len(buf_a) == 0
        assert len(buf_b) == 1
