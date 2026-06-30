"""Tests for Bucket C: Autonomy / Triggers.

Covers:
- PollingTrigger._poll_loop condition filtering
- PollingTrigger._poll_loop on_outcome dispatching
- PollingTrigger._poll_loop error isolation (one bad item doesn't stop the loop)
- TriggerRunner.run() starts all triggers
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from gantry.models import (
    Outcome,
    Plan,
    PolicyDecision,
    Signal,
    Task,
    Verification,
)
from gantry.triggers import PollingTrigger, TriggerRunner, ScheduledTrigger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outcome(action: str = "refund") -> Outcome:
    signal = Signal(intent="refund", risk=1, missing_fields=())
    policy = PolicyDecision(allowed_actions=("refund",), blocked_actions=())
    plan = Plan(action=action, confidence=0.9, response="Done")
    verification = Verification(approved=True, score=0.9)
    return Outcome(
        task_id="t-poll-1",
        use_case="support",
        signal=signal,
        evidence=(),
        policy=policy,
        draft=plan,
        verification=verification,
        final_action=action,
        response="Done",
    )


def _make_task_from_item(item: dict) -> Task:
    return Task(
        id=f"poll-{item.get('id', 'x')}",
        use_case="support",
        title=item.get("title", "Poll task"),
        body=item.get("body", ""),
    )


# ---------------------------------------------------------------------------
# PollingTrigger tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPollingTrigger:
    async def test_run_called_only_for_matching_items(self):
        items = [
            {"id": 1, "urgent": True},
            {"id": 2, "urgent": False},   # should be skipped
            {"id": 3, "urgent": True},
        ]
        weaver = MagicMock()
        weaver.run.return_value = _make_outcome()
        outcomes = []

        trigger = PollingTrigger(
            interval_seconds=9999,
            source=lambda: items,
            condition=lambda item: item["urgent"],
            task_factory=_make_task_from_item,
            weaver=weaver,
            on_outcome=outcomes.append,
        )

        # Run exactly one iteration then cancel
        async def _run_once():
            task = asyncio.ensure_future(trigger._poll_loop())
            await asyncio.sleep(0)   # yield so poll_loop starts
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await _run_once()

        # Only 2 items match condition
        assert weaver.run.call_count == 2
        assert len(outcomes) == 2

    async def test_on_outcome_called_for_each_match(self):
        items = [{"id": 1, "flag": True}, {"id": 2, "flag": True}]
        weaver = MagicMock()
        weaver.run.return_value = _make_outcome()
        received = []

        trigger = PollingTrigger(
            interval_seconds=9999,
            source=lambda: items,
            condition=lambda item: item["flag"],
            task_factory=_make_task_from_item,
            weaver=weaver,
            on_outcome=received.append,
        )

        async def _run_once():
            t = asyncio.ensure_future(trigger._poll_loop())
            await asyncio.sleep(0)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        await _run_once()
        assert len(received) == 2

    async def test_skips_non_matching_items(self):
        items = [
            {"id": 1, "flag": False},
            {"id": 2, "flag": False},
        ]
        weaver = MagicMock()
        weaver.run.return_value = _make_outcome()

        trigger = PollingTrigger(
            interval_seconds=9999,
            source=lambda: items,
            condition=lambda item: item["flag"],
            task_factory=_make_task_from_item,
            weaver=weaver,
            on_outcome=lambda _: None,
        )

        async def _run_once():
            t = asyncio.ensure_future(trigger._poll_loop())
            await asyncio.sleep(0)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        await _run_once()
        weaver.run.assert_not_called()

    async def test_error_in_weaver_does_not_stop_loop(self):
        """If weaver.run() raises for one item, the loop must continue."""
        call_log: list[int] = []

        def side_effect(task: Task) -> Outcome:
            # First call raises, second succeeds
            if len(call_log) == 0:
                call_log.append(1)
                raise RuntimeError("simulated failure")
            call_log.append(2)
            return _make_outcome()

        items = [{"id": 1, "flag": True}, {"id": 2, "flag": True}]
        weaver = MagicMock()
        weaver.run.side_effect = side_effect
        received = []

        trigger = PollingTrigger(
            interval_seconds=9999,
            source=lambda: items,
            condition=lambda item: item["flag"],
            task_factory=_make_task_from_item,
            weaver=weaver,
            on_outcome=received.append,
        )

        async def _run_once():
            t = asyncio.ensure_future(trigger._poll_loop())
            await asyncio.sleep(0)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        await _run_once()

        # Both items were attempted; second one succeeded
        assert weaver.run.call_count == 2
        assert len(received) == 1  # only the successful one


# ---------------------------------------------------------------------------
# TriggerRunner tests
# ---------------------------------------------------------------------------

class TestTriggerRunner:
    def test_run_starts_polling_triggers(self):
        """TriggerRunner.run() calls start() on PollingTriggers."""
        mock_poll_trigger = MagicMock(spec=PollingTrigger)
        mock_poll_trigger.interval_seconds = 60
        mock_poll_trigger.start.return_value = MagicMock()  # asyncio.Task mock

        runner = TriggerRunner([mock_poll_trigger])

        async def _run_and_stop():
            run_task = asyncio.ensure_future(runner.run())
            await asyncio.sleep(0)  # let runner reach stop_event.wait()
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass

        asyncio.run(_run_and_stop())
        mock_poll_trigger.start.assert_called_once()

    def test_run_stops_polling_triggers_on_exit(self):
        """TriggerRunner calls stop() on PollingTriggers during shutdown."""
        mock_poll_trigger = MagicMock(spec=PollingTrigger)
        mock_poll_trigger.interval_seconds = 60
        mock_poll_trigger.start.return_value = MagicMock()

        runner = TriggerRunner([mock_poll_trigger])

        async def _run_and_stop():
            run_task = asyncio.ensure_future(runner.run())
            await asyncio.sleep(0)
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass

        asyncio.run(_run_and_stop())
        mock_poll_trigger.stop.assert_called_once()
