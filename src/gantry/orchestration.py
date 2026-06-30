"""Gantry Orchestration — higher-level wrappers around pattern weavers.

Provides orchestration primitives that sit *above* individual LangGraph
weavers and coordinate across runs or strategies:

* ``RetryOrchestrator`` — runs weavers in a strategy sequence, escalating
  to the next strategy when ``verification.approved`` is False.

These classes deliberately expose the same ``run(task) -> Outcome`` interface
as every weaver, so they are composable and drop-in replaceable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .models import Outcome, Task

logger = logging.getLogger(__name__)


class RetryOrchestrator:
    """Run a task through a sequence of weavers, escalating on failure.

    Each element in ``strategies`` is a zero-argument factory that returns a
    weaver.  Factories are called lazily — the second strategy is only
    instantiated if the first one's outcome is not approved.

    This keeps each individual weaver simple and focuses retry logic in one
    place outside the LangGraph graph.

    Args:
        strategies: Ordered list of ``Callable[[], weaver]`` factories.
                    Must contain at least one entry.
        stop_on_approved: If True (default), returns as soon as any strategy
                          produces an approved outcome. If False, runs all
                          strategies and returns the last outcome.

    Raises:
        ValueError: If ``strategies`` is empty.

    Example::

        from gantry.orchestration import RetryOrchestrator
        from gantry.scenarios import weaver_for

        orchestrator = RetryOrchestrator(
            strategies=[
                # Strategy 1: fast template planner (no LLM cost)
                lambda: weaver_for("support", planner_type="template"),
                # Strategy 2: LLM planner for richer reasoning
                lambda: weaver_for("support", planner_type="openai"),
                # Strategy 3: always produces an escalation (safety net)
                lambda: weaver_for("support", planner_type="template"),
            ]
        )
        outcome = orchestrator.run(task)

    Notes:
        - Each factory is called fresh for every ``run()`` invocation —
          weavers are not shared between runs or between strategies.
        - If all strategies fail (none produce ``approved=True``), the last
          outcome is returned so the caller always gets a complete ``Outcome``.
        - Metrics and memory args can be baked into each factory closure.
    """

    def __init__(
        self,
        strategies: list[Callable[[], Any]],
        stop_on_approved: bool = True,
    ) -> None:
        if not strategies:
            raise ValueError("RetryOrchestrator requires at least one strategy.")
        self._strategies = strategies
        self._stop_on_approved = stop_on_approved

    def run(self, task: Task) -> Outcome:
        """Run the task through strategies in order.

        Args:
            task: The input task.

        Returns:
            The first approved ``Outcome``, or the last ``Outcome`` if no
            strategy produces an approved result.
        """
        outcome: Outcome | None = None

        for i, factory in enumerate(self._strategies):
            logger.info(
                "RetryOrchestrator: trying strategy %d/%d for task '%s'",
                i + 1, len(self._strategies), task.id,
            )
            weaver = factory()
            outcome = weaver.run(task)

            if outcome.verification.approved:
                logger.info(
                    "RetryOrchestrator: strategy %d approved — returning outcome "
                    "(action=%s, confidence=%.2f)",
                    i + 1, outcome.final_action, outcome.draft.confidence,
                )
                return outcome

            logger.warning(
                "RetryOrchestrator: strategy %d did not approve "
                "(action=%s, findings=%s)",
                i + 1, outcome.final_action,
                "; ".join(outcome.verification.findings),
            )

            if not self._stop_on_approved:
                continue

        # All strategies exhausted — return the last outcome regardless
        logger.warning(
            "RetryOrchestrator: all %d strategies exhausted for task '%s'; "
            "returning last outcome (action=%s)",
            len(self._strategies), task.id,
            outcome.final_action if outcome else "none",
        )
        return outcome  # type: ignore[return-value]  # always set — strategies is non-empty
