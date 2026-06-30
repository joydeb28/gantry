"""Gantry Triggers — proactive task execution from external events.

Provides three trigger types that autonomously call ``weaver.run(task)``
and dispatch outcomes to a user-supplied callback:

* ``ScheduledTrigger``  — cron-based recurring execution (APScheduler)
* ``PollingTrigger``    — async interval polling with a condition filter
* ``WebhookTrigger``    — HTTP POST handler (aiohttp, lazy import)
* ``TriggerRunner``     — manages multiple triggers in one event loop

All trigger types share a common design contract:
* ``start(...)`` — register and activate the trigger
* ``stop(...)``  — cleanly deactivate and release resources
* ``on_outcome`` — callback ``(Outcome) -> None``; defaults to logging
"""

from __future__ import annotations

import asyncio
import logging
import signal as _signal
from typing import Any, Callable

from .models import Outcome, Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ScheduledTrigger
# ---------------------------------------------------------------------------

class ScheduledTrigger:
    """Run ``weaver.run(task)`` on a cron schedule via APScheduler.

    Requires ``apscheduler>=3.10``.  Install with::

        pip install 'gantry[prod]'

    Args:
        cron:         Cron expression string, e.g. ``'0 8 * * 1-5'``
                      (08:00 Monday–Friday).
        task_factory: Zero-argument callable returning a fresh ``Task``
                      for each scheduled run.
        weaver:       Any weaver object with ``run(task: Task) -> Outcome``.
        on_outcome:   Callback invoked with each ``Outcome``.
                      Defaults to logging the outcome at INFO level.
        job_id:       Unique APScheduler job ID.  Defaults to
                      ``'gantry_<ClassName>'``.

    Example::

        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from gantry.triggers import ScheduledTrigger

        trigger = ScheduledTrigger(
            cron='0 8 * * 1-5',
            task_factory=lambda: Task(use_case='finance', ...),
            weaver=weaver_for('finance'),
        )
        scheduler = AsyncIOScheduler()
        trigger.start(scheduler)
        scheduler.start()
    """

    def __init__(
        self,
        cron: str,
        task_factory: Callable[[], Task],
        weaver: Any,
        on_outcome: Callable[[Outcome], None] | None = None,
        job_id: str | None = None,
    ) -> None:
        self.cron = cron
        self.task_factory = task_factory
        self.weaver = weaver
        self.on_outcome = on_outcome or _default_on_outcome
        self.job_id = job_id or f"gantry_{type(weaver).__name__}"

    def _execute(self) -> None:
        """Called by APScheduler; runs the weaver and dispatches the outcome."""
        try:
            task = self.task_factory()
            logger.info("ScheduledTrigger[%s]: running task %s", self.job_id, task.id)
            outcome = self.weaver.run(task)
            self.on_outcome(outcome)
        except Exception:  # noqa: BLE001
            logger.exception(
                "ScheduledTrigger[%s]: unhandled error during execution", self.job_id
            )

    def start(self, scheduler: Any) -> None:
        """Register this trigger with an APScheduler ``AsyncIOScheduler``.

        Args:
            scheduler: An ``apscheduler.schedulers.asyncio.AsyncIOScheduler``
                       instance (must not yet have been started or must already
                       be running).
        """
        try:
            from apscheduler.triggers.cron import CronTrigger  # lazy import
        except ImportError as exc:
            raise ImportError(
                "ScheduledTrigger requires APScheduler.  "
                "Install with: pip install 'gantry[prod]'"
            ) from exc

        scheduler.add_job(
            self._execute,
            trigger=CronTrigger.from_crontab(self.cron),
            id=self.job_id,
            replace_existing=True,
            misfire_grace_time=60,
        )
        logger.info(
            "ScheduledTrigger[%s]: registered with cron='%s'", self.job_id, self.cron
        )

    def stop(self, scheduler: Any) -> None:
        """Remove this trigger's job from the scheduler.

        Args:
            scheduler: The same ``AsyncIOScheduler`` passed to ``start()``.
        """
        try:
            scheduler.remove_job(self.job_id)
            logger.info("ScheduledTrigger[%s]: removed from scheduler", self.job_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "ScheduledTrigger[%s]: could not remove job (may already be gone)",
                self.job_id,
            )


# ---------------------------------------------------------------------------
# PollingTrigger
# ---------------------------------------------------------------------------

class PollingTrigger:
    """Poll a data source at a fixed interval and run ``weaver.run(task)``
    for each item that matches a condition filter.

    Uses pure ``asyncio.sleep``; no extra dependencies required.

    Args:
        interval_seconds: How often to poll the ``source`` callable.
        source:           Zero-argument callable returning a ``list`` of items
                          to inspect each interval.
        condition:        Predicate ``(item) -> bool``; only matching items
                          trigger a weaver run.
        task_factory:     ``(item) -> Task``; builds the task for a matched item.
        weaver:           Any object with ``run(task: Task) -> Outcome``.
        on_outcome:       Callback ``(Outcome) -> None``.  Defaults to logging.

    Example::

        trigger = PollingTrigger(
            interval_seconds=60,
            source=lambda: db.fetch_pending_invoices(),
            condition=lambda inv: inv['amount_usd'] > 10_000,
            task_factory=lambda inv: Task(use_case='finance', ...),
            weaver=weaver_for('finance'),
        )
        asyncio.run(trigger.start())
    """

    def __init__(
        self,
        interval_seconds: int,
        source: Callable[[], list[Any]],
        condition: Callable[[Any], bool],
        task_factory: Callable[[Any], Task],
        weaver: Any,
        on_outcome: Callable[[Outcome], None] | None = None,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.source = source
        self.condition = condition
        self.task_factory = task_factory
        self.weaver = weaver
        self.on_outcome = on_outcome or _default_on_outcome
        self._task: asyncio.Task | None = None

    async def _poll_loop(self) -> None:
        """Main async polling loop.  Runs until cancelled."""
        logger.info(
            "PollingTrigger: starting loop (interval=%ds)", self.interval_seconds
        )
        while True:
            try:
                items = self.source()
                for item in items:
                    if not self.condition(item):
                        continue
                    try:
                        task = self.task_factory(item)
                        logger.info("PollingTrigger: matched item → task %s", task.id)
                        outcome = self.weaver.run(task)
                        self.on_outcome(outcome)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "PollingTrigger: error processing item %r", item
                        )
            except asyncio.CancelledError:
                logger.info("PollingTrigger: poll loop cancelled")
                raise
            except Exception:  # noqa: BLE001
                logger.exception("PollingTrigger: error fetching from source")
            await asyncio.sleep(self.interval_seconds)

    def start(self) -> "asyncio.Task[None]":
        """Schedule the polling loop as an asyncio background task.

        Must be called inside a running event loop (e.g. inside ``async def``
        or via ``asyncio.run()``).

        Returns:
            The ``asyncio.Task`` wrapping the poll loop.  Retain a reference
            to cancel it later via ``stop()``.
        """
        self._task = asyncio.ensure_future(self._poll_loop())
        return self._task

    def stop(self) -> None:
        """Cancel the background polling task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            logger.info("PollingTrigger: poll loop stop requested")


# ---------------------------------------------------------------------------
# WebhookTrigger
# ---------------------------------------------------------------------------

class WebhookTrigger:
    """Accept an HTTP POST and run ``weaver.run(task)`` for each request.

    Requires ``aiohttp``::

        pip install aiohttp

    Args:
        route:        HTTP path to listen on, e.g. ``'/hooks/zendesk'``.
        task_factory: ``(body: dict) -> Task``; called with the parsed JSON body.
        weaver:       Any object with ``run(task: Task) -> Outcome``.
        on_outcome:   Callback ``(Outcome) -> None``.  Defaults to logging.
        port:         HTTP server port.  Default: ``8080``.

    Example::

        import asyncio
        from gantry.triggers import WebhookTrigger

        trigger = WebhookTrigger(
            route='/hooks/support',
            task_factory=lambda body: Task(use_case='support', ...),
            weaver=weaver_for('support'),
            port=9000,
        )
        asyncio.run(trigger.start())
    """

    def __init__(
        self,
        route: str,
        task_factory: Callable[[dict], Task],
        weaver: Any,
        on_outcome: Callable[[Outcome], None] | None = None,
        port: int = 8080,
    ) -> None:
        self.route = route
        self.task_factory = task_factory
        self.weaver = weaver
        self.on_outcome = on_outcome or _default_on_outcome
        self.port = port
        self._runner: Any = None
        self._site: Any = None

    async def start(self) -> None:
        """Start the aiohttp HTTP server.  Blocks until ``stop()`` is called."""
        try:
            from aiohttp import web  # lazy import
        except ImportError as exc:
            raise ImportError(
                "WebhookTrigger requires aiohttp.  Install with: pip install aiohttp"
            ) from exc

        app = web.Application()

        async def _handler(request: Any) -> Any:
            try:
                body = await request.json()
            except Exception:  # noqa: BLE001
                logger.warning("WebhookTrigger[%s]: failed to parse JSON body", self.route)
                return web.Response(status=400, text="Invalid JSON body")
            try:
                task = self.task_factory(body)
                logger.info(
                    "WebhookTrigger[%s]: received request → task %s", self.route, task.id
                )
                outcome = self.weaver.run(task)
                self.on_outcome(outcome)
                return web.json_response(
                    {
                        "status": "ok",
                        "task_id": task.id,
                        "final_action": outcome.final_action,
                    }
                )
            except Exception:  # noqa: BLE001
                logger.exception("WebhookTrigger[%s]: error handling request", self.route)
                return web.Response(status=500, text="Internal server error")

        app.router.add_post(self.route, _handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await self._site.start()
        logger.info(
            "WebhookTrigger: listening on http://0.0.0.0:%d%s", self.port, self.route
        )

    async def stop(self) -> None:
        """Cleanly shut down the aiohttp server."""
        if self._runner is not None:
            await self._runner.cleanup()
            logger.info("WebhookTrigger[%s]: server stopped", self.route)


# ---------------------------------------------------------------------------
# TriggerRunner
# ---------------------------------------------------------------------------

class TriggerRunner:
    """Manage multiple triggers in a single asyncio event loop.

    Handles graceful shutdown on ``SIGTERM`` / ``SIGINT``.

    Supported trigger types:

    * ``ScheduledTrigger`` — uses an ``AsyncIOScheduler``; TriggerRunner starts
      and stops the scheduler automatically.
    * ``PollingTrigger``   — started via ``trigger.start()``; TriggerRunner
      cancels via ``trigger.stop()``.
    * ``WebhookTrigger``   — started via ``await trigger.start()``; TriggerRunner
      stops via ``await trigger.stop()``.

    Example::

        runner = TriggerRunner([
            ScheduledTrigger('0 8 * * 1-5', task_factory, weaver),
            PollingTrigger(60, source, condition, task_factory, weaver),
        ])
        asyncio.run(runner.run())
    """

    def __init__(self, triggers: list[Any]) -> None:
        self.triggers = triggers

    async def run(self) -> None:
        """Start all triggers and block until a shutdown signal is received."""
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _shutdown(*_: Any) -> None:
            logger.info("TriggerRunner: shutdown signal received")
            stop_event.set()

        for sig in (_signal.SIGTERM, _signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _shutdown)
            except (NotImplementedError, RuntimeError):
                # Windows or environments without signal support
                pass

        scheduler: Any = None
        scheduled_triggers: list[ScheduledTrigger] = []

        # Separate ScheduledTriggers (need a shared APScheduler instance)
        other_triggers: list[Any] = []
        for t in self.triggers:
            if isinstance(t, ScheduledTrigger):
                scheduled_triggers.append(t)
            else:
                other_triggers.append(t)

        # Start APScheduler if any ScheduledTriggers exist
        if scheduled_triggers:
            try:
                from apscheduler.schedulers.asyncio import AsyncIOScheduler
                scheduler = AsyncIOScheduler()
                for t in scheduled_triggers:
                    t.start(scheduler)
                scheduler.start()
                logger.info(
                    "TriggerRunner: APScheduler started with %d cron job(s)",
                    len(scheduled_triggers),
                )
            except ImportError as exc:
                raise ImportError(
                    "ScheduledTrigger requires APScheduler.  "
                    "Install with: pip install 'gantry[prod]'"
                ) from exc

        # Start PollingTriggers and WebhookTriggers
        for t in other_triggers:
            if isinstance(t, PollingTrigger):
                t.start()
                logger.info("TriggerRunner: PollingTrigger started (interval=%ds)", t.interval_seconds)
            elif isinstance(t, WebhookTrigger):
                await t.start()
                logger.info(
                    "TriggerRunner: WebhookTrigger started on port %d route %s",
                    t.port, t.route,
                )
            else:
                # Generic trigger — try start() without args
                try:
                    result = t.start()
                    if asyncio.iscoroutine(result):
                        await result
                    logger.info("TriggerRunner: generic trigger %r started", t)
                except Exception:  # noqa: BLE001
                    logger.exception("TriggerRunner: failed to start trigger %r", t)

        logger.info(
            "TriggerRunner: %d trigger(s) running.  Waiting for shutdown signal...",
            len(self.triggers),
        )

        try:
            await stop_event.wait()
        finally:
            logger.info("TriggerRunner: stopping all triggers...")

            # Stop PollingTriggers and WebhookTriggers
            for t in other_triggers:
                try:
                    if isinstance(t, PollingTrigger):
                        t.stop()
                    elif isinstance(t, WebhookTrigger):
                        await t.stop()
                    else:
                        result = t.stop()
                        if asyncio.iscoroutine(result):
                            await result
                except Exception:  # noqa: BLE001
                    logger.exception("TriggerRunner: error stopping trigger %r", t)

            # Stop APScheduler
            if scheduler is not None:
                scheduler.shutdown(wait=False)
                logger.info("TriggerRunner: APScheduler shut down")

            logger.info("TriggerRunner: all triggers stopped")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_on_outcome(outcome: Outcome) -> None:
    """Default outcome handler — logs the result at INFO level."""
    logger.info(
        "Trigger outcome: task_id=%s use_case=%s action=%s approved=%s score=%.2f",
        outcome.task_id,
        outcome.use_case,
        outcome.final_action,
        outcome.verification.approved,
        outcome.verification.score,
    )
