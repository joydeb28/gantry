"""Gantry Metrics — emitter protocol and built-in implementations.

Usage::

    from gantry.metrics import LoggingEmitter, NoOpEmitter
    from gantry.scenarios import weaver_for

    # Dev: log metrics to Python logger
    weaver = weaver_for("support", metrics=LoggingEmitter())

    # Production: plug in Prometheus or Datadog
    class PrometheusEmitter:
        def emit(self, event: str, labels: dict[str, str], value: float) -> None:
            counter = self._registry.get_or_create(event, labels.keys())
            counter.labels(**labels).inc(value)

    weaver = weaver_for("support", metrics=PrometheusEmitter(...))

Every ``run(task) -> Outcome`` call emits three standard events:

* ``gantry.outcome.action``  — labels: use_case, pattern, action, approved
* ``gantry.outcome.count``   — labels: use_case, pattern (value=1, for counting)
* ``gantry.verification``    — labels: use_case, approved (value=verification.score)
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetricsEmitter Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MetricsEmitter(Protocol):
    """Protocol satisfied by any metrics backend.

    Any object with an ``emit(event, labels, value)`` method is a valid
    emitter — no inheritance required.

    Args:
        event:  Metric name, e.g. ``"gantry.outcome.action"``.
        labels: Key/value string tags for dimensions, e.g.
                ``{"use_case": "support", "action": "replace"}``.
        value:  Numeric metric value. Use ``1.0`` for counters.
    """

    def emit(self, event: str, labels: dict[str, str], value: float) -> None: ...


# ---------------------------------------------------------------------------
# Built-in emitter implementations
# ---------------------------------------------------------------------------

class NoOpEmitter:
    """Default emitter — silently discards all metrics. Zero overhead.

    This is the default everywhere so that adding metrics to a weaver
    is strictly opt-in and never impacts existing callers.
    """

    def emit(self, event: str, labels: dict[str, str], value: float) -> None:
        pass


class LoggingEmitter:
    """Emitter that logs metrics via the Python ``logging`` module.

    Useful in development and testing — no external dependencies.

    Args:
        level:  Python logging level for metric lines. Default: ``DEBUG``.
        logger_name: Logger name. Defaults to ``"gantry.metrics"``.

    Example::

        weaver = weaver_for("support", metrics=LoggingEmitter(level=logging.INFO))
        weaver.run(task)
        # Logs: metric gantry.outcome.action value=1.0000 use_case=support action=replace approved=True
    """

    def __init__(
        self,
        level: int = logging.DEBUG,
        logger_name: str = "gantry.metrics",
    ) -> None:
        self._level = level
        self._logger = logging.getLogger(logger_name)

    def emit(self, event: str, labels: dict[str, str], value: float) -> None:
        label_str = " ".join(f"{k}={v}" for k, v in labels.items())
        self._logger.log(
            self._level,
            "metric %s value=%.4f %s",
            event, value, label_str,
        )


# ---------------------------------------------------------------------------
# Shared emit helper used by pattern weavers
# ---------------------------------------------------------------------------

def emit_outcome(
    emitter: Any,
    use_case: str,
    pattern: str,
    final_action: str,
    approved: bool,
    verification_score: float,
) -> None:
    """Emit the three standard outcome metrics from any weaver's ``run()`` method.

    This is a module-level helper so each pattern weaver doesn't duplicate
    the same three emit() calls. Called unconditionally — ``NoOpEmitter``
    makes it zero-cost when no emitter is configured.

    Args:
        emitter:            Any ``MetricsEmitter``-compatible object.
        use_case:           e.g. ``"support"``.
        pattern:            e.g. ``"pipeline"``.
        final_action:       The outcome's ``final_action`` field.
        approved:           The outcome's ``verification.approved`` field.
        verification_score: The outcome's ``verification.score`` field.
    """
    base = {"use_case": use_case, "pattern": pattern}
    emitter.emit("gantry.outcome.count",  {**base}, 1.0)
    emitter.emit("gantry.outcome.action", {**base, "action": final_action, "approved": str(approved)}, 1.0)
    emitter.emit("gantry.verification",   {**base, "approved": str(approved)}, verification_score)
