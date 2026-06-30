"""Gantry Memory — short-term conversation buffer for session-aware agents.

Architecture::

    MemoryBackend (Protocol)
        ├── InMemoryBackend    — process-local dict (default; tests and dev)
        ├── RedisBackend       — stub interface (implement when needed)
        └── PostgresBackend    — stub interface (implement when needed)

    Turn (Pydantic model)
        — a single task+outcome pair stored in the buffer

    ConversationBuffer
        — manages a capped window of Turns per session_id
        — serialises to a formatted string for LLM prompt injection

Usage::

    from gantry.memory import ConversationBuffer, InMemoryBackend

    buffer = ConversationBuffer(session_id="user-123", max_turns=10)

    # After a weaver run:
    buffer.add(task, outcome)

    # Before the next LLM call:
    context = buffer.get_context()
    # "Past interactions:\\n[1] intent=replacement action=replace ...\\n"

Integration with LangChainPlanner::

    # weaver_for() wires this automatically when memory= is set:
    weaver = weaver_for("support", memory=buffer)
    outcome = weaver.run(task)
    # buffer.add(task, outcome) is called automatically after each run
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .models import Outcome, Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn — a single stored interaction
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    """A single task + outcome pair stored in the conversation buffer."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    use_case: str
    title: str
    intent: str
    risk: int
    final_action: str
    approved: bool
    response: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_run(cls, task: Task, outcome: Outcome) -> "Turn":
        """Construct a Turn from a completed task/outcome pair."""
        return cls(
            task_id=task.id,
            use_case=task.use_case,
            title=task.title,
            intent=outcome.signal.intent,
            risk=outcome.signal.risk,
            final_action=outcome.final_action,
            approved=outcome.verification.approved,
            response=outcome.response,
        )

    def format(self, index: int) -> str:
        """One-line summary for LLM context injection."""
        return (
            f"[{index}] task={self.task_id} intent={self.intent} "
            f"action={self.final_action} approved={self.approved} "
            f"ts={self.timestamp}"
        )


# ---------------------------------------------------------------------------
# MemoryBackend Protocol + implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for pluggable conversation history storage.

    Any object with ``load`` and ``save`` methods satisfies this protocol.
    """

    def load(self, session_id: str) -> list[Turn]: ...
    def save(self, session_id: str, turns: list[Turn]) -> None: ...


class InMemoryBackend:
    """Process-local in-memory backend. Default for dev and testing.

    Data is lost when the process exits.  Use ``RedisBackend`` or
    ``PostgresBackend`` for multi-instance or durable storage.

    Thread-safety: protected by a module-level dict; suitable for single-
    process async use.  For true multi-thread safety, add a ``threading.Lock``.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}

    def load(self, session_id: str) -> list[Turn]:
        raw = self._store.get(session_id, [])
        return [Turn(**r) for r in raw]

    def save(self, session_id: str, turns: list[Turn]) -> None:
        self._store[session_id] = [t.model_dump() for t in turns]


class RedisBackend:
    """Redis-backed conversation history. Stub — implement when needed.

    Install dependency: ``pip install redis``

    Example::

        backend = RedisBackend(url="redis://localhost:6379/0", ttl_seconds=3600)
        buffer  = ConversationBuffer("user-123", backend=backend)
    """

    def __init__(self, url: str = "redis://localhost:6379/0", ttl_seconds: int = 3600) -> None:
        self._url = url
        self._ttl = ttl_seconds
        self._client: Any = None  # lazy-initialised on first use

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import redis  # type: ignore[import-not-found]
                self._client = redis.from_url(self._url)
            except ImportError as exc:
                raise ImportError(
                    "RedisBackend requires 'redis': pip install redis"
                ) from exc
        return self._client

    def load(self, session_id: str) -> list[Turn]:
        import json
        client = self._get_client()
        raw = client.get(f"gantry:session:{session_id}")
        if raw is None:
            return []
        return [Turn(**item) for item in json.loads(raw)]

    def save(self, session_id: str, turns: list[Turn]) -> None:
        import json
        client = self._get_client()
        data = json.dumps([t.model_dump() for t in turns])
        client.setex(f"gantry:session:{session_id}", self._ttl, data)


class PostgresBackend:
    """PostgreSQL-backed conversation history. Stub — implement when needed.

    Install dependency: ``pip install psycopg2-binary`` or ``asyncpg``.

    Requires a ``gantry_turns`` table::

        CREATE TABLE gantry_turns (
            session_id TEXT NOT NULL,
            turn_index  INT  NOT NULL,
            data        JSONB NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (session_id, turn_index)
        );

    Example::

        backend = PostgresBackend(dsn="postgresql://user:pass@localhost/gantry")
        buffer  = ConversationBuffer("user-123", backend=backend)
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def load(self, session_id: str) -> list[Turn]:
        try:
            import psycopg2  # type: ignore[import-not-found]
            import psycopg2.extras
        except ImportError as exc:
            raise ImportError(
                "PostgresBackend requires 'psycopg2-binary': pip install psycopg2-binary"
            ) from exc
        with psycopg2.connect(self._dsn) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT data FROM gantry_turns WHERE session_id = %s ORDER BY turn_index",
                    (session_id,),
                )
                return [Turn(**row["data"]) for row in cur.fetchall()]

    def save(self, session_id: str, turns: list[Turn]) -> None:
        try:
            import psycopg2  # type: ignore[import-not-found]
            import psycopg2.extras
        except ImportError as exc:
            raise ImportError(
                "PostgresBackend requires 'psycopg2-binary': pip install psycopg2-binary"
            ) from exc
        with psycopg2.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM gantry_turns WHERE session_id = %s", (session_id,))
                for i, turn in enumerate(turns):
                    cur.execute(
                        "INSERT INTO gantry_turns (session_id, turn_index, data) VALUES (%s, %s, %s)",
                        (session_id, i, psycopg2.extras.Json(turn.model_dump())),
                    )
            conn.commit()


# ---------------------------------------------------------------------------
# ConversationBuffer — the main public class
# ---------------------------------------------------------------------------

class ConversationBuffer:
    """Fixed-window conversation history for a single session.

    Stores a capped list of ``Turn`` objects and serialises them into a
    context string suitable for injection into an LLM system prompt.

    Args:
        session_id:  Unique identifier for this conversation. Used as the
                     storage key in the backend.
        max_turns:   Maximum number of turns to retain. When exceeded, the
                     oldest turn is evicted (FIFO). Default: 10.
        backend:     Storage backend. Defaults to ``InMemoryBackend()``.

    Example::

        buffer = ConversationBuffer(session_id="ticket-999", max_turns=5)

        # After weaver.run():
        buffer.add(task, outcome)

        # Before the next LLM call:
        context = buffer.get_context()
        # Inject into the planner:
        plan = planner.plan(task, signal, evidence, policy, session_context=context)
    """

    def __init__(
        self,
        session_id: str,
        max_turns: int = 10,
        backend: MemoryBackend | None = None,
    ) -> None:
        self.session_id = session_id
        self.max_turns = max_turns
        self._backend: MemoryBackend = backend or InMemoryBackend()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, task: Task, outcome: Outcome) -> None:
        """Store a completed task/outcome pair.

        If the buffer is full, the oldest turn is evicted.
        """
        turns = self._backend.load(self.session_id)
        turns.append(Turn.from_run(task, outcome))
        if len(turns) > self.max_turns:
            turns = turns[-self.max_turns :]
        self._backend.save(self.session_id, turns)
        logger.debug(
            "ConversationBuffer[%s]: stored turn %d/%d",
            self.session_id, len(turns), self.max_turns,
        )

    def get_turns(self) -> list[Turn]:
        """Return the stored turns in chronological order."""
        return self._backend.load(self.session_id)

    def get_context(self) -> str:
        """Return a formatted multi-line string for LLM prompt injection.

        Returns an empty string if no turns have been stored yet.

        Example output::

            Past interactions:
            [1] task=SUP-001 intent=replacement action=replace approved=True ts=2026-06-30T...
            [2] task=SUP-002 intent=refund action=escalate approved=False ts=2026-06-30T...
        """
        turns = self._backend.load(self.session_id)
        if not turns:
            return ""
        lines = ["Past interactions:"]
        for i, turn in enumerate(turns, 1):
            lines.append(turn.format(i))
        return "\n".join(lines)

    def clear(self) -> None:
        """Erase all stored turns for this session."""
        self._backend.save(self.session_id, [])
        logger.debug("ConversationBuffer[%s]: cleared", self.session_id)

    def __len__(self) -> int:
        return len(self._backend.load(self.session_id))

    def __repr__(self) -> str:
        return (
            f"ConversationBuffer(session_id={self.session_id!r}, "
            f"max_turns={self.max_turns}, turns={len(self)})"
        )
