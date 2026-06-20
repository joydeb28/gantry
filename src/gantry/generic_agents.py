"""Generic agent components used across all LangGraph patterns.

Each class is a configurable, stateless agent that performs one step of
the agentic pipeline. They are used as node functions inside LangGraph
StateGraphs — each node calls one agent and returns the result as a
state update dict.

Agents:
    KeywordSignalAgent   : Keyword-based intent + risk extraction
    SemanticSignalAgent  : Embedding-based intent detection (more robust)
    RulePolicyAgent      : Rule-based action gating (supports custom conditions)
    TemplatePlannerAgent : Template-based plan generation (no LLM)
    BasicVerifierAgent   : Policy + confidence + evidence gate

Utilities:
    safe_node            : Wraps a LangGraph node function to catch exceptions
                           and return a safe fallback state update instead of
                           crashing the entire graph run.

LLM-powered planning is in llm.py (LangChainPlanner).
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from .models import Evidence, Plan, PolicyDecision, Signal, Task, Verification

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe-node wrapper
# ---------------------------------------------------------------------------

def safe_node(fn: Callable, fallback: dict | None = None) -> Callable:
    """Wrap a LangGraph node function so exceptions are caught gracefully.

    If the wrapped node raises any exception, it logs the error and returns
    a **copy** of ``fallback`` instead of propagating the crash through the
    graph.

    Design notes:
        - Returns ``copy.deepcopy(fallback)`` on every failure — a full deep
          copy — so callers cannot corrupt the fallback by mutating returned
          lists or dicts, even nested ones (e.g., ``audit_trail`` lists).
        - Re-raises ``KeyboardInterrupt`` and ``SystemExit`` unconditionally
          so that process signals and interpreter shutdowns are never swallowed.

    Args:
        fn:       The node function to wrap.
        fallback: State update dict to return on failure. Defaults to ``{}``.
                  A defensive copy is taken at wrap time and again on every
                  failure invocation.

    Returns:
        A wrapped callable with the same signature as ``fn``.

    Example::

        builder.add_node("signal", safe_node(self._signal_node, {"audit_trail": ["signal:error"]}))
    """
    _fallback: dict = dict(fallback) if fallback else {}

    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise  # never swallow process signals
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "safe_node: '%s' failed — %s: %s",
                fn.__qualname__, type(exc).__name__, exc,
                exc_info=True,
            )
            return copy.deepcopy(_fallback)  # deep copy — nested lists/dicts are also isolated

    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    return wrapper


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
# Semantic Signal Agent
# ---------------------------------------------------------------------------

@dataclass
class SemanticSignalAgent:
    """Classify a task intent using embedding cosine similarity.

    More robust than keyword matching — handles natural-language paraphrases
    (e.g. "I want my money back" matches the ``refund`` intent even without
    the literal word "refund").

    Uses the same ``BAAI/bge-small-en-v1.5`` model already installed by
    ``KnowledgeBaseRetriever`` — fully offline, no API key required.

    Falls back to ``KeywordSignalAgent`` behaviour if the embedding model
    cannot be loaded.

    Args:
        intent_exemplars:        ``{intent_name: (example_sentence, ...)}``
                                  mapping. The agent picks the intent whose
                                  exemplars have the highest average cosine
                                  similarity to the task text.
        high_risk_words:         Words that bump risk to 2 (at minimum).
        missing_fields_by_intent: ``{intent: (required_field, ...)}`` — fields
                                   that must be present in task.metadata.
        similarity_threshold:    Minimum similarity to classify as an intent
                                  (vs. falling back to ``"general"``). Default: 0.35.
        model_name:              HuggingFace embedding model. Default: bge-small-en-v1.5.

    Example::

        agent = SemanticSignalAgent(
            intent_exemplars={
                "refund": ("I want a refund", "charge me back", "return my money"),
                "replacement": ("send a new item", "broken product", "defective device"),
            },
            high_risk_words=("angry", "lawsuit"),
        )
        signal = agent.run(task)
    """

    intent_exemplars: dict[str, tuple[str, ...]]
    high_risk_words: tuple[str, ...] = ()
    missing_fields_by_intent: dict[str, tuple[str, ...]] = field(default_factory=dict)
    similarity_threshold: float = 0.35
    model_name: str = "BAAI/bge-small-en-v1.5"

    # Lazily initialised — set in __post_init__ only if model loads successfully
    _embed_fn: Callable[[list[str]], list[list[float]]] | None = field(
        init=False, repr=False, default=None
    )
    _exemplar_vecs: dict[str, list[list[float]]] | None = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self) -> None:
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            _model = HuggingFaceEmbedding(model_name=self.model_name)

            def _embed(texts: list[str]) -> list[list[float]]:
                return [_model.get_text_embedding(t) for t in texts]

            self._embed_fn = _embed
            # Pre-compute exemplar embeddings once at construction time
            self._exemplar_vecs = {}
            for intent, exemplars in self.intent_exemplars.items():
                self._exemplar_vecs[intent] = _embed(list(exemplars))
            logger.info(
                "SemanticSignalAgent: loaded model '%s', %d intents indexed.",
                self.model_name, len(self.intent_exemplars),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SemanticSignalAgent: could not load embedding model (%s). "
                "Falling back to keyword matching.",
                exc,
            )

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def run(self, task: Task) -> Signal:
        text = task.text.lower()

        # Risk and missing-field logic is the same regardless of embedding availability
        high_risk_hit = any(w in text for w in self.high_risk_words)
        risk = 2 if high_risk_hit else 1

        # --- Intent classification ---
        intent = "general"
        if self._embed_fn is not None and self._exemplar_vecs:
            # Embed the task text and compute similarity to each intent's exemplars
            query_vec = self._embed_fn([task.text])[0]
            best_score = 0.0
            for candidate_intent, vecs in self._exemplar_vecs.items():
                avg_sim = sum(self._cosine(query_vec, v) for v in vecs) / len(vecs)
                if avg_sim > best_score:
                    best_score = avg_sim
                    intent = candidate_intent
            if best_score < self.similarity_threshold:
                intent = "general"
                logger.debug(
                    "SemanticSignalAgent: best similarity %.3f below threshold %.3f → 'general'",
                    best_score, self.similarity_threshold,
                )
        else:
            # Keyword fallback
            for name, exemplars in self.intent_exemplars.items():
                if any(e.lower() in text for e in exemplars):
                    intent = name
                    break

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
        base_actions:       Actions always available regardless of intent.
        intent_actions:     ``{intent: (additional_action, ...)}`` mapping.
        blocked_when:       ``{action: (condition_name, ...)}`` — conditions
                             that block an action when true.
        custom_conditions:  Optional ``{condition_name: Callable[[Task], bool]}``
                             dict that extends or overrides the built-in
                             condition evaluators. Use this to add domain-specific
                             blocking logic without subclassing.

    Built-in conditions:
        - ``high_value``        : task.metadata["value_usd"] > 250
        - ``outside_30_days``   : task.metadata["days_since_event"] > 30
        - ``production_change`` : task.metadata["production_change"] is truthy
        - ``no_sources``        : task.metadata["source_count"] == 0
        - ``external_send``     : task.metadata["external_send"] is truthy

    Example (adding a custom condition)::

        agent = RulePolicyAgent(
            base_actions=("approve", "escalate"),
            blocked_when={"approve": ("high_risk_country",)},
            custom_conditions={
                "high_risk_country": lambda t: t.metadata.get("country") in {"XX", "YY"},
            },
        )
    """

    base_actions: tuple[str, ...]
    intent_actions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    blocked_when: dict[str, tuple[str, ...]] = field(default_factory=dict)
    custom_conditions: dict[str, Callable[[Task], bool]] = field(default_factory=dict)

    # Built-in condition evaluators (merged with custom_conditions in __post_init__)
    _CONDITIONS: dict[str, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        _builtin: dict[str, Callable[[Task], bool]] = {
            "high_value":        lambda t: float(t.metadata.get("value_usd", 0)) > 250,
            "outside_30_days":   lambda t: int(t.metadata.get("days_since_event", 0)) > 30,
            "production_change": lambda t: bool(t.metadata.get("production_change", False)),
            "no_sources":        lambda t: int(t.metadata.get("source_count", 1)) == 0,
            "external_send":     lambda t: bool(t.metadata.get("external_send", False)),
        }
        # custom_conditions can extend OR override built-ins
        self._CONDITIONS = {**_builtin, **self.custom_conditions}

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
