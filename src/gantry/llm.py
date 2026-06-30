"""LangChain LLM integration — multi-provider structured output planning.

Provides a LangChain-powered planner that calls an LLM and returns a typed
Plan object via ``with_structured_output(Plan)``.

Supported providers:
    - ``ollama``    : Local Ollama server (default; no API key required)
    - ``vllm``      : Local vLLM OpenAI-compatible server
    - ``openai``    : OpenAI API (requires ``pip install langchain-openai``)
    - ``gemini``    : Google Gemini API (requires ``pip install langchain-google-genai``)
    - ``anthropic`` : Anthropic API (requires ``pip install langchain-anthropic``)

Usage::

    # Ollama (no API key, local)
    planner = LangChainPlanner(model="qwen3:4b", provider="ollama")

    # OpenAI (requires OPENAI_API_KEY env var)
    planner = LangChainPlanner(model="gpt-4o-mini", provider="openai")

    # Gemini (requires GOOGLE_API_KEY env var)
    planner = LangChainPlanner(model="gemini-2.0-flash", provider="gemini")

    # Anthropic (requires ANTHROPIC_API_KEY env var)
    planner = LangChainPlanner(model="claude-3-5-haiku-latest", provider="anthropic")

    plan = planner.plan(task, signal, evidence, policy)
    # Returns a fully validated Plan Pydantic object directly
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .models import Evidence, Plan, PolicyDecision, Signal, Task

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a production-grade agentic planning engine. Your job is to produce \
a single, concise action plan for the given task.

Rules:
- The action MUST be one of the allowed_actions listed.
- The response must be professional and customer-facing.
- Cite the evidence sources (document titles) you relied on.
- Be concise — response ≤ 3 sentences.
- Confidence must reflect how certain you are given the evidence available.
"""

_HUMAN_TEMPLATE = """\
TASK
----
Use case : {use_case}
Title    : {title}
Body     : {body}

SIGNAL
------
Intent          : {intent}
Risk level      : {risk}/3
Missing fields  : {missing_fields}

POLICY
------
Allowed actions : {allowed_actions}
Blocked actions : {blocked_actions}
Policy reason   : {policy_reason}

EVIDENCE (retrieved KB documents)
----------------------------------
{evidence_text}

{session_context_block}Produce a plan. Return valid JSON matching the Plan schema.
"""


# ---------------------------------------------------------------------------
# Evidence budget helper (C-3)
# ---------------------------------------------------------------------------

_EVIDENCE_CHAR_BUDGET = 6_000  # ~1 500 tokens at 4 chars/token; safe for all providers


def _build_evidence_text(
    evidence: tuple[Evidence, ...],
    char_budget: int = _EVIDENCE_CHAR_BUDGET,
) -> str:
    """Build the evidence block for the LLM prompt, respecting a total character budget.

    Documents are included in score order (already ranked by the retriever).
    Each document gets an equal initial share of the budget; any leftover from
    short documents rolls over to subsequent ones.  Truncation is marked with
    ``[truncated]`` so the LLM knows the text is cut.

    Args:
        evidence:    Ranked Evidence tuple from the retriever.
        char_budget: Maximum total characters for the evidence block.
                     Default: 6 000 (~1 500 tokens).  Adjust per provider limits.

    Returns:
        A formatted string ready for the ``{evidence_text}`` prompt slot.
    """
    if not evidence:
        return "No documents retrieved."

    per_doc = max(200, char_budget // len(evidence))
    parts: list[str] = []
    remaining = char_budget

    for i, e in enumerate(evidence):
        if remaining <= 0:
            break
        alloc = min(per_doc, remaining)
        if len(e.text) <= alloc:
            text = e.text
        else:
            text = e.text[:alloc] + "[truncated]"
        parts.append(f"[{i + 1}] {e.title} (score={e.score:.2f})\n{text}")
        remaining -= len(text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

_PROVIDER_INSTALL_HINTS: dict[str, str] = {
    "openai":    "pip install langchain-openai",
    "gemini":    "pip install langchain-google-genai",
    "anthropic": "pip install langchain-anthropic",
}

_DEFAULT_MODELS: dict[str, str] = {
    "ollama":    "qwen3:4b",
    "vllm":      "Qwen/Qwen3-4B",
    "openai":    "gpt-4o-mini",
    "gemini":    "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
}

_DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434",
    "vllm":   "http://localhost:8000/v1",
}

# Retry parameters for LLM calls (C-4)
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles on each attempt (exponential backoff)


def build_llm(
    provider: str,
    model: str | None = None,
    temperature: float = 0.3,
    base_url: str | None = None,
) -> BaseChatModel:
    """Construct a LangChain ``BaseChatModel`` for the given provider.

    Providers ``openai``, ``gemini``, and ``anthropic`` are lazily imported
    so the corresponding package only needs to be installed when actually used.

    Args:
        provider:    One of ``ollama``, ``vllm``, ``openai``, ``gemini``, ``anthropic``.
        model:       Model name. Falls back to a sensible default per provider.
        temperature: Sampling temperature. Default: 0.3 (deterministic for planning).
        base_url:    Override the API base URL (useful for proxies or local servers).

    Returns:
        A ``BaseChatModel`` instance ready for ``.with_structured_output(Plan)``.

    Raises:
        ImportError:  If the required provider package is not installed.
        ValueError:   If an unknown provider is specified.
    """
    resolved_model = model or _DEFAULT_MODELS.get(provider, "")
    resolved_url = base_url or _DEFAULT_BASE_URLS.get(provider)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=resolved_model,
            temperature=temperature,
            base_url=resolved_url or "http://localhost:11434",
        )

    if provider == "vllm":
        # vLLM exposes an OpenAI-compatible endpoint; use langchain-openai
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "vLLM requires the 'langchain-openai' package. "
                "Install it with: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            base_url=resolved_url or "http://localhost:8000/v1",
            api_key=os.environ.get("VLLM_API_KEY", "not-needed"),  # type: ignore[arg-type]
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                f"OpenAI provider requires 'langchain-openai'. "
                f"Install with: {_PROVIDER_INSTALL_HINTS['openai']}"
            ) from exc
        kwargs: dict = {"model": resolved_model, "temperature": temperature}
        if resolved_url:
            kwargs["base_url"] = resolved_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                f"Gemini provider requires 'langchain-google-genai'. "
                f"Install with: {_PROVIDER_INSTALL_HINTS['gemini']}"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            temperature=temperature,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                f"Anthropic provider requires 'langchain-anthropic'. "
                f"Install with: {_PROVIDER_INSTALL_HINTS['anthropic']}"
            ) from exc
        return ChatAnthropic(
            model=resolved_model,  # type: ignore[call-arg]
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        f"Choose one of: ollama, vllm, openai, gemini, anthropic."
    )


# ---------------------------------------------------------------------------
# LangChain Planner
# ---------------------------------------------------------------------------

class LangChainPlanner:
    """LLM-powered planner — any provider, structured output.

    Calls the configured LLM and returns a typed ``Plan`` object directly
    via ``with_structured_output(Plan)``. No manual JSON parsing required.

    Args:
        model:       Model name. Falls back to the provider default when omitted.
        provider:    LLM provider. One of ``ollama``, ``vllm``, ``openai``,
                     ``gemini``, ``anthropic``. Default: ``ollama``.
        temperature: Sampling temperature. Default: 0.3.
        base_url:    Override API base URL.
        llm:         Pass a pre-built ``BaseChatModel`` directly to bypass the
                     provider factory (useful for testing or custom models).

    Example::

        # Use OpenAI
        planner = LangChainPlanner(provider="openai", model="gpt-4o-mini")

        # Bring your own model instance
        from langchain_openai import ChatOpenAI
        planner = LangChainPlanner(llm=ChatOpenAI(model="gpt-4o"))
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str = "ollama",
        temperature: float = 0.3,
        base_url: str | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        if llm is None:
            llm = build_llm(
                provider=provider,
                model=model,
                temperature=temperature,
                base_url=base_url,
            )

        structured_llm = llm.with_structured_output(Plan)
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_TEMPLATE),
        ])
        self._chain = prompt | structured_llm
        self._provider = provider
        self._model = model or _DEFAULT_MODELS.get(provider, "unknown")

    def plan(
        self,
        task: Task,
        signal: Signal,
        evidence: tuple[Evidence, ...],
        policy: PolicyDecision,
        session_context: str = "",
    ) -> Plan:
        """Generate a Plan using the configured LLM provider.

        Retries up to ``_MAX_RETRIES`` times with exponential backoff + jitter
        on transient failures.  Raises ``RuntimeError`` only after all attempts
        are exhausted.

        Args:
            task:             The input task.
            signal:           Extracted intent and risk.
            evidence:         Retrieved KB documents.
            policy:           Allowed/blocked actions.
            session_context:  Optional formatted conversation history string
                              from ``ConversationBuffer.get_context()``.  When
                              non-empty, injected above the instruction line so
                              the LLM can reference prior interactions.

        Returns:
            A fully validated ``Plan`` Pydantic object.
        """
        evidence_text = _build_evidence_text(evidence)
        session_context_block = (
            f"{session_context}\n\n" if session_context else ""
        )

        invoke_kwargs = {
            "use_case":              task.use_case,
            "title":                 task.title,
            "body":                  task.body,
            "intent":                signal.intent,
            "risk":                  signal.risk,
            "missing_fields":        ", ".join(signal.missing_fields) or "none",
            "allowed_actions":       ", ".join(policy.allowed_actions),
            "blocked_actions":       ", ".join(policy.blocked_actions) or "none",
            "policy_reason":         policy.reason or "standard policy",
            "evidence_text":         evidence_text,
            "session_context_block": session_context_block,
        }

        last_exc: Exception | None = None
        plan: Plan | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                plan = self._chain.invoke(invoke_kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.warning(
                        "LangChain planner attempt %d/%d failed (%s: %s); retrying in %.1fs",
                        attempt, _MAX_RETRIES, type(exc).__name__, exc, delay,
                    )
                    time.sleep(delay)
        else:
            raise RuntimeError(
                f"LangChain planner failed after {_MAX_RETRIES} attempts"
            ) from last_exc

        logger.info(
            "LangChain planner [%s/%s] → action=%s confidence=%.2f",
            self._provider, self._model, plan.action, plan.confidence,
        )
        return plan

    async def aplan(
        self,
        task: Task,
        signal: Signal,
        evidence: tuple[Evidence, ...],
        policy: PolicyDecision,
        session_context: str = "",
    ) -> Plan:
        """Async variant of :meth:`plan` — uses ``ainvoke`` on the LangChain chain.

        Retries up to ``_MAX_RETRIES`` times with exponential backoff + jitter.
        Safe to call from any ``asyncio`` event loop; does **not** block threads.

        Args:
            task:     The input task.
            signal:   Extracted intent and risk.
            evidence: Retrieved KB documents.
            policy:   Allowed/blocked actions.

        Returns:
            A fully validated ``Plan`` Pydantic object.
        """
        import asyncio

        evidence_text = _build_evidence_text(evidence)
        session_context_block = (
            f"{session_context}\n\n" if session_context else ""
        )
        invoke_kwargs = {
            "use_case":              task.use_case,
            "title":                 task.title,
            "body":                  task.body,
            "intent":                signal.intent,
            "risk":                  signal.risk,
            "missing_fields":        ", ".join(signal.missing_fields) or "none",
            "allowed_actions":       ", ".join(policy.allowed_actions),
            "blocked_actions":       ", ".join(policy.blocked_actions) or "none",
            "policy_reason":         policy.reason or "standard policy",
            "evidence_text":         evidence_text,
            "session_context_block": session_context_block,
        }

        last_exc: Exception | None = None
        plan: Plan | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                plan = await self._chain.ainvoke(invoke_kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.warning(
                        "LangChain planner (async) attempt %d/%d failed (%s: %s); retrying in %.1fs",
                        attempt, _MAX_RETRIES, type(exc).__name__, exc, delay,
                    )
                    await asyncio.sleep(delay)
        else:
            raise RuntimeError(
                f"LangChain planner (async) failed after {_MAX_RETRIES} attempts"
            ) from last_exc

        logger.info(
            "LangChain planner async [%s/%s] → action=%s confidence=%.2f",
            self._provider, self._model, plan.action, plan.confidence,
        )
        return plan


class LangChainPlannerAgent:
    """Wrapper that adapts LangChainPlanner to the agentic runner interface."""

    def __init__(self, planner: LangChainPlanner) -> None:
        self.planner = planner

    def run(
        self,
        task: Task,
        signal: Signal,
        evidence: tuple[Evidence, ...],
        policy: PolicyDecision,
    ) -> Plan:
        return self.planner.plan(task, signal, evidence, policy)

    async def arun(
        self,
        task: Task,
        signal: Signal,
        evidence: tuple[Evidence, ...],
        policy: PolicyDecision,
    ) -> Plan:
        """Async variant of :meth:`run` — delegates to :meth:`LangChainPlanner.aplan`."""
        return await self.planner.aplan(task, signal, evidence, policy)


# ---------------------------------------------------------------------------
# Dynamic Step Planner (for PlanExecuteWeaver)
# ---------------------------------------------------------------------------

from pydantic import BaseModel as _BaseModel, ConfigDict as _ConfigDict  # noqa: E402


class DynamicStepPlan(_BaseModel):
    """Structured output schema for LLM-generated execution step lists.

    Used by ``DynamicStepPlanner`` with ``with_structured_output`` so the LLM
    returns a validated, typed step list rather than free-form text.

    Each step is a ``[step_name, step_action]`` pair where:
    - ``step_name`` is a descriptive label (e.g. ``"validate_invoice"``)
    - ``step_action`` must be one of the ``allowed_actions`` in the current policy
    """

    model_config = _ConfigDict(frozen=True)

    steps: list[list[str]]   # [[step_name, step_action], ...]
    reasoning: str = ""


class DynamicStepPlanner:
    """LLM-generated step list for ``PlanExecuteWeaver``.

    Calls the LLM with a structured output schema to generate the sequence
    of steps for a given task, replacing the static ``step_map`` lookup in
    ``PlanExecuteWeaver``.

    Args:
        provider:   LLM provider (same options as ``LangChainPlanner``).
        model:      Model name override.
        base_url:   API base URL override.

    Example::

        from gantry.scenarios import weaver_for
        from gantry.llm import DynamicStepPlanner

        planner = DynamicStepPlanner(provider="openai", model="gpt-4o-mini")
        weaver  = weaver_for("finance", dynamic_step_planner=planner)
        outcome = weaver.run(task)
        # Steps are generated by the LLM rather than looked up from step_map
    """

    _STEP_SYSTEM_PROMPT = """\
You are a task decomposition engine. Given a task and allowed actions,
produce an ordered list of execution steps. Each step must use one of the
allowed_actions. Generate 2-5 steps. Be specific and actionable.
"""

    _STEP_HUMAN_TEMPLATE = """\
Task: {title}\nBody: {body}\nIntent: {intent}\nAllowed actions: {allowed_actions}\n
Generate an ordered execution plan as a list of [step_name, step_action] pairs.
"""

    def __init__(
        self,
        provider: str = "ollama",
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        llm = build_llm(provider=provider, model=model, base_url=base_url)
        structured = llm.with_structured_output(DynamicStepPlan)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._STEP_SYSTEM_PROMPT),
            ("human", self._STEP_HUMAN_TEMPLATE),
        ])
        self._chain = prompt | structured
        self._provider = provider

    def generate_steps(
        self,
        task: "Task",
        signal: "Signal",
        policy: "PolicyDecision",
    ) -> list[tuple[str, str]]:
        """Call the LLM and return a step list suitable for ``PlanExecuteWeaver``.

        Args:
            task:    The input task.
            signal:  Extracted intent and risk.
            policy:  Current policy decision (source of allowed_actions).

        Returns:
            List of ``(step_name, step_action)`` tuples.
            Falls back to ``[("default", allowed_actions[0])]`` if the LLM
            call fails or returns an empty step list.
        """
        try:
            result: DynamicStepPlan = self._chain.invoke({
                "title":          task.title,
                "body":           task.body,
                "intent":         signal.intent,
                "allowed_actions": ", ".join(policy.allowed_actions),
            })
            steps = [(s[0], s[1]) for s in result.steps if len(s) >= 2]
            if steps:
                logger.info(
                    "DynamicStepPlanner [%s]: generated %d steps for intent='%s'",
                    self._provider, len(steps), signal.intent,
                )
                return steps
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DynamicStepPlanner: LLM call failed (%s) — using fallback step.",
                exc,
            )
        # Safe fallback: single step using the first allowed action
        fallback_action = policy.allowed_actions[0] if policy.allowed_actions else "escalate"
        return [("default", fallback_action)]
