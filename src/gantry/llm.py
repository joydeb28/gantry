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

Produce a plan. Return valid JSON matching the Plan schema.
"""


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
    ) -> Plan:
        """Generate a Plan using the LLM.

        Args:
            task:     The input task.
            signal:   Extracted intent and risk.
            evidence: Retrieved KB documents.
            policy:   Allowed/blocked actions.

        Returns:
            A fully validated ``Plan`` Pydantic object.
        """
        evidence_text = "\n".join(
            f"[{i+1}] {e.title} (score={e.score:.2f})\n{e.text[:300]}"
            for i, e in enumerate(evidence)
        ) or "No documents retrieved."

        plan: Plan = self._chain.invoke({
            "use_case":       task.use_case,
            "title":          task.title,
            "body":           task.body,
            "intent":         signal.intent,
            "risk":           signal.risk,
            "missing_fields": ", ".join(signal.missing_fields) or "none",
            "allowed_actions": ", ".join(policy.allowed_actions),
            "blocked_actions": ", ".join(policy.blocked_actions) or "none",
            "policy_reason":   policy.reason or "standard policy",
            "evidence_text":   evidence_text,
        })

        logger.info(
            "LangChain planner [%s/%s] → action=%s confidence=%.2f",
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
