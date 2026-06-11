"""LangChain LLM integration — ChatOllama + structured output.

Provides a LangChain-powered planner that uses a local Ollama model
(default: qwen3:4b) to generate typed Plan objects via structured output.

The planner is used as a drop-in replacement for TemplatePlannerAgent in
any LangGraph node that needs LLM-powered reasoning instead of rule-based
template matching.

Usage::

    planner = LangChainPlanner(model="qwen3:4b")
    plan = planner.plan(task, signal, evidence, policy)
    # Returns a fully validated Plan Pydantic object directly
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

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
# LangChain Planner
# ---------------------------------------------------------------------------

class LangChainPlanner:
    """LLM-powered planner using ChatOllama + LangChain structured output.

    Calls a local Ollama model and returns a typed ``Plan`` object directly
    via ``with_structured_output(Plan)``. No manual JSON parsing required.

    Args:
        model:       Ollama model name. Default: ``qwen3:4b``.
        temperature: Sampling temperature. Default: 0.3 (deterministic for planning).
        base_url:    Ollama API base URL. Default: ``http://localhost:11434``.
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        temperature: float = 0.3,
        base_url: str = "http://localhost:11434",
    ) -> None:
        llm = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
        )
        # with_structured_output uses the Plan JSON schema to constrain output.
        # Returns a Plan Pydantic object directly — no parsing needed.
        structured_llm = llm.with_structured_output(Plan)

        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_TEMPLATE),
        ])

        self._chain = prompt | structured_llm
        self._model = model

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
            "LangChain planner [%s] → action=%s confidence=%.2f",
            self._model, plan.action, plan.confidence,
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

