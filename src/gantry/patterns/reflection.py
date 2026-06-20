"""Reflection / Critic Loop Pattern — LangGraph StateGraph implementation.

A drafter agent produces a plan. A critic agent scores it.
If the score is below threshold and turns remain, the drafter
revises the plan using the critique findings as feedback.
The loop continues until approved or max_turns is reached.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict
from pydantic import BaseModel, ConfigDict
from langgraph.graph import StateGraph, START, END

from ..generic_agents import BasicVerifierAgent, KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent, safe_node
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import KnowledgeBaseRetriever


class Critique(BaseModel):
    """Structured feedback from the CriticAgent."""

    model_config = ConfigDict(frozen=True)

    score: float
    findings: tuple[str, ...]
    approved: bool


class CriticAgent:
    """Reviews a draft plan and returns a Critique."""

    def __init__(self, min_confidence: float = 0.75, require_citations: bool = True) -> None:
        self.min_confidence = min_confidence
        self.require_citations = require_citations

    def run(
        self,
        draft: Plan,
        policy: PolicyDecision,
        evidence: tuple[Evidence, ...],
    ) -> Critique:
        findings: list[str] = []

        if draft.action not in policy.allowed_actions:
            findings.append(f"Action '{draft.action}' is not in allowed actions.")
        if draft.confidence < self.min_confidence:
            findings.append(
                f"Confidence {draft.confidence:.2f} is below minimum {self.min_confidence}."
            )
        if self.require_citations and not draft.citations:
            findings.append("No citations provided; legal review requires source evidence.")
        if draft.action not in {"escalate", "ask_for_info", "log_only"} and not evidence:
            findings.append("No evidence retrieved for a substantive action.")

        score = max(0.0, 1.0 - 0.3 * len(findings))
        return Critique(score=score, findings=tuple(findings), approved=not findings)


class ReflectionState(TypedDict):
    task: Task
    signal: Optional[Signal]
    evidence: tuple[Evidence, ...]
    policy: Optional[PolicyDecision]
    draft: Optional[Plan]
    critique: Optional[Critique]
    turn: int
    verification: Optional[Verification]
    outcome: Optional[Outcome]
    audit_trail: Annotated[list[str], operator.add]  # LangGraph merges via operator.add


class ReflectionWeaver:
    """Reflection / Critic Loop pattern using LangGraph StateGraph."""

    def __init__(
        self,
        signal_agent: KeywordSignalAgent,
        policy_agent: RulePolicyAgent,
        drafter_agent: TemplatePlannerAgent,
        critic_agent: CriticAgent,
        verifier_agent: BasicVerifierAgent,
        retriever: KnowledgeBaseRetriever,
        max_turns: int = 3,
        approval_threshold: float = 0.7,
        fallback_action: str = "escalate",
        fallback_response: str = "Could not produce an approved plan after reflection; escalating.",
    ) -> None:
        self.signal_agent = signal_agent
        self.policy_agent = policy_agent
        self.drafter_agent = drafter_agent
        self.critic_agent = critic_agent
        self.verifier_agent = verifier_agent
        self.retriever = retriever
        self.max_turns = max_turns
        self.approval_threshold = approval_threshold
        self.fallback_action = fallback_action
        self.fallback_response = fallback_response

        builder = StateGraph(ReflectionState)
        builder.add_node("signal",   safe_node(self._signal_node,   {"audit_trail": ["signal:error"]}))
        builder.add_node("retrieve", safe_node(self._retrieve_node, {"evidence": (), "audit_trail": ["retrieve:error"]}))
        builder.add_node("policy",   safe_node(self._policy_node,   {"audit_trail": ["policy:error"], "turn": 0}))
        builder.add_node("drafter",  safe_node(self._drafter_node,  {"audit_trail": ["drafter:error"]}))
        builder.add_node("critic",   safe_node(self._critic_node,   {"audit_trail": ["critic:error"]}))
        builder.add_node("verify",   safe_node(self._verify_node,   {"audit_trail": ["verify:error"]}))
        builder.add_node(
            "finalize",
            safe_node(self._finalize_node, {"outcome": None, "audit_trail": ["finalize:error"]}),
        )

        builder.add_edge(START, "signal")
        builder.add_edge("signal", "retrieve")
        builder.add_edge("retrieve", "policy")
        builder.add_edge("policy", "drafter")
        builder.add_edge("drafter", "critic")

        # Cycle decision after critic node
        builder.add_conditional_edges(
            "critic",
            self._should_loop,
            {
                "drafter": "drafter",
                "verify": "verify",
            }
        )
        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)

        self.graph = builder.compile()

    def _signal_node(self, state: ReflectionState) -> dict:
        task = state["task"]
        signal = self.signal_agent.run(task)
        return {"signal": signal, "audit_trail": [f"signal:intent={signal.intent}:risk={signal.risk}"]}

    def _retrieve_node(self, state: ReflectionState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        return {"evidence": evidence, "audit_trail": [f"retrieval:documents={len(evidence)}"]}

    def _policy_node(self, state: ReflectionState) -> dict:
        task = state["task"]
        signal = state["signal"]
        policy = self.policy_agent.run(task, signal)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")
        return {"policy": policy, "audit_trail": audit, "turn": 0}

    def _drafter_node(self, state: ReflectionState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        turn = state.get("turn", 0) + 1

        if turn == 1:
            draft = self.drafter_agent.run(task, signal, evidence, policy)
        else:
            # Revision node logic: revise existing draft based on critique findings
            old_draft = state["draft"]
            critique = state["critique"]
            critique_note = "; ".join(critique.findings)
            revised_action = (
                old_draft.action if old_draft.action in policy.allowed_actions else "escalate"
            )
            draft = Plan(
                action=revised_action,
                confidence=min(1.0, old_draft.confidence + 0.06 * (turn - 1)),
                response=old_draft.response,
                internal_note=(
                    f"{old_draft.internal_note} | revision:{turn - 1} | critique:{critique_note}"
                ),
                citations=old_draft.citations or tuple(e.title for e in evidence[:2]),
            )

        return {"draft": draft, "turn": turn}

    def _critic_node(self, state: ReflectionState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        turn = state["turn"]

        critique = self.critic_agent.run(draft, policy, evidence)

        audit = list(state.get("audit_trail") or [])
        audit.append(
            f"turn:{turn}:action={draft.action}:confidence={draft.confidence:.2f}"
            f":critique_score={critique.score:.2f}:approved={critique.approved}"
        )

        return {"critique": critique, "audit_trail": audit}

    def _should_loop(self, state: ReflectionState) -> str:
        critique = state["critique"]
        turn = state["turn"]

        if critique.approved or critique.score >= self.approval_threshold or turn >= self.max_turns:
            return "verify"
        
        return "drafter"

    def _verify_node(self, state: ReflectionState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.verifier_agent.run(draft, policy, evidence)

        return {"verification": verification, "audit_trail": [f"verify:approved={verification.approved}:score={verification.score}"]}

    def _finalize_node(self, state: ReflectionState) -> dict:
        task = state["task"]
        signal = state.get("signal")
        evidence = state.get("evidence") or ()
        policy = state.get("policy")
        draft = state.get("draft")
        verification = state.get("verification")
        audit = list(state.get("audit_trail") or [])

        # Guard: upstream safe_node failure may have left None in state.
        if policy is None or draft is None or verification is None:
            missing = [k for k, v in [("policy", policy), ("draft", draft), ("verification", verification)] if v is None]
            audit.append(f"finalize:upstream_failure:missing={','.join(missing)}:escalating")
            outcome = Outcome(
                task_id=task.id,
                use_case=task.use_case,
                signal=signal or Signal(intent="unknown", risk=3),
                evidence=evidence,
                policy=policy or PolicyDecision(
                    allowed_actions=(self.fallback_action,),
                    reason="upstream node failure",
                ),
                draft=draft or Plan(
                    action=self.fallback_action,
                    confidence=0.0,
                    response=self.fallback_response,
                    internal_note="upstream node failure",
                ),
                verification=verification or Verification(
                    approved=False, score=0.0,
                    findings=("upstream node failed — see audit_trail",),
                ),
                final_action=self.fallback_action,
                response=self.fallback_response,
                internal_note=f"Upstream node failure: {', '.join(missing)} was None.",
                audit_trail=audit,
            )
            return {"outcome": outcome}

        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            note = (
                f"Reflection failed after {self.max_turns} turns. "
                f"Findings: {'; '.join(verification.findings)}"
            )

        outcome = Outcome(
            task_id=task.id,
            use_case=task.use_case,
            signal=signal,
            evidence=evidence,
            policy=policy,
            draft=draft,
            verification=verification,
            final_action=final_action,
            response=response,
            internal_note=note,
            audit_trail=audit,
        )
        return {"outcome": outcome}

    def run(self, task: Task) -> Outcome:
        initial_audit = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:reflection",
        ]
        result = self.graph.invoke({"task": task, "audit_trail": initial_audit, "turn": 0})
        return result["outcome"]
