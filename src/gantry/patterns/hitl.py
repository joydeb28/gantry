"""Human-in-the-Loop (HITL) Pattern — LangGraph StateGraph implementation.

Pauses execution using LangGraph's native interrupt() mechanism at the approval gate
for intents requiring human sign-off.

Checkpoint backends:
    - ``sqlite``   (default) : Persists to ``.gantry_checkpoints.db`` in the working dir.
    - ``postgres`` : Reads ``DATABASE_URL`` env var (e.g. postgresql://user:pass@host/db).
                     Requires ``langgraph-checkpoint-postgres`` (installed with the [prod] extra).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

from ..generic_agents import BasicVerifierAgent, KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent, safe_node
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import KnowledgeBaseRetriever

logger = logging.getLogger(__name__)

CHECKPOINT_ACTION = "pending_approval"


class HITLState(TypedDict):
    task: Task
    signal: Optional[Signal]
    evidence: tuple[Evidence, ...]
    policy: Optional[PolicyDecision]
    draft: Optional[Plan]
    verification: Optional[Verification]
    outcome: Optional[Outcome]
    human_approved: bool
    audit_trail: list[str]


def _build_checkpointer(backend: str, db_url: str | None = None) -> Any:
    """Construct a LangGraph checkpointer for the given backend.

    Args:
        backend: ``"sqlite"`` or ``"postgres"``.
        db_url:  Database URL. For postgres, reads ``DATABASE_URL`` env var if omitted.

    Returns:
        A LangGraph checkpointer instance.
    """
    if backend == "postgres":
        resolved_url = db_url or os.environ.get("DATABASE_URL", "")
        if not resolved_url:
            raise ValueError(
                "PostgreSQL checkpointer requires a database URL. "
                "Set the DATABASE_URL environment variable or pass db_url explicitly.\n"
                "Example: DATABASE_URL=postgresql://user:pass@localhost/gantry"
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            conn = PostgresSaver.from_conn_string(resolved_url)
            conn.setup()  # creates tables if they don't exist
            logger.info("HITL: using PostgreSQL checkpointer at %s", resolved_url.split("@")[-1])
            return conn
        except ImportError as exc:
            raise ImportError(
                "PostgreSQL checkpointer requires 'langgraph-checkpoint-postgres'. "
                "Install with: pip install 'gantry[prod]'"
            ) from exc

    # Default: SQLite
    conn = sqlite3.connect(".gantry_checkpoints.db", check_same_thread=False)
    logger.info("HITL: using SQLite checkpointer (.gantry_checkpoints.db)")
    return SqliteSaver(conn)


class HumanInTheLoopWeaver:
    """Human-in-the-Loop (HITL) pattern using LangGraph StateGraph.

    Args:
        signal_agent:              Intent + risk extraction agent.
        policy_agent:              Action gating policy agent.
        planner_agent:             Draft plan generation agent.
        verifier_agent:            Plan verification agent.
        retriever:                 Knowledge base retriever.
        approval_required_intents: Intents that trigger the HITL checkpoint.
        fallback_action:           Action when workflow is rejected/fails.
        fallback_response:         Response text for the fallback action.
        checkpointer:              Pre-built LangGraph checkpointer. If None,
                                   uses ``checkpoint_backend`` to build one.
        checkpoint_backend:        ``"sqlite"`` (default) or ``"postgres"``.
        db_url:                    DB URL for postgres backend. Reads ``DATABASE_URL``
                                   env var if not provided.
    """

    def __init__(
        self,
        signal_agent: KeywordSignalAgent,
        policy_agent: RulePolicyAgent,
        planner_agent: TemplatePlannerAgent,
        verifier_agent: BasicVerifierAgent,
        retriever: KnowledgeBaseRetriever,
        approval_required_intents: tuple[str, ...],
        fallback_action: str = "escalate",
        fallback_response: str = "Workflow failed after human approval; escalating.",
        checkpointer: Optional[Any] = None,
        checkpoint_backend: str = "sqlite",
        db_url: str | None = None,
    ) -> None:
        self.signal_agent = signal_agent
        self.policy_agent = policy_agent
        self.planner_agent = planner_agent
        self.verifier_agent = verifier_agent
        self.retriever = retriever
        self.approval_required_intents = approval_required_intents
        self.fallback_action = fallback_action
        self.fallback_response = fallback_response

        if checkpointer is None:
            checkpointer = _build_checkpointer(checkpoint_backend, db_url)
        self.checkpointer = checkpointer

        builder = StateGraph(HITLState)
        builder.add_node("signal",   safe_node(self._signal_node,   {"audit_trail": ["signal:error"]}))
        builder.add_node("retrieve", safe_node(self._retrieve_node, {"evidence": (), "audit_trail": ["retrieve:error"]}))
        builder.add_node("policy",   safe_node(self._policy_node,   {"audit_trail": ["policy:error"]}))
        builder.add_node("gate",     self._approval_gate_node)  # cannot safe_node — uses interrupt()
        builder.add_node("plan",     safe_node(self._plan_node,     {"audit_trail": ["plan:error"]}))
        builder.add_node("verify",   safe_node(self._verify_node,   {"audit_trail": ["verify:error"]}))
        builder.add_node("finalize", self._finalize_node)
        builder.add_node("finalize_rejected", self._finalize_rejected_node)

        builder.add_edge(START, "signal")
        builder.add_edge("signal", "retrieve")
        builder.add_edge("retrieve", "policy")
        builder.add_edge("policy", "gate")

        builder.add_conditional_edges(
            "gate",
            self._after_gate,
            {
                "plan": "plan",
                "finalize_rejected": "finalize_rejected",
            }
        )

        builder.add_edge("plan", "verify")
        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)
        builder.add_edge("finalize_rejected", END)

        self.graph = builder.compile(checkpointer=self.checkpointer)

    def _signal_node(self, state: HITLState) -> dict:
        task = state["task"]
        signal = self.signal_agent.run(task)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"signal:intent={signal.intent}:risk={signal.risk}")
        return {"signal": signal, "audit_trail": audit}

    def _retrieve_node(self, state: HITLState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"retrieval:documents={len(evidence)}")
        return {"evidence": evidence, "audit_trail": audit}

    def _policy_node(self, state: HITLState) -> dict:
        task = state["task"]
        signal = state["signal"]
        policy = self.policy_agent.run(task, signal)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")
        return {"policy": policy, "audit_trail": audit}

    def _approval_gate_node(self, state: HITLState) -> dict:
        signal = state["signal"]
        needs_approval = signal.intent in self.approval_required_intents
        human_approved = state.get("human_approved", False)

        if needs_approval and not human_approved:
            audit = list(state.get("audit_trail") or [])
            audit.append(f"checkpoint:awaiting_human_approval:intent={signal.intent}")

            # native LangGraph interrupt
            decision = interrupt({
                "task_id": state["task"].id,
                "message": f"Action for intent '{signal.intent}' requires human approval. Re-submit this task with human_approved=true to proceed.",
                "intent": signal.intent,
            })

            approved = False
            if isinstance(decision, dict):
                approved = bool(decision.get("approved", False))

            audit = list(state.get("audit_trail") or [])
            if approved:
                audit.append("checkpoint:human_approved=true:resuming_pipeline")
            return {"human_approved": approved, "audit_trail": audit}

        return {"human_approved": True}

    def _after_gate(self, state: HITLState) -> str:
        if not state.get("human_approved", True):
            return "finalize_rejected"
        return "plan"

    def _plan_node(self, state: HITLState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = self.planner_agent.run(task, signal, evidence, policy)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")
        return {"draft": draft, "audit_trail": audit}

    def _verify_node(self, state: HITLState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.verifier_agent.run(draft, policy, evidence)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")
        return {"verification": verification, "audit_trail": audit}

    def _finalize_node(self, state: HITLState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = state["draft"]
        verification = state["verification"]
        audit = list(state.get("audit_trail") or [])

        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            note = f"Verifier blocked after approval: {'; '.join(verification.findings)}"

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

    def _finalize_rejected_node(self, state: HITLState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        audit = list(state.get("audit_trail") or [])
        audit.append("checkpoint:human_approved=false:workflow_rejected")

        outcome = Outcome(
            task_id=task.id,
            use_case=task.use_case,
            signal=signal,
            evidence=evidence,
            policy=policy,
            draft=Plan(
                action=self.fallback_action,
                confidence=1.0,
                response=self.fallback_response,
                internal_note="Human rejected request at checkpoint."
            ),
            verification=Verification(approved=False, score=0.0, findings=("Human rejected approval request.",)),
            final_action=self.fallback_action,
            response=self.fallback_response,
            internal_note="Workflow rejected by human reviewer.",
            audit_trail=audit,
        )
        return {"outcome": outcome}

    def run(self, task: Task) -> Outcome:
        config = {"configurable": {"thread_id": task.id}}
        initial_audit = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:human_in_the_loop",
        ]

        result = self.graph.invoke(
            {"task": task, "audit_trail": initial_audit, "human_approved": False},
            config=config
        )

        # Check if the graph execution paused at the gate interrupt
        if "__interrupt__" in result and result["__interrupt__"]:
            signal = result["signal"]
            evidence = result["evidence"]
            policy = result["policy"]
            audit = result["audit_trail"]

            pending_plan = Plan(
                action=CHECKPOINT_ACTION,
                confidence=1.0,
                response=(
                    f"Action for intent '{signal.intent}' requires human approval. "
                    f"Re-submit this task with human_approved=true to proceed."
                ),
                internal_note=f"HITL checkpoint triggered for intent={signal.intent}",
            )
            pending_policy = PolicyDecision(
                allowed_actions=(CHECKPOINT_ACTION,),
                blocked_actions=tuple(sorted(policy.allowed_actions)),
                reason="HITL checkpoint active; awaiting human approval.",
            )
            pending_verification = Verification(approved=True, score=1.0)

            return Outcome(
                task_id=task.id,
                use_case=task.use_case,
                signal=signal,
                evidence=evidence,
                policy=pending_policy,
                draft=pending_plan,
                verification=pending_verification,
                final_action=CHECKPOINT_ACTION,
                response=pending_plan.response,
                internal_note=pending_plan.internal_note,
                audit_trail=audit,
            )

        return result["outcome"]
