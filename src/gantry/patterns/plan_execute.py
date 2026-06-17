"""Plan → Execute Pattern — LangGraph StateGraph implementation.

A planner generates a multi-step sequence. Each step is executed and verified
independently. If any step fails, the loop short-circuits to the fallback path.
"""

from __future__ import annotations

from typing import Optional, TypedDict
from pydantic import BaseModel, ConfigDict
from langgraph.graph import StateGraph, START, END

from ..generic_agents import BasicVerifierAgent, KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent, safe_node
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import KnowledgeBaseRetriever


class ExecutionStep(BaseModel):
    """Result of a single executed step."""

    model_config = ConfigDict(frozen=True)

    name: str
    action: str
    result: str
    success: bool


class PlanExecuteState(TypedDict):
    task: Task
    signal: Optional[Signal]
    evidence: tuple[Evidence, ...]
    policy: Optional[PolicyDecision]
    draft: Optional[Plan]
    verification: Optional[Verification]
    outcome: Optional[Outcome]
    remaining_steps: list[tuple[str, str]]
    executed_steps: list[ExecutionStep]
    all_success: bool
    audit_trail: list[str]


class PlanExecuteWeaver:
    """Plan → Execute pattern using LangGraph StateGraph."""

    def __init__(
        self,
        signal_agent: KeywordSignalAgent,
        policy_agent: RulePolicyAgent,
        planner_agent: TemplatePlannerAgent,
        verifier_agent: BasicVerifierAgent,
        retriever: KnowledgeBaseRetriever,
        step_map: dict[str, tuple[tuple[str, str], ...]],
        fallback_action: str = "escalate",
        fallback_response: str = "One or more execution steps failed; escalating for review.",
    ) -> None:
        self.signal_agent = signal_agent
        self.policy_agent = policy_agent
        self.planner_agent = planner_agent
        self.verifier_agent = verifier_agent
        self.retriever = retriever
        self.step_map = step_map
        self.fallback_action = fallback_action
        self.fallback_response = fallback_response

        builder = StateGraph(PlanExecuteState)
        builder.add_node("signal",          safe_node(self._signal_node,   {"audit_trail": ["signal:error"]}))
        builder.add_node("retrieve",         safe_node(self._retrieve_node, {"evidence": (), "audit_trail": ["retrieve:error"]}))
        builder.add_node("policy",           safe_node(self._policy_node,   {"audit_trail": ["policy:error"]}))
        builder.add_node("plan",             safe_node(self._plan_node,     {"audit_trail": ["plan:error"]}))
        builder.add_node("execute",          self._execute_node)
        builder.add_node("verify",           safe_node(self._verify_node,   {"audit_trail": ["verify:error"]}))
        builder.add_node("finalize",         self._finalize_node)
        builder.add_node("finalize_failed",  self._finalize_failed_node)

        builder.add_edge(START, "signal")
        builder.add_edge("signal", "retrieve")
        builder.add_edge("retrieve", "policy")
        builder.add_edge("policy", "plan")

        # Cycle / loop decision after plan and execute nodes
        builder.add_conditional_edges(
            "plan",
            self._should_execute,
            {
                "execute": "execute",
                "verify": "verify",
                "finalize_failed": "finalize_failed",
            }
        )
        builder.add_conditional_edges(
            "execute",
            self._should_execute,
            {
                "execute": "execute",
                "verify": "verify",
                "finalize_failed": "finalize_failed",
            }
        )

        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)
        builder.add_edge("finalize_failed", END)

        self.graph = builder.compile()

    def _signal_node(self, state: PlanExecuteState) -> dict:
        task = state["task"]
        signal = self.signal_agent.run(task)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"signal:intent={signal.intent}:risk={signal.risk}")
        return {"signal": signal, "audit_trail": audit}

    def _retrieve_node(self, state: PlanExecuteState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"retrieval:documents={len(evidence)}")
        return {"evidence": evidence, "audit_trail": audit}

    def _policy_node(self, state: PlanExecuteState) -> dict:
        task = state["task"]
        signal = state["signal"]
        policy = self.policy_agent.run(task, signal)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")
        return {"policy": policy, "audit_trail": audit}

    def _plan_node(self, state: PlanExecuteState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]

        draft = self.planner_agent.run(task, signal, evidence, policy)
        steps = list(self.step_map.get(signal.intent, ()))

        audit = list(state.get("audit_trail") or [])
        audit.append(f"plan:action={draft.action}:steps={len(steps)}")

        return {
            "draft": draft,
            "remaining_steps": steps,
            "executed_steps": [],
            "all_success": True,
            "audit_trail": audit,
        }

    def _execute_node(self, state: PlanExecuteState) -> dict:
        task = state["task"]
        policy = state["policy"]
        remaining = list(state["remaining_steps"])
        executed = list(state["executed_steps"])
        audit = list(state.get("audit_trail") or [])

        step_name, step_action = remaining.pop(0)

        if step_action not in policy.allowed_actions:
            step = ExecutionStep(
                name=step_name,
                action=step_action,
                result=f"Step '{step_action}' blocked by policy.",
                success=False,
            )
            executed.append(step)
            audit.append(
                f"step:{step_name}:action={step_action}:success=False:reason=policy_blocked"
            )
            return {
                "remaining_steps": remaining,
                "executed_steps": executed,
                "all_success": False,
                "audit_trail": audit,
            }

        result = f"Step '{step_name}' completed: {step_action} applied to {task.id}."
        step = ExecutionStep(name=step_name, action=step_action, result=result, success=True)
        executed.append(step)
        audit.append(f"step:{step_name}:action={step_action}:success=True")

        return {
            "remaining_steps": remaining,
            "executed_steps": executed,
            "audit_trail": audit,
        }

    def _should_execute(self, state: PlanExecuteState) -> str:
        if not state.get("all_success", True):
            return "finalize_failed"
        if state.get("remaining_steps"):
            return "execute"
        return "verify"

    def _verify_node(self, state: PlanExecuteState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.verifier_agent.run(draft, policy, evidence)

        audit = list(state.get("audit_trail") or [])
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")
        return {"verification": verification, "audit_trail": audit}

    def _finalize_node(self, state: PlanExecuteState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = state["draft"]
        verification = state["verification"]
        executed = state["executed_steps"]
        audit = list(state.get("audit_trail") or [])

        if verification.approved:
            final_action = executed[-1].action if executed else draft.action
            step_names = ", ".join(s.name for s in executed)
            response = draft.response + (
                f" Steps completed: {step_names}." if step_names else ""
            )
            note = f"{draft.internal_note} | executed_steps={len(executed)}"
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            note = f"Verifier blocked: {'; '.join(verification.findings)}"

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

    def _finalize_failed_node(self, state: PlanExecuteState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = state["draft"]
        executed = state["executed_steps"]
        audit = list(state.get("audit_trail") or [])

        final_action = self.fallback_action
        response = self.fallback_response
        failed = [s.name for s in executed if not s.success]
        note = (
            f"Failed steps: {', '.join(failed) or 'verification_failed'}. "
            f"Escalating."
        )

        outcome = Outcome(
            task_id=task.id,
            use_case=task.use_case,
            signal=signal,
            evidence=evidence,
            policy=policy,
            draft=draft,
            verification=Verification(approved=False, score=0.0, findings=("Execution step failed.",)),
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
            "pattern:plan_execute",
        ]
        result = self.graph.invoke({"task": task, "audit_trail": initial_audit})
        return result["outcome"]
