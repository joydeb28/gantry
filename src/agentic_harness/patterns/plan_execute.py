"""Plan → Execute Pattern.

A planner generates a multi-step approval sequence tailored to the
detected intent. Each step is executed deterministically and verified
independently. If any step fails (blocked by policy or verification),
the workflow short-circuits to the fallback action.

Pattern flow::

    Task -> Signal -> Evidence -> Policy
      -> PlannerAgent -> [step1, step2, step3, ...]
      -> Execute step 1 -> verify
      -> Execute step 2 -> verify
      -> Execute step N -> verify
      -> Aggregate results -> Outcome

To customise the execution steps, set step_map in the recipe:

    step_map = {
        "invoice_approval": (
            ("validate_invoice", "validate_invoice"),
            ("check_budget",    "check_budget"),
            ("approve_payment", "approve_payment"),
        ),
    }

Use case: finance & accounting (invoice approval workflows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..generic_agents import BasicVerifierAgent, KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import TinyRetriever


@dataclass(frozen=True)
class ExecutionStep:
    """Result of a single executed step."""

    name: str
    action: str
    result: str
    success: bool


@dataclass
class PlanExecuteWeaver:
    """Plan → Execute pattern.

    A planner outputs a sequence of steps. Each step is executed
    and verified independently. Any step failure causes the workflow
    to short-circuit to the fallback action.
    """

    signal_agent: KeywordSignalAgent
    policy_agent: RulePolicyAgent
    planner_agent: TemplatePlannerAgent
    verifier_agent: BasicVerifierAgent
    retriever: TinyRetriever
    # Maps intent -> ordered tuple of (step_name, action)
    step_map: dict[str, tuple[tuple[str, str], ...]]
    fallback_action: str = "escalate"
    fallback_response: str = "One or more execution steps failed; escalating for review."

    def run(self, task: Task) -> Outcome:
        audit: list[str] = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:plan_execute",
        ]

        # 1. Signal + Evidence + Policy
        signal = self.signal_agent.run(task)
        audit.append(f"signal:intent={signal.intent}:risk={signal.risk}")

        evidence = self.retriever.search(task.text)
        audit.append(f"retrieval:documents={len(evidence)}")

        policy = self.policy_agent.run(task, signal)
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")

        # 2. Initial plan
        draft = self.planner_agent.run(task, signal, evidence, policy)
        steps = self.step_map.get(signal.intent, ())
        audit.append(f"plan:action={draft.action}:steps={len(steps)}")

        # 3. Execute steps
        executed: list[ExecutionStep] = []
        all_success = True

        for step_name, step_action in steps:
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
                all_success = False
                break

            result = f"Step '{step_name}' completed: {step_action} applied to {task.id}."
            step = ExecutionStep(name=step_name, action=step_action, result=result, success=True)
            executed.append(step)
            audit.append(f"step:{step_name}:action={step_action}:success=True")

        # 4. Verify overall draft
        verification = self.verifier_agent.run(draft, policy, evidence)
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")

        if all_success and verification.approved:
            final_action = executed[-1].action if executed else draft.action
            step_names = ", ".join(s.name for s in executed)
            response = draft.response + (
                f" Steps completed: {step_names}." if step_names else ""
            )
            note = f"{draft.internal_note} | executed_steps={len(executed)}"
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            failed = [s.name for s in executed if not s.success]
            note = (
                f"Failed steps: {', '.join(failed) or 'verification_failed'}. "
                f"Escalating."
            )

        return Outcome(
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
