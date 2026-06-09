"""Human-in-the-Loop (HITL) Pattern.

The workflow runs the standard signal → evidence → policy pipeline.
When it reaches an intent that requires human approval, it pauses
and returns a `pending_approval` Outcome instead of taking action.

The caller stores the Outcome, notifies a human reviewer, then
re-submits the original task with metadata["human_approved"] = True.
On the second run the workflow skips the checkpoint and continues
to the planner + verifier.

Pattern flow::

    Run 1:  Task -> Signal -> Evidence -> Policy
                 -> checkpoint (if approval required)
                 -> Outcome(final_action="pending_approval")

    Run 2:  Task (human_approved=True) -> Signal -> Evidence -> Policy
                 -> checkpoint skipped
                 -> Plan -> Verify
                 -> Outcome(final_action=<approved_action>)

To configure which intents require approval, set approval_required_intents
in the recipe.

Use case: HR & onboarding (leave approval, role provisioning).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..generic_agents import BasicVerifierAgent, KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification
from ..retrieval import TinyRetriever

CHECKPOINT_ACTION = "pending_approval"


@dataclass
class HumanInTheLoopWeaver:
    """Human-in-the-Loop (HITL) pattern.

    Pauses at a checkpoint for intents that require human sign-off.
    Returns pending_approval on the first run, resumes on the second
    run when task.metadata['human_approved'] is True.
    """

    signal_agent: KeywordSignalAgent
    policy_agent: RulePolicyAgent
    planner_agent: TemplatePlannerAgent
    verifier_agent: BasicVerifierAgent
    retriever: TinyRetriever
    approval_required_intents: tuple[str, ...]  # intents that need HITL
    fallback_action: str = "escalate"
    fallback_response: str = "Workflow failed after human approval; escalating."

    def run(self, task: Task) -> Outcome:
        audit: list[str] = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:human_in_the_loop",
        ]

        # 1. Standard pipeline start
        signal = self.signal_agent.run(task)
        audit.append(f"signal:intent={signal.intent}:risk={signal.risk}")

        evidence = self.retriever.search(task.text)
        audit.append(f"retrieval:documents={len(evidence)}")

        policy = self.policy_agent.run(task, signal)
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")

        # 2. Check for HITL checkpoint
        needs_approval = signal.intent in self.approval_required_intents
        human_approved = bool(task.metadata.get("human_approved", False))

        if needs_approval and not human_approved:
            audit.append(f"checkpoint:awaiting_human_approval:intent={signal.intent}")
            pending_plan = Plan(
                action=CHECKPOINT_ACTION,
                confidence=1.0,
                response=(
                    f"Action for intent '{signal.intent}' requires human approval. "
                    f"Re-submit this task with human_approved=true to proceed."
                ),
                internal_note=f"HITL checkpoint triggered for intent={signal.intent}",
            )
            # While pending, block all real actions
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

        # 3. Approved or intent does not require HITL — run full pipeline
        if human_approved:
            audit.append("checkpoint:human_approved=true:resuming_pipeline")

        draft = self.planner_agent.run(task, signal, evidence, policy)
        audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")

        verification = self.verifier_agent.run(draft, policy, evidence)
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")

        if verification.approved:
            final_action = draft.action
            response = draft.response
            note = draft.internal_note
        else:
            final_action = self.fallback_action
            response = self.fallback_response
            note = f"Verifier blocked after approval: {'; '.join(verification.findings)}"

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
