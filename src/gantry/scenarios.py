"""Use-case recipes for Agentic Harness.

Each recipe returns a weaver instance configured with the right
agentic pattern and agents for that domain.

Pattern summary:
  support   -> PipelineWeaver          (sequential claim chain)
  crm       -> PipelineWeaver          (sequential claim chain)
  research  -> PipelineWeaver          (sequential claim chain)
  coding    -> PipelineWeaver          (sequential claim chain)
  guardrail -> OrchestratorWeaver      (orchestrator + sub-agents)
  fraud     -> FraudOrchestratorWeaver (5 specialist fraud sub-agents)
  it        -> RouterWeaver            (router + specialist dispatch)
  legal     -> ReflectionWeaver        (draft -> critic loop)
  hr        -> HumanInTheLoopWeaver    (checkpoint + resume)
  finance   -> PlanExecuteWeaver       (multi-step plan execution)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .generic_agents import BasicVerifierAgent, KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent
from .models import AgentRecipe
from .patterns.pipeline import PipelineWeaver
from .patterns.hitl import HumanInTheLoopWeaver
from .patterns.orchestrator import (
    ExternalSendSubAgent,
    OrchestratorWeaver,
    PIISubAgent,
    SafetySubAgent,
    ToneSubAgent,
)
from .patterns.plan_execute import PlanExecuteWeaver
from .patterns.reflection import CriticAgent, ReflectionWeaver
from .patterns.router import RouterWeaver, SpecialistAgent
from .patterns.fraud import (
    FraudOrchestratorWeaver,
    CardFraudSubAgent,
    AccountTakeoverSubAgent,
    SyntheticIdentitySubAgent,
    VelocitySubAgent,
    GeoRiskSubAgent,
)
from .retrieval import KnowledgeBaseRetriever


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def weaver_for(
    use_case: str,
    kb_root: str | Path = "examples",
    planner_type: str = "template",
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """Return the correct weaver for a use case, loaded with its KB.

    Each use case uses a different agentic pattern. The returned object
    always exposes a run(task) -> Outcome method.
    """
    builders = {
        "support": _pipeline_weaver,
        "crm": _pipeline_weaver,
        "research": _pipeline_weaver,
        "coding": _pipeline_weaver,
        "guardrail": _orchestrator_weaver,
        "fraud": _fraud_weaver,
        "it": _router_weaver,
        "legal": _reflection_weaver,
        "hr": _hitl_weaver,
        "finance": _plan_execute_weaver,
    }
    try:
        builder = builders[use_case]
    except KeyError as exc:
        raise ValueError(
            f"Unknown use case '{use_case}'. Choose one of: {', '.join(sorted(builders))}"
        ) from exc

    retriever = KnowledgeBaseRetriever.from_use_case(use_case, kb_root=kb_root)

    # Dynamically build the LLM-powered planner agent if requested
    planner_agent = None
    if planner_type in ("ollama", "vllm"):
        from .llm import LangChainPlanner, LangChainPlannerAgent
        model = model_name or ("qwen3:4b" if planner_type == "ollama" else "Qwen/Qwen3-4B")
        default_url = "http://localhost:11434" if planner_type == "ollama" else "http://localhost:8000"
        url = base_url or default_url
        planner = LangChainPlanner(model=model, base_url=url)
        planner_agent = LangChainPlannerAgent(planner)

    return builder(use_case, retriever, planner_agent=planner_agent)


def recipe_for(use_case: str) -> AgentRecipe:
    """Return an AgentRecipe for pipeline-pattern use cases.

    For non-pipeline use cases (guardrail, it, legal, hr, finance)
    use weaver_for() instead.
    """
    recipes = {
        "support": support_recipe,
        "crm": crm_recipe,
        "research": research_recipe,
        "coding": coding_recipe,
    }
    try:
        return recipes[use_case]()
    except KeyError as exc:
        raise ValueError(
            f"'{use_case}' uses a non-pipeline pattern. Use weaver_for() instead."
        ) from exc


# ---------------------------------------------------------------------------
# Internal builder helpers
# ---------------------------------------------------------------------------

def _pipeline_weaver(
    use_case: str,
    retriever: KnowledgeBaseRetriever,
    planner_agent: Optional[Any] = None,
) -> PipelineWeaver:
    recipe_map = {
        "support": support_recipe,
        "crm": crm_recipe,
        "research": research_recipe,
        "coding": coding_recipe,
    }
    recipe = recipe_map[use_case]()
    if planner_agent is not None:
        recipe = recipe.model_copy(update={"planner_agent": planner_agent})
    return PipelineWeaver(recipe, retriever)


def _orchestrator_weaver(
    use_case: str,
    retriever: KnowledgeBaseRetriever,
    planner_agent: Optional[Any] = None,
) -> OrchestratorWeaver:
    return guardrail_harness(retriever, planner_agent=planner_agent)


def _router_weaver(
    use_case: str,
    retriever: KnowledgeBaseRetriever,
    planner_agent: Optional[Any] = None,
) -> RouterWeaver:
    return it_harness(retriever, planner_agent=planner_agent)


def _reflection_weaver(
    use_case: str,
    retriever: KnowledgeBaseRetriever,
    planner_agent: Optional[Any] = None,
) -> ReflectionWeaver:
    return legal_harness(retriever, planner_agent=planner_agent)


def _hitl_weaver(
    use_case: str,
    retriever: KnowledgeBaseRetriever,
    planner_agent: Optional[Any] = None,
) -> HumanInTheLoopWeaver:
    return hr_harness(retriever, planner_agent=planner_agent)


def _plan_execute_weaver(
    use_case: str,
    retriever: KnowledgeBaseRetriever,
    planner_agent: Optional[Any] = None,
) -> PlanExecuteWeaver:
    return finance_harness(retriever, planner_agent=planner_agent)


def _fraud_weaver(
    use_case: str,
    retriever: KnowledgeBaseRetriever,
    planner_agent: Optional[Any] = None,
) -> FraudOrchestratorWeaver:
    return fraud_harness(retriever, planner_agent=planner_agent)


# ---------------------------------------------------------------------------
# Pipeline recipes (support, crm, research, coding)
# ---------------------------------------------------------------------------

def support_recipe() -> AgentRecipe:
    return AgentRecipe(
        use_case="support",
        signal_agent=KeywordSignalAgent(
            intents={
                "refund": ("refund", "money back", "charged"),
                "replacement": ("broken", "damaged", "cracked", "defective"),
                "account_access": ("login", "password", "account"),
            },
            high_risk_words=("angry", "urgent", "frustrated"),
            missing_fields_by_intent={"refund": ("value_usd",), "replacement": ("value_usd",)},
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("answer", "ask_for_info", "escalate"),
            intent_actions={"refund": ("refund",), "replacement": ("replace",), "account_access": ("answer",)},
            blocked_when={"refund": ("high_value",), "replace": ("outside_30_days",)},
        ),
        planner_agent=TemplatePlannerAgent(
            templates={
                "refund": ("refund", "Your request appears eligible for a refund under the support policy."),
                "replacement": ("replace", "I can arrange a replacement under the damaged item policy."),
                "account_access": ("answer", "Start with password reset; escalate if reset email cannot be received."),
            },
            default_action="answer",
            default_response="I found a matching support article that should help.",
        ),
        verifier_agent=BasicVerifierAgent(),
    )


def crm_recipe() -> AgentRecipe:
    return AgentRecipe(
        use_case="crm",
        signal_agent=KeywordSignalAgent(
            intents={
                "hot_lead": ("budget", "demo", "decision maker", "this week"),
                "renewal_risk": ("cancel", "too expensive", "competitor"),
                "follow_up": ("follow up", "next month", "circle back"),
            },
            high_risk_words=("cancel", "competitor", "urgent"),
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("create_task", "log_note", "escalate"),
            intent_actions={"hot_lead": ("schedule_demo",), "renewal_risk": ("escalate",), "follow_up": ("create_task",)},
        ),
        planner_agent=TemplatePlannerAgent(
            templates={
                "hot_lead": ("schedule_demo", "Create a demo task and prioritize this lead for sales outreach."),
                "renewal_risk": ("escalate", "Escalate to the account owner with churn-risk context."),
                "follow_up": ("create_task", "Create a dated follow-up task in the CRM."),
            },
            default_action="log_note",
            default_response="Log the interaction and keep the account timeline updated.",
        ),
        verifier_agent=BasicVerifierAgent(),
    )


def research_recipe() -> AgentRecipe:
    return AgentRecipe(
        use_case="research",
        signal_agent=KeywordSignalAgent(
            intents={
                "literature_scan": ("papers", "survey", "literature", "state of the art"),
                "fact_check": ("verify", "citation", "source", "claim"),
                "summary": ("summarize", "brief", "overview"),
            },
            high_risk_words=("medical", "legal", "financial", "latest"),
            missing_fields_by_intent={"fact_check": ("source_count",)},
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("summarize", "ask_for_info", "escalate"),
            intent_actions={"literature_scan": ("build_reading_list",), "fact_check": ("fact_check",), "summary": ("summarize",)},
            blocked_when={"fact_check": ("no_sources",)},
        ),
        planner_agent=TemplatePlannerAgent(
            templates={
                "literature_scan": ("build_reading_list", "Build a reading list grouped by theme and evidence quality."),
                "fact_check": ("fact_check", "Check the claim against retrieved sources and mark uncertainty."),
                "summary": ("summarize", "Summarize the retrieved material with citations."),
            },
            default_action="summarize",
            default_response="Summarize the available evidence with citations.",
        ),
        verifier_agent=BasicVerifierAgent(),
    )


def coding_recipe() -> AgentRecipe:
    return AgentRecipe(
        use_case="coding",
        signal_agent=KeywordSignalAgent(
            intents={
                "bugfix": ("bug", "traceback", "failing test", "exception"),
                "feature": ("add", "implement", "feature"),
                "review": ("review", "risk", "regression"),
            },
            high_risk_words=("production", "security", "database", "migration"),
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("explain", "write_patch", "run_tests", "escalate"),
            intent_actions={"bugfix": ("write_patch", "run_tests"), "feature": ("write_patch",), "review": ("explain",)},
            blocked_when={"write_patch": ("production_change",)},
        ),
        planner_agent=TemplatePlannerAgent(
            templates={
                "bugfix": ("write_patch", "Patch the suspected bug and run focused tests."),
                "feature": ("write_patch", "Implement the feature behind the smallest clear interface."),
                "review": ("explain", "Report risks first, then summarize residual test gaps."),
            },
            default_action="explain",
            default_response="Explain the relevant code path before taking action.",
        ),
        verifier_agent=BasicVerifierAgent(),
    )


# ---------------------------------------------------------------------------
# Non-pipeline harnesses
# ---------------------------------------------------------------------------

def guardrail_harness(retriever: KnowledgeBaseRetriever, planner_agent: Optional[Any] = None) -> OrchestratorWeaver:
    """Guardrails — Orchestrator + Sub-agents pattern.

    Four specialist sub-agents check different risk dimensions in parallel.
    The orchestrator aggregates all findings into a single Signal, then
    runs the shared Policy -> Plan -> Verify pipeline.

    To add a new guardrail dimension, create a sub-agent with a
    run(task) -> SubAgentFinding method and add it to sub_agents.
    """
    return OrchestratorWeaver(
        sub_agents=(
            PIISubAgent(),
            SafetySubAgent(),
            ExternalSendSubAgent(),
            ToneSubAgent(),
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("allow", "redact", "escalate", "log_only"),
            intent_actions={
                "pii": ("redact",),
                "unsafe_prompt": ("escalate",),
                "external_send": ("escalate",),
                "safe": ("allow",),
            },
            blocked_when={"allow": ("external_send",)},
        ),
        planner_agent=planner_agent or TemplatePlannerAgent(
            templates={
                "pii": ("redact", "Redact all detected PII fields before processing continues."),
                "unsafe_prompt": ("escalate", "Route this prompt for policy review; do not execute."),
                "external_send": ("escalate", "Human approval required before any external data sharing."),
                "safe": ("allow", "No guardrail violations found; request may proceed."),
            },
            default_action="log_only",
            default_response="No clear classification; logging for review.",
        ),
        verifier_agent=BasicVerifierAgent(),
        retriever=retriever,
        fallback_action="escalate",
        fallback_response="Guardrail verification failed; route to review.",
    )


def it_harness(retriever: KnowledgeBaseRetriever, planner_agent: Optional[Any] = None) -> RouterWeaver:
    """IT Helpdesk — Router / Dispatcher pattern.

    A router classifies the ticket type and dispatches to one of three
    specialist agents, each with their own isolated policy + plan pipeline.

    To add a new specialist: create a SpecialistAgent, add route keywords,
    and register it in the routes dict.
    """
    access_specialist = SpecialistAgent(
        name="access_specialist",
        policy_agent=RulePolicyAgent(
            base_actions=("ask_for_info", "escalate"),
            intent_actions={"access_request": ("grant_access",)},
        ),
        planner_agent=planner_agent or TemplatePlannerAgent(
            templates={
                "access_request": ("grant_access", "Provision access per the role matrix; confirm with the manager."),
            },
            default_action="ask_for_info",
            default_response="Please specify the system and permission level required.",
        ),
    )
    incident_specialist = SpecialistAgent(
        name="incident_specialist",
        policy_agent=RulePolicyAgent(
            base_actions=("open_ticket", "escalate"),
            intent_actions={"incident": ("open_ticket",)},
        ),
        planner_agent=planner_agent or TemplatePlannerAgent(
            templates={
                "incident": ("open_ticket", "Open a P2 incident ticket and page the on-call engineer."),
            },
            default_action="open_ticket",
            default_response="Creating an incident ticket for tracking and resolution.",
        ),
    )
    hardware_specialist = SpecialistAgent(
        name="hardware_specialist",
        policy_agent=RulePolicyAgent(
            base_actions=("ask_for_info", "escalate"),
            intent_actions={"hardware_request": ("raise_procurement",)},
        ),
        planner_agent=planner_agent or TemplatePlannerAgent(
            templates={
                "hardware_request": ("raise_procurement", "Raise a procurement request with the hardware team."),
            },
            default_action="ask_for_info",
            default_response="Please provide the device type, specs, and business justification.",
        ),
    )
    return RouterWeaver(
        router_keywords={
            "access_request": ("access", "permission", "login", "account", "vpn", "role"),
            "incident": ("down", "outage", "broken", "error", "crash", "slow", "not working"),
            "hardware_request": ("laptop", "monitor", "keyboard", "mouse", "device", "hardware"),
        },
        routes={
            "access_request": access_specialist,
            "incident": incident_specialist,
            "hardware_request": hardware_specialist,
        },
        default_route="incident",
        verifier_agent=BasicVerifierAgent(),
        retriever=retriever,
    )


def legal_harness(retriever: KnowledgeBaseRetriever, planner_agent: Optional[Any] = None) -> ReflectionWeaver:
    """Legal & Compliance — Reflection / Critic Loop pattern.

    A drafter produces a plan. The critic checks citations, confidence,
    and policy compliance. The loop runs up to 3 turns until the critic
    approves or the threshold is met.

    To tune the critic: adjust min_confidence or require_citations, or
    subclass CriticAgent to add domain-specific checks (e.g. clause validation).
    """
    return ReflectionWeaver(
        signal_agent=KeywordSignalAgent(
            intents={
                "contract_review": ("contract", "agreement", "clause", "nda", "terms"),
                "compliance_check": ("gdpr", "hipaa", "sox", "compliance", "regulation"),
                "dispute": ("dispute", "breach", "liability", "claim", "lawsuit"),
            },
            high_risk_words=("lawsuit", "breach", "penalty", "liability", "gdpr", "hipaa"),
            missing_fields_by_intent={},
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("flag_for_review", "ask_for_info", "escalate"),
            intent_actions={
                "contract_review": ("flag_for_review", "summarize_contract"),
                "compliance_check": ("flag_for_review",),
                "dispute": ("escalate",),
            },
        ),
        drafter_agent=planner_agent or TemplatePlannerAgent(
            templates={
                "contract_review": ("flag_for_review", "Flag the contract for legal counsel review and highlight non-standard clauses."),
                "compliance_check": ("flag_for_review", "Run a compliance checklist and document any gaps found."),
                "dispute": ("escalate", "Escalate to legal team immediately with all supporting documentation."),
            },
            default_action="flag_for_review",
            default_response="Flagging this matter for legal review.",
        ),
        critic_agent=CriticAgent(min_confidence=0.75, require_citations=True),
        verifier_agent=BasicVerifierAgent(),
        retriever=retriever,
        max_turns=3,
        approval_threshold=0.7,
    )


def hr_harness(retriever: KnowledgeBaseRetriever, planner_agent: Optional[Any] = None) -> HumanInTheLoopWeaver:
    """HR & Onboarding — Human-in-the-Loop pattern.

    Leave requests and onboarding trigger a HITL checkpoint: the workflow
    returns pending_approval on the first run and resumes on the second
    run when task.metadata['human_approved'] is True.

    Policy queries are answered directly without a checkpoint.

    To change which intents require approval, update approval_required_intents.
    """
    return HumanInTheLoopWeaver(
        signal_agent=KeywordSignalAgent(
            intents={
                "onboarding": ("new hire", "onboard", "joining", "start date", "welcome kit"),
                "leave_request": ("leave", "vacation", "pto", "time off", "annual leave"),
                "policy_query": ("policy", "handbook", "rules", "entitlement", "benefit"),
            },
            high_risk_words=("immediate", "urgent", "backdated"),
            missing_fields_by_intent={},
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("answer", "ask_for_info", "escalate"),
            intent_actions={
                "onboarding": ("send_welcome_kit", "provision_accounts"),
                "leave_request": ("approve_leave", "reject_leave"),
                "policy_query": ("answer",),
            },
        ),
        planner_agent=planner_agent or TemplatePlannerAgent(
            templates={
                "onboarding": ("send_welcome_kit", "Send the onboarding kit and provision all required system accounts."),
                "leave_request": ("approve_leave", "Approve the leave request per the HR policy and update the HR system."),
                "policy_query": ("answer", "Provide the relevant policy excerpt from the HR handbook."),
            },
            default_action="answer",
            default_response="I can help with that HR query. Let me look up the relevant policy.",
        ),
        verifier_agent=BasicVerifierAgent(),
        retriever=retriever,
        approval_required_intents=("onboarding", "leave_request"),
    )


def finance_harness(retriever: KnowledgeBaseRetriever, planner_agent: Optional[Any] = None) -> PlanExecuteWeaver:
    """Finance & Accounting — Plan -> Execute pattern.

    Invoice approvals run through a multi-step verified execution sequence
    (validate -> check_budget -> approve_payment) before funds are committed.
    Any step blocked by policy causes an immediate escalation.

    To add new step sequences, extend step_map with the intent key and
    an ordered tuple of (step_name, action) pairs.
    """
    return PlanExecuteWeaver(
        signal_agent=KeywordSignalAgent(
            intents={
                "invoice_approval": ("invoice", "payment", "vendor", "bill", "purchase order"),
                "expense_claim": ("expense", "reimbursement", "receipt", "claim"),
                "budget_query": ("budget", "forecast", "spend", "allocation"),
            },
            high_risk_words=("overdue", "urgent", "penalty", "late fee"),
            missing_fields_by_intent={"invoice_approval": ("value_usd",)},
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("ask_for_info", "escalate", "log_note"),
            intent_actions={
                "invoice_approval": ("validate_invoice", "check_budget", "approve_payment"),
                "expense_claim": ("validate_receipt", "approve_expense"),
                "budget_query": ("summarize",),
            },
            blocked_when={"approve_payment": ("high_value",)},
        ),
        planner_agent=planner_agent or TemplatePlannerAgent(
            templates={
                "invoice_approval": ("approve_payment", "Invoice validated and approved for payment processing."),
                "expense_claim": ("approve_expense", "Expense claim validated and approved for reimbursement."),
                "budget_query": ("summarize", "Budget summary prepared with current spend vs allocation."),
            },
            default_action="ask_for_info",
            default_response="Please provide the invoice amount and vendor details.",
        ),
        verifier_agent=BasicVerifierAgent(),
        retriever=retriever,
        step_map={
            "invoice_approval": (
                ("validate_invoice", "validate_invoice"),
                ("check_budget",    "check_budget"),
                ("approve_payment", "approve_payment"),
            ),
            "expense_claim": (
                ("validate_receipt", "validate_receipt"),
                ("approve_expense",  "approve_expense"),
            ),
            "budget_query": (
                ("summarize_budget", "summarize"),
            ),
        },
    )


def fraud_harness(retriever: KnowledgeBaseRetriever, planner_agent: Optional[Any] = None) -> FraudOrchestratorWeaver:
    """Fraud Detection — FraudOrchestratorWeaver + 5 specialist sub-agents.

    Five sub-agents run in parallel, each checking one fraud dimension:
    - CardFraudSubAgent       : payment card fraud, card-not-present, skimming
    - AccountTakeoverSubAgent : ATO signals, device mismatch, credential abuse
    - SyntheticIdentitySubAgent: fake identity, new account + high-value tx
    - VelocitySubAgent        : transaction velocity abuse / bot-driven attacks
    - GeoRiskSubAgent         : geo anomalies, impossible travel, high-risk regions

    Risk thresholds:
      composite score 0-1  -> allow
      composite score 2-3  -> challenge_user  (step-up auth / OTP)
      composite score 4-5  -> flag_for_review (analyst queue)
      composite score 6-8  -> block_transaction
      composite score 9+   -> freeze_account

    To add a new fraud dimension: create a sub-agent with
    run(task) -> FraudFinding and add it to sub_agents.
    """
    return FraudOrchestratorWeaver(
        sub_agents=(
            CardFraudSubAgent(),
            AccountTakeoverSubAgent(),
            SyntheticIdentitySubAgent(),
            VelocitySubAgent(),
            GeoRiskSubAgent(),
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("allow", "challenge_user", "flag_for_review", "block_transaction", "freeze_account"),
            intent_actions={
                "allow":             ("allow",),
                "challenge_user":    ("challenge_user",),
                "flag_for_review":   ("flag_for_review",),
                "block_transaction": ("block_transaction",),
                "freeze_account":    ("freeze_account", "block_transaction"),
            },
        ),
        planner_agent=planner_agent or TemplatePlannerAgent(
            templates={
                "allow":             ("allow",             "Transaction cleared. No fraud signals detected across all checks."),
                "challenge_user":    ("challenge_user",    "Suspicious signals detected. Sending OTP step-up challenge to the account holder."),
                "flag_for_review":   ("flag_for_review",   "Multiple fraud signals detected. Flagging transaction for analyst review within 15 minutes."),
                "block_transaction": ("block_transaction", "High-confidence fraud detected. Transaction blocked. Customer notified via secure channel."),
                "freeze_account":    ("freeze_account",    "Critical fraud risk. Account frozen and transaction blocked. Fraud team alerted immediately."),
            },
            default_action="flag_for_review",
            default_response="Fraud risk inconclusive. Routing to analyst queue as a precaution.",
        ),
        verifier_agent=BasicVerifierAgent(),
        retriever=retriever,
        challenge_threshold=2,
        flag_threshold=4,
        block_threshold=6,
        freeze_threshold=9,
        fallback_action="flag_for_review",
        fallback_response="Fraud check inconclusive; routing to analyst queue.",
    )
