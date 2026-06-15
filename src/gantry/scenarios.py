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
from .models import AgentRecipe, SubAgentFinding, FraudFinding, Task, Plan
from .patterns.pipeline import PipelineWeaver
from .patterns.hitl import HumanInTheLoopWeaver
from .patterns.orchestrator import OrchestratorWeaver
from .patterns.parallel_orchestrator import ParallelOrchestratorWeaver
from .patterns.plan_execute import PlanExecuteWeaver
from .patterns.reflection import CriticAgent, ReflectionWeaver
from .patterns.router import RouterWeaver, SpecialistAgent
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
) -> ParallelOrchestratorWeaver:
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

class PIISubAgent:
    """Detects personally identifiable information."""

    name: str = "pii_detector"
    keywords: tuple[str, ...] = (
        "ssn",
        "credit card",
        "passport",
        "phone number",
        "date of birth",
        "bank account",
        "social security",
    )

    def run(self, task: Task) -> SubAgentFinding:
        text = task.text.lower()
        matches = [k for k in self.keywords if k in text]
        triggered = bool(matches)
        return SubAgentFinding(
            name=self.name,
            triggered=triggered,
            reason=f"PII detected: {', '.join(matches)}" if matches else "No PII detected.",
            risk_delta=2 if triggered else 0,
        )


class SafetySubAgent:
    """Detects unsafe prompt patterns and jailbreak attempts."""

    name: str = "safety_checker"
    keywords: tuple[str, ...] = (
        "ignore previous",
        "jailbreak",
        "bypass policy",
        "ignore instructions",
        "act as",
        "pretend you are",
        "disregard",
    )

    def run(self, task: Task) -> SubAgentFinding:
        text = task.text.lower()
        matches = [k for k in self.keywords if k in text]
        triggered = bool(matches)
        return SubAgentFinding(
            name=self.name,
            triggered=triggered,
            reason=f"Unsafe prompt pattern detected: {', '.join(matches)}" if matches else "No unsafe patterns.",
            risk_delta=3 if triggered else 0,
        )


class ExternalSendSubAgent:
    """Detects attempts to send data outside the system."""

    name: str = "external_send_detector"
    keywords: tuple[str, ...] = (
        "email customer",
        "post publicly",
        "send outside",
        "forward to",
        "share with",
        "send to external",
    )

    def run(self, task: Task) -> SubAgentFinding:
        text = task.text.lower()
        matches = [k for k in self.keywords if k in text]
        triggered = bool(matches)
        return SubAgentFinding(
            name=self.name,
            triggered=triggered,
            reason=f"External data send attempt: {', '.join(matches)}" if matches else "No external send detected.",
            risk_delta=2 if triggered else 0,
        )


class ToneSubAgent:
    """Detects hostile or threatening language."""

    name: str = "tone_guard"
    keywords: tuple[str, ...] = (
        "threat",
        "sue",
        "lawyer",
        "hate",
        "destroy",
        "attack",
        "burn",
    )

    def run(self, task: Task) -> SubAgentFinding:
        text = task.text.lower()
        matches = [k for k in self.keywords if k in text]
        triggered = bool(matches)
        return SubAgentFinding(
            name=self.name,
            triggered=triggered,
            reason=f"Hostile tone detected: {', '.join(matches)}" if matches else "Tone acceptable.",
            risk_delta=1 if triggered else 0,
        )


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
        intent_map={
            "pii_detector": "pii",
            "safety_checker": "unsafe_prompt",
            "external_send_detector": "external_send",
            "tone_guard": "unsafe_prompt",
        },
        default_intent="safe",
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


def _score_to_action(risk_score: int) -> str:
    """Map a sub-agent risk score to a recommended action."""
    if risk_score >= 3:
        return "freeze_account"
    if risk_score == 2:
        return "block_transaction"
    if risk_score == 1:
        return "challenge_user"
    return "allow"


class CardFraudSubAgent:
    """Detects payment card fraud signals."""

    name: str = "card_fraud_detector"
    high_amount_threshold: float = 5000.0
    keywords: tuple[str, ...] = (
        "card not present",
        "multiple cards",
        "declined then approved",
        "chargeback",
        "stolen card",
        "card cloning",
        "skimming",
        "cvv mismatch",
        "card testing",
    )

    def run(self, task: Task) -> FraudFinding:
        text = task.text.lower()
        keyword_hits = [k for k in self.keywords if k in text]
        amount = float(task.metadata.get("amount_usd", 0))
        high_amount = amount > self.high_amount_threshold

        triggered = bool(keyword_hits) or high_amount
        risk_score = 0
        reasons: list[str] = []

        if keyword_hits:
            risk_score += 2
            reasons.append(f"Card fraud keywords: {', '.join(keyword_hits)}")
        if high_amount:
            risk_score += 1
            reasons.append(f"High transaction amount: ${amount:,.2f}")

        risk_score = min(risk_score, 3)
        action = _score_to_action(risk_score)

        return FraudFinding(
            name=self.name,
            triggered=triggered,
            fraud_type="payment_card_fraud",
            reason="; ".join(reasons) if reasons else "No card fraud signals detected.",
            risk_score=risk_score,
            recommended_action=action,
        )


class AccountTakeoverSubAgent:
    """Detects Account Takeover (ATO) signals."""

    name: str = "ato_detector"
    keywords: tuple[str, ...] = (
        "password reset",
        "login attempt",
        "new device",
        "credential",
        "session hijack",
        "unauthorized access",
        "account locked",
        "multiple failed login",
        "suspicious login",
        "device change",
    )

    def run(self, task: Task) -> FraudFinding:
        text = task.text.lower()
        keyword_hits = [k for k in self.keywords if k in text]
        device_mismatch = not bool(task.metadata.get("device_fingerprint_match", True))
        recent_password_reset = bool(task.metadata.get("recent_password_reset", False))

        triggered = bool(keyword_hits) or device_mismatch or recent_password_reset
        risk_score = 0
        reasons: list[str] = []

        if keyword_hits:
            risk_score += 2
            reasons.append(f"ATO signals: {', '.join(keyword_hits)}")
        if device_mismatch:
            risk_score += 2
            reasons.append("Device fingerprint mismatch detected.")
        if recent_password_reset:
            risk_score += 1
            reasons.append("Recent password reset before high-value transaction.")

        risk_score = min(risk_score, 3)
        action = _score_to_action(risk_score)

        return FraudFinding(
            name=self.name,
            triggered=triggered,
            fraud_type="account_takeover",
            reason="; ".join(reasons) if reasons else "No ATO signals detected.",
            risk_score=risk_score,
            recommended_action=action,
        )


class SyntheticIdentitySubAgent:
    """Detects synthetic identity fraud signals."""

    name: str = "synthetic_identity_detector"
    new_account_days_threshold: int = 30
    new_account_high_amount: float = 1000.0
    keywords: tuple[str, ...] = (
        "synthetic identity",
        "fake identity",
        "fabricated",
        "identity mismatch",
        "ssn mismatch",
        "address mismatch",
        "multiple identities",
        "bust out",
        "credit washing",
    )

    def run(self, task: Task) -> FraudFinding:
        text = task.text.lower()
        keyword_hits = [k for k in self.keywords if k in text]
        account_age = int(task.metadata.get("account_age_days", 999))
        amount = float(task.metadata.get("amount_usd", 0))

        new_account_high_tx = (
            account_age < self.new_account_days_threshold
            and amount > self.new_account_high_amount
        )

        triggered = bool(keyword_hits) or new_account_high_tx
        risk_score = 0
        reasons: list[str] = []

        if keyword_hits:
            risk_score += 2
            reasons.append(f"Synthetic identity keywords: {', '.join(keyword_hits)}")
        if new_account_high_tx:
            reasons.append(
                f"New account ({account_age}d old) attempting high-value "
                f"transaction (${amount:,.2f})."
            )
            risk_score += 2

        risk_score = min(risk_score, 3)
        action = _score_to_action(risk_score)

        return FraudFinding(
            name=self.name,
            triggered=triggered,
            fraud_type="synthetic_identity",
            reason="; ".join(reasons) if reasons else "No synthetic identity signals detected.",
            risk_score=risk_score,
            recommended_action=action,
        )


class VelocitySubAgent:
    """Detects transaction velocity abuse."""

    name: str = "velocity_checker"
    velocity_threshold_1h: int = 5
    velocity_threshold_10min: int = 3

    def run(self, task: Task) -> FraudFinding:
        tx_1h = int(task.metadata.get("transaction_count_1h", 0))
        tx_10min = int(task.metadata.get("transaction_count_10min", 0))

        high_velocity_1h = tx_1h >= self.velocity_threshold_1h
        high_velocity_10min = tx_10min >= self.velocity_threshold_10min

        triggered = high_velocity_1h or high_velocity_10min
        risk_score = 0
        reasons: list[str] = []

        if high_velocity_10min:
            risk_score += 3
            reasons.append(f"Critical velocity: {tx_10min} transactions in last 10 minutes.")
        elif high_velocity_1h:
            risk_score += 2
            reasons.append(f"High velocity: {tx_1h} transactions in last hour.")

        risk_score = min(risk_score, 3)
        action = _score_to_action(risk_score)

        return FraudFinding(
            name=self.name,
            triggered=triggered,
            fraud_type="velocity_abuse",
            reason="; ".join(reasons) if reasons else f"Velocity normal ({tx_1h}/h, {tx_10min}/10min).",
            risk_score=risk_score,
            recommended_action=action,
        )


class GeoRiskSubAgent:
    """Detects geographic risk signals."""

    name: str = "geo_risk_detector"
    high_risk_countries: tuple[str, ...] = (
        "ng", "gh", "ro", "ru", "ua", "cn", "id",
    )
    high_risk_keywords: tuple[str, ...] = (
        "impossible travel",
        "location mismatch",
        "vpn detected",
        "tor exit",
        "proxy detected",
        "high risk country",
        "sanctioned country",
    )

    def run(self, task: Task) -> FraudFinding:
        text = task.text.lower()
        keyword_hits = [k for k in self.high_risk_keywords if k in text]
        country = str(task.metadata.get("country_code", "")).lower()
        high_risk_country = country in self.high_risk_countries
        impossible_travel = bool(task.metadata.get("impossible_travel", False))

        triggered = bool(keyword_hits) or high_risk_country or impossible_travel
        risk_score = 0
        reasons: list[str] = []

        if keyword_hits:
            risk_score += 2
            reasons.append(f"Geo risk keywords: {', '.join(keyword_hits)}")
        if high_risk_country:
            risk_score += 1
            reasons.append(f"Transaction from high-risk country: {country.upper()}.")
        if impossible_travel:
            risk_score += 3
            reasons.append("Impossible travel detected: two locations too far apart in time.")

        risk_score = min(risk_score, 3)
        action = _score_to_action(risk_score)

        return FraudFinding(
            name=self.name,
            triggered=triggered,
            fraud_type="geo_risk",
            reason="; ".join(reasons) if reasons else "No geo risk detected.",
            risk_score=risk_score,
            recommended_action=action,
        )


def _aggregate_fraud_findings(findings: list[FraudFinding]) -> tuple[str, int, list[str], list[str]]:
    audit = []
    for f in findings:
        audit.append(
            f"sub_agent:{f.name}:triggered={f.triggered}"
            f":risk_score={f.risk_score}"
            f":fraud_type={f.fraud_type}"
            f":reason={f.reason}"
        )

    composite_score = sum(f.risk_score for f in findings)
    triggered_findings = [f for f in findings if f.triggered]
    fraud_types = list(dict.fromkeys(f.fraud_type for f in triggered_findings))

    audit.append(
        f"composite_fraud_score:{composite_score}"
        f":sub_agents_triggered={len(triggered_findings)}/{len(findings)}"
        f":fraud_types={','.join(fraud_types) or 'none'}"
    )

    if composite_score >= 9:
        intent = "freeze_account"
    elif composite_score >= 6:
        intent = "block_transaction"
    elif composite_score >= 4:
        intent = "flag_for_review"
    elif composite_score >= 2:
        intent = "challenge_user"
    else:
        intent = "allow"

    risk = min(3, 1 + len(triggered_findings))
    return intent, risk, fraud_types, audit


def _enrich_fraud_plan(draft: Plan, findings: list[Any]) -> Plan:
    composite_score = sum(f.risk_score for f in findings)
    triggered_findings = [f for f in findings if f.triggered]

    if triggered_findings:
        fraud_summary = "; ".join(
            f"{f.fraud_type}(score={f.risk_score})" for f in triggered_findings
        )
        return Plan(
            action=draft.action,
            confidence=draft.confidence,
            response=draft.response,
            internal_note=(
                f"{draft.internal_note} | composite_score={composite_score} "
                f"| fraud_signals=[{fraud_summary}]"
            ),
            citations=draft.citations,
        )
    return draft


def _initial_fraud_audit(task: Task) -> list[str]:
    return [
        f"task:{task.id}:received",
        f"use_case:{task.use_case}",
        "pattern:fraud_orchestrator",
        f"amount_usd:{task.metadata.get('amount_usd', 'unknown')}",
        f"account_age_days:{task.metadata.get('account_age_days', 'unknown')}",
    ]


def fraud_harness(retriever: KnowledgeBaseRetriever, planner_agent: Optional[Any] = None) -> ParallelOrchestratorWeaver:
    """Fraud Detection — ParallelOrchestratorWeaver + 5 specialist sub-agents.

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
    return ParallelOrchestratorWeaver(
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
        aggregation_policy=_aggregate_fraud_findings,
        plan_enrichment_fn=_enrich_fraud_plan,
        initial_audit_fn=_initial_fraud_audit,
        fallback_action="flag_for_review",
        fallback_response="Fraud check inconclusive; routing to analyst queue.",
    )
