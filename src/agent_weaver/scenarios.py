from __future__ import annotations

from pathlib import Path

from .generic_agents import BasicVerifierAgent, KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent
from .pattern import AgentRecipe, ClaimWeaver
from .retrieval import TinyRetriever


def recipe_for(use_case: str) -> AgentRecipe:
    recipes = {
        "support": support_recipe,
        "guardrail": guardrail_recipe,
        "crm": crm_recipe,
        "research": research_recipe,
        "coding": coding_recipe,
    }
    try:
        return recipes[use_case]()
    except KeyError as exc:
        raise ValueError(f"Unknown use case '{use_case}'. Choose one of: {', '.join(sorted(recipes))}") from exc


def weaver_for(use_case: str, kb_root: str | Path = "examples") -> ClaimWeaver:
    kb_path = Path(kb_root) / use_case / "kb"
    return ClaimWeaver(recipe_for(use_case), TinyRetriever.from_markdown_dir(kb_path))


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


def guardrail_recipe() -> AgentRecipe:
    return AgentRecipe(
        use_case="guardrail",
        signal_agent=KeywordSignalAgent(
            intents={
                "pii": ("ssn", "credit card", "passport", "phone number"),
                "unsafe_prompt": ("ignore previous", "jailbreak", "bypass policy"),
                "external_send": ("email customer", "post publicly", "send outside"),
            },
            high_risk_words=("ssn", "credit card", "jailbreak", "bypass"),
        ),
        policy_agent=RulePolicyAgent(
            base_actions=("allow", "redact", "escalate", "log_only"),
            intent_actions={"pii": ("redact",), "unsafe_prompt": ("escalate",), "external_send": ("escalate",)},
            blocked_when={"allow": ("external_send",)},
        ),
        planner_agent=TemplatePlannerAgent(
            templates={
                "pii": ("redact", "Redact sensitive fields before continuing."),
                "unsafe_prompt": ("escalate", "Route this prompt for policy review."),
                "external_send": ("escalate", "Human approval is required before external sharing."),
            },
            default_action="allow",
            default_response="No guardrail violation was found.",
        ),
        verifier_agent=BasicVerifierAgent(),
        fallback_action="escalate",
        fallback_response="Guardrail verification failed; route to review.",
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
