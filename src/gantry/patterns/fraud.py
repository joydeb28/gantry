"""Fraud Detection Pattern — LangGraph implementation of Orchestrator + Specialist Sub-agents.

An orchestrator fans out to five specialist sub-agents, each monitoring
one dimension of fraud risk. Findings are aggregated into a composite
fraud score. The orchestrator then applies a risk-threshold policy
to decide: allow, challenge, flag, block, or freeze.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, Protocol, TypedDict, runtime_checkable
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from ..generic_agents import BasicVerifierAgent, RulePolicyAgent, TemplatePlannerAgent
from ..models import Evidence, Outcome, Plan, PolicyDecision, Signal, Task, Verification, FraudFinding
from ..retrieval import KnowledgeBaseRetriever


@runtime_checkable
class FraudSubAgent(Protocol):
    """Protocol that every fraud sub-agent must satisfy."""

    name: str

    def run(self, task: Task) -> FraudFinding:
        ...


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


class FraudSubAgentState(TypedDict):
    task: Task
    agent_idx: int


class FraudState(TypedDict):
    task: Task
    findings: Annotated[list[FraudFinding], operator.add]
    signal: Optional[Signal]
    evidence: tuple[Evidence, ...]
    policy: Optional[PolicyDecision]
    draft: Optional[Plan]
    verification: Optional[Verification]
    outcome: Optional[Outcome]
    audit_trail: list[str]


class FraudOrchestratorWeaver:
    """Fraud Detection Orchestrator using LangGraph StateGraph."""

    def __init__(
        self,
        sub_agents: tuple[FraudSubAgent, ...],
        policy_agent: RulePolicyAgent,
        planner_agent: TemplatePlannerAgent,
        verifier_agent: BasicVerifierAgent,
        retriever: KnowledgeBaseRetriever,
        challenge_threshold: int = 2,
        flag_threshold: int = 4,
        block_threshold: int = 6,
        freeze_threshold: int = 9,
        fallback_action: str = "flag_for_review",
        fallback_response: str = "Fraud check inconclusive; routing to analyst queue.",
    ) -> None:
        self.sub_agents = sub_agents
        self.policy_agent = policy_agent
        self.planner_agent = planner_agent
        self.verifier_agent = verifier_agent
        self.retriever = retriever
        self.challenge_threshold = challenge_threshold
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold
        self.freeze_threshold = freeze_threshold
        self.fallback_action = fallback_action
        self.fallback_response = fallback_response

        builder = StateGraph(FraudState)
        builder.add_node("run_sub_agent", self._run_sub_agent_node)
        builder.add_node("aggregate", self._aggregate_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("policy", self._policy_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("verify", self._verify_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_conditional_edges(START, self._dispatch_sub_agents, ["run_sub_agent"])
        builder.add_edge("run_sub_agent", "aggregate")
        builder.add_edge("aggregate", "retrieve")
        builder.add_edge("retrieve", "policy")
        builder.add_edge("policy", "plan")
        builder.add_edge("plan", "verify")
        builder.add_edge("verify", "finalize")
        builder.add_edge("finalize", END)

        self.graph = builder.compile()

    def _dispatch_sub_agents(self, state: FraudState) -> list[Send]:
        return [
            Send("run_sub_agent", {"task": state["task"], "agent_idx": i})
            for i in range(len(self.sub_agents))
        ]

    def _run_sub_agent_node(self, state: FraudSubAgentState) -> dict:
        agent = self.sub_agents[state["agent_idx"]]
        finding = agent.run(state["task"])
        return {"findings": [finding]}

    def _aggregate_node(self, state: FraudState) -> dict:
        findings = state["findings"]
        audit = list(state.get("audit_trail") or [])

        for f in findings:
            audit.append(
                f"sub_agent:{f.name}:triggered={f.triggered}"
                f":risk_score={f.risk_score}"
                f":fraud_type={f.fraud_type}"
                f":reason={f.reason}"
            )

        composite_score = sum(f.risk_score for f in findings)
        triggered_findings = [f for f in findings if f.triggered]
        fraud_types = tuple(dict.fromkeys(f.fraud_type for f in triggered_findings))

        audit.append(
            f"composite_fraud_score:{composite_score}"
            f":sub_agents_triggered={len(triggered_findings)}/{len(self.sub_agents)}"
            f":fraud_types={','.join(fraud_types) or 'none'}"
        )

        if composite_score >= self.freeze_threshold:
            intent = "freeze_account"
        elif composite_score >= self.block_threshold:
            intent = "block_transaction"
        elif composite_score >= self.flag_threshold:
            intent = "flag_for_review"
        elif composite_score >= self.challenge_threshold:
            intent = "challenge_user"
        else:
            intent = "allow"

        risk = min(3, 1 + len(triggered_findings))
        signal = Signal(intent=intent, risk=risk, tags=fraud_types)
        audit.append(f"signal:intent={intent}:risk={risk}")

        return {"signal": signal, "audit_trail": audit}

    def _retrieve_node(self, state: FraudState) -> dict:
        task = state["task"]
        evidence = self.retriever.search(task.text)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"retrieval:documents={len(evidence)}")
        return {"evidence": evidence, "audit_trail": audit}

    def _policy_node(self, state: FraudState) -> dict:
        task = state["task"]
        signal = state["signal"]
        policy = self.policy_agent.run(task, signal)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"policy:allowed={','.join(policy.allowed_actions)}")
        return {"policy": policy, "audit_trail": audit}

    def _plan_node(self, state: FraudState) -> dict:
        task = state["task"]
        signal = state["signal"]
        evidence = state["evidence"]
        policy = state["policy"]
        draft = self.planner_agent.run(task, signal, evidence, policy)

        # Enrich responses with fraud context as in original design
        findings = state["findings"]
        composite_score = sum(f.risk_score for f in findings)
        triggered_findings = [f for f in findings if f.triggered]

        if triggered_findings:
            fraud_summary = "; ".join(
                f"{f.fraud_type}(score={f.risk_score})" for f in triggered_findings
            )
            draft = Plan(
                action=draft.action,
                confidence=draft.confidence,
                response=draft.response,
                internal_note=(
                    f"{draft.internal_note} | composite_score={composite_score} "
                    f"| fraud_signals=[{fraud_summary}]"
                ),
                citations=draft.citations,
            )

        audit = list(state.get("audit_trail") or [])
        audit.append(f"draft:action={draft.action}:confidence={draft.confidence}")
        return {"draft": draft, "audit_trail": audit}

    def _verify_node(self, state: FraudState) -> dict:
        draft = state["draft"]
        policy = state["policy"]
        evidence = state["evidence"]
        verification = self.verifier_agent.run(draft, policy, evidence)
        audit = list(state.get("audit_trail") or [])
        audit.append(f"verify:approved={verification.approved}:score={verification.score}")
        return {"verification": verification, "audit_trail": audit}

    def _finalize_node(self, state: FraudState) -> dict:
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
            note = f"Verifier blocked: {'; '.join(verification.findings)}"
            audit.append(f"fallback:{self.fallback_action}")

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
            "pattern:fraud_orchestrator",
            f"amount_usd:{task.metadata.get('amount_usd', 'unknown')}",
            f"account_age_days:{task.metadata.get('account_age_days', 'unknown')}",
        ]
        result = self.graph.invoke({"task": task, "audit_trail": initial_audit, "findings": []})
        return result["outcome"]


def _score_to_action(risk_score: int) -> str:
    """Map a sub-agent risk score to a recommended action."""
    if risk_score >= 3:
        return "freeze_account"
    if risk_score == 2:
        return "block_transaction"
    if risk_score == 1:
        return "challenge_user"
    return "allow"
