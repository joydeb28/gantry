"""Unit tests for Bucket 3 (LLM robustness) and Bucket 5 (Protocol types).

Bucket 3:
    TokenBudgetTest  — _build_evidence_text respects char budget, truncates, handles edge cases
    RetryLogicTest   — plan() retries on failure, raises after exhaustion, succeeds on N-th try

Bucket 5:
    ProtocolRuntimeCheckTest — Protocol isinstance checks work as expected
    AgentRecipeMisconfigTest — ValueError raised at construction with wrong agent
    AgentRecipeValidConfigTest — valid recipe (duck-typed) passes
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from gantry.models import (
    Evidence,
    Plan,
    PolicyDecision,
    Signal,
    Task,
    Verification,
    SignalAgentProtocol,
    PolicyAgentProtocol,
    PlannerAgentProtocol,
    VerifierAgentProtocol,
    AgentRecipe,
)
from gantry.llm import _build_evidence_text, _MAX_RETRIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evidence(text: str, score: float = 0.9, title: str = "Doc") -> Evidence:
    return Evidence(source="test", title=title, text=text, score=score)


def _make_task() -> Task:
    return Task(id="t1", use_case="test", title="Test", body="body")


def _make_signal() -> Signal:
    return Signal(intent="refund", risk=1)


def _make_policy() -> PolicyDecision:
    return PolicyDecision(allowed_actions=("refund",))


def _make_plan() -> Plan:
    return Plan(action="refund", confidence=0.9, response="Approved")


# ---------------------------------------------------------------------------
# Bucket 3 — Token budget
# ---------------------------------------------------------------------------

class TokenBudgetTest(unittest.TestCase):

    def test_empty_evidence_returns_placeholder(self):
        result = _build_evidence_text(())
        self.assertEqual(result, "No documents retrieved.")

    def test_short_evidence_not_truncated(self):
        e = _make_evidence("short text")
        result = _build_evidence_text((e,), char_budget=6000)
        self.assertNotIn("[truncated]", result)
        self.assertIn("short text", result)

    def test_long_evidence_is_truncated(self):
        long_text = "x" * 10_000
        e = _make_evidence(long_text)
        result = _build_evidence_text((e,), char_budget=500)
        self.assertIn("[truncated]", result)
        self.assertLessEqual(len(result), 600)  # some overhead for header

    def test_total_budget_respected(self):
        """Total chars of evidence text must not exceed budget + header overhead."""
        docs = tuple(
            _make_evidence("a" * 2000, title=f"Doc{i}")
            for i in range(5)
        )
        budget = 3_000
        result = _build_evidence_text(docs, char_budget=budget)
        # Strip the per-doc headers to measure only content
        # Result length should be within budget + reasonable header overhead per doc
        self.assertLessEqual(len(result), budget + 5 * 50)  # 50 chars per header

    def test_single_doc_gets_full_budget(self):
        e = _make_evidence("b" * 1000)
        result = _build_evidence_text((e,), char_budget=2000)
        self.assertNotIn("[truncated]", result)
        self.assertIn("b" * 1000, result)

    def test_includes_title_and_score(self):
        e = _make_evidence("content", score=0.85, title="Policy Doc")
        result = _build_evidence_text((e,))
        self.assertIn("Policy Doc", result)
        self.assertIn("0.85", result)

    def test_very_small_budget_still_returns_something(self):
        e = _make_evidence("hello world")
        result = _build_evidence_text((e,), char_budget=50)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_multiple_short_docs_all_included(self):
        docs = tuple(_make_evidence(f"doc{i}", title=f"T{i}") for i in range(3))
        result = _build_evidence_text(docs, char_budget=6000)
        for i in range(3):
            self.assertIn(f"doc{i}", result)


# ---------------------------------------------------------------------------
# Bucket 3 — Retry logic
# ---------------------------------------------------------------------------

class RetryLogicTest(unittest.TestCase):

    def _make_planner(self, mock_chain):
        from gantry.llm import LangChainPlanner
        planner = LangChainPlanner.__new__(LangChainPlanner)
        planner._chain = mock_chain
        planner._provider = "test"
        planner._model = "test-model"
        return planner

    def test_success_on_first_attempt(self):
        expected_plan = _make_plan()
        chain = MagicMock()
        chain.invoke.return_value = expected_plan

        planner = self._make_planner(chain)
        result = planner.plan(_make_task(), _make_signal(), (), _make_policy())

        self.assertEqual(result, expected_plan)
        self.assertEqual(chain.invoke.call_count, 1)

    def test_retries_on_transient_failure_then_succeeds(self):
        expected_plan = _make_plan()
        chain = MagicMock()
        chain.invoke.side_effect = [
            RuntimeError("network timeout"),
            RuntimeError("rate limit"),
            expected_plan,
        ]

        with patch("gantry.llm.time.sleep"):  # don't actually sleep in tests
            planner = self._make_planner(chain)
            result = planner.plan(_make_task(), _make_signal(), (), _make_policy())

        self.assertEqual(result, expected_plan)
        self.assertEqual(chain.invoke.call_count, 3)

    def test_raises_after_all_retries_exhausted(self):
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("always fails")

        with patch("gantry.llm.time.sleep"):
            planner = self._make_planner(chain)
            with self.assertRaises(RuntimeError) as ctx:
                planner.plan(_make_task(), _make_signal(), (), _make_policy())

        self.assertIn(str(_MAX_RETRIES), str(ctx.exception))
        self.assertEqual(chain.invoke.call_count, _MAX_RETRIES)

    def test_retry_count_matches_max_retries(self):
        """chain.invoke must be called exactly _MAX_RETRIES times on exhaustion."""
        chain = MagicMock()
        chain.invoke.side_effect = ConnectionError("unreachable")

        with patch("gantry.llm.time.sleep"):
            planner = self._make_planner(chain)
            with self.assertRaises(RuntimeError):
                planner.plan(_make_task(), _make_signal(), (), _make_policy())

        self.assertEqual(chain.invoke.call_count, _MAX_RETRIES)


# ---------------------------------------------------------------------------
# Bucket 5 — Protocol runtime checks
# ---------------------------------------------------------------------------

class ProtocolRuntimeCheckTest(unittest.TestCase):

    def test_signal_agent_protocol_is_runtime_checkable(self):
        class GoodSignalAgent:
            def run(self, task): ...
        self.assertIsInstance(GoodSignalAgent(), SignalAgentProtocol)

    def test_policy_agent_protocol_is_runtime_checkable(self):
        class GoodPolicyAgent:
            def run(self, task, signal): ...
        self.assertIsInstance(GoodPolicyAgent(), PolicyAgentProtocol)

    def test_planner_agent_protocol_is_runtime_checkable(self):
        class GoodPlannerAgent:
            def run(self, task, signal, evidence, policy): ...
        self.assertIsInstance(GoodPlannerAgent(), PlannerAgentProtocol)

    def test_verifier_agent_protocol_is_runtime_checkable(self):
        class GoodVerifierAgent:
            def run(self, draft, policy, evidence): ...
        self.assertIsInstance(GoodVerifierAgent(), VerifierAgentProtocol)

    def test_object_without_run_fails_protocol_check(self):
        class NoRunMethod:
            pass
        self.assertNotIsInstance(NoRunMethod(), SignalAgentProtocol)

    def test_protocols_exported_from_gantry(self):
        """All four Protocol classes must be importable from top-level gantry package."""
        import gantry
        for name in ["SignalAgentProtocol", "PolicyAgentProtocol",
                     "PlannerAgentProtocol", "VerifierAgentProtocol"]:
            self.assertTrue(hasattr(gantry, name), f"gantry.{name} not exported")


class AgentRecipeMisconfigTest(unittest.TestCase):

    def _good_agents(self):
        class SA:
            def run(self, task): return Signal(intent="x", risk=0)
        class PA:
            def run(self, task, signal): return PolicyDecision(allowed_actions=("x",))
        class PLA:
            def run(self, task, signal, evidence, policy): return _make_plan()
        class VA:
            def run(self, draft, policy, evidence): return Verification(approved=True)
        return SA(), PA(), PLA(), VA()

    def test_valid_recipe_passes(self):
        sa, pa, pla, va = self._good_agents()
        recipe = AgentRecipe(
            use_case="test",
            signal_agent=sa,
            policy_agent=pa,
            planner_agent=pla,
            verifier_agent=va,
        )
        self.assertEqual(recipe.use_case, "test")

    def test_wrong_signal_agent_raises_at_construction(self):
        _, pa, pla, va = self._good_agents()
        with self.assertRaises(Exception) as ctx:
            AgentRecipe(
                use_case="test",
                signal_agent="not_an_agent",
                policy_agent=pa,
                planner_agent=pla,
                verifier_agent=va,
            )
        self.assertIn("signal_agent", str(ctx.exception))

    def test_error_message_names_all_bad_agents(self):
        """When multiple agents are wrong, error must name all of them."""
        with self.assertRaises(Exception) as ctx:
            AgentRecipe(
                use_case="test",
                signal_agent=object(),
                policy_agent=object(),
                planner_agent=object(),
                verifier_agent=object(),
            )
        msg = str(ctx.exception)
        for name in ["signal_agent", "policy_agent", "planner_agent", "verifier_agent"]:
            self.assertIn(name, msg)

    def test_duck_typed_agent_passes_without_inheritance(self):
        """An agent that doesn't subclass anything but has the right .run() passes."""
        class CustomSignalAgent:  # no inheritance at all
            def run(self, task):
                return Signal(intent="custom", risk=0)

        sa, _, pla, va = self._good_agents()
        _, pa, _, _ = self._good_agents()

        recipe = AgentRecipe(
            use_case="test",
            signal_agent=CustomSignalAgent(),
            policy_agent=pa,
            planner_agent=pla,
            verifier_agent=va,
        )
        self.assertIsNotNone(recipe)

    def test_real_scenario_recipe_passes(self):
        """support_recipe() from gantry.scenarios must pass validation."""
        from gantry.scenarios import support_recipe
        recipe = support_recipe()
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.use_case, "support")


if __name__ == "__main__":
    unittest.main()
