"""Unit tests for Bucket 1 fixes (C-2 null-state cascade, C-6 safe_node).

Tests:
    - safe_node returns a defensive fallback copy on every failure (C-6)
    - Upstream node crash leaves None in state; finalize_node degrades gracefully (C-2)
"""

from __future__ import annotations

import unittest
from pathlib import Path

from gantry.generic_agents import safe_node
from gantry.models import Evidence, Task

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# C-6: safe_node defensive copy tests
# ---------------------------------------------------------------------------

class SafeNodeDefensiveCopyTest(unittest.TestCase):

    def test_returns_fallback_on_exception(self):
        """safe_node must return the fallback dict when the wrapped fn raises."""
        def failing(state):
            raise ValueError("simulated failure")

        wrapped = safe_node(failing, fallback={"signal": None, "audit_trail": ["error"]})
        result = wrapped({})
        self.assertEqual(result, {"signal": None, "audit_trail": ["error"]})

    def test_fallback_is_defensive_copy(self):
        """Mutating a nested value in the returned fallback must not corrupt subsequent calls.

        Pre-fix bug: _fallback was returned by reference (and later dict() was a
        shallow copy), so r1["audit_trail"].append(...) would mutate the inner list
        still shared with _fallback, corrupting r2 on the next invocation.
        The fix is copy.deepcopy(_fallback) — full isolation at every level.
        """
        def always_fails(state):
            raise RuntimeError("always")

        wrapped = safe_node(always_fails, fallback={"audit_trail": ["base_error"]})

        r1 = wrapped({})
        r1["audit_trail"].append("mutated_by_caller")   # caller mutates returned dict

        r2 = wrapped({})
        self.assertNotIn(
            "mutated_by_caller", r2["audit_trail"],
            "safe_node returned a shared dict reference — fallback was corrupted.",
        )

    def test_preserves_fn_name(self):
        """safe_node must preserve __name__ and embed __qualname__ for logging."""
        def my_special_node(state):
            return {}

        wrapped = safe_node(my_special_node)
        self.assertEqual(wrapped.__name__, "my_special_node")
        # __qualname__ includes enclosing scope for nested functions, so use assertIn
        self.assertIn("my_special_node", wrapped.__qualname__)

    def test_keyboard_interrupt_propagates(self):
        """safe_node must NOT swallow KeyboardInterrupt."""
        def raises_kbi(state):
            raise KeyboardInterrupt

        wrapped = safe_node(raises_kbi, fallback={"x": 1})
        with self.assertRaises(KeyboardInterrupt):
            wrapped({})

    def test_system_exit_propagates(self):
        """safe_node must NOT swallow SystemExit."""
        def raises_sysexit(state):
            raise SystemExit(1)

        wrapped = safe_node(raises_sysexit, fallback={"x": 1})
        with self.assertRaises(SystemExit):
            wrapped({})

    def test_returns_fn_result_on_success(self):
        """safe_node must transparently return the result when fn succeeds."""
        def good_node(state):
            return {"result": "ok"}

        wrapped = safe_node(good_node, fallback={"result": "fallback"})
        self.assertEqual(wrapped({}), {"result": "ok"})

    def test_empty_fallback_default(self):
        """safe_node with no fallback arg returns empty dict on failure."""
        def bad_node(state):
            raise Exception("boom")

        wrapped = safe_node(bad_node)
        self.assertEqual(wrapped({}), {})


# ---------------------------------------------------------------------------
# C-2: Null-state cascade tests (upstream node failure → graceful finalize)
# ---------------------------------------------------------------------------

class _FailingPolicyAgent:
    """Stub agent that always raises — simulates an upstream service outage."""
    def run(self, task, signal):
        raise RuntimeError("Policy service unavailable")


class _MockRetriever:
    """Minimal retriever — no LlamaIndex needed."""
    def search(self, query: str) -> tuple[Evidence, ...]:
        return (Evidence(
            source="mock",
            title="Policy Doc",
            text="Standard replacement policy.",
            score=0.9,
        ),)


class NullStateCascadeTest(unittest.TestCase):
    """Verify a crashing upstream node does NOT crash finalize_node.

    Before the fix:
        policy_node crash → policy=None
        → plan_node crash → draft=None
        → verify_node crash → verification=None
        → finalize_node raises ValidationError (Outcome requires non-None fields)

    After the fix:
        finalize_node detects None fields and returns a typed escalation Outcome
        with final_action = fallback_action.
    """

    def _make_task(self) -> Task:
        return Task(
            id="TEST-001",
            use_case="support",
            title="Broken lid",
            body="The lid is cracked and damaged.",
            metadata={"value_usd": 50, "days_since_event": 5},
        )

    def test_pipeline_degrades_gracefully_when_policy_fails(self):
        """Pipeline finalize_node returns escalation Outcome when policy=None."""
        from gantry.patterns.pipeline import PipelineWeaver
        from gantry.scenarios import support_recipe

        recipe = support_recipe()
        bad_recipe = recipe.model_copy(update={"policy_agent": _FailingPolicyAgent()})
        weaver = PipelineWeaver(recipe=bad_recipe, retriever=_MockRetriever())

        # Must NOT raise — must return a graceful escalation Outcome
        outcome = weaver.run(self._make_task())

        self.assertIsNotNone(outcome, "run() returned None — expected an Outcome")
        self.assertEqual(
            outcome.final_action, recipe.fallback_action,
            f"Expected fallback '{recipe.fallback_action}', got '{outcome.final_action}'",
        )
        self.assertFalse(outcome.verification.approved)

    def test_audit_trail_records_which_fields_were_none(self):
        """The escalation audit entry must name the missing fields for observability."""
        from gantry.patterns.pipeline import PipelineWeaver
        from gantry.scenarios import support_recipe

        recipe = support_recipe()
        bad_recipe = recipe.model_copy(update={"policy_agent": _FailingPolicyAgent()})
        weaver = PipelineWeaver(recipe=bad_recipe, retriever=_MockRetriever())

        outcome = weaver.run(self._make_task())
        upstream_entries = [e for e in outcome.audit_trail if "upstream_failure" in e]

        self.assertTrue(
            upstream_entries,
            "audit_trail must contain an 'upstream_failure' entry",
        )
        # 'policy' must be named since _FailingPolicyAgent raises
        self.assertIn("policy", upstream_entries[0])


if __name__ == "__main__":
    unittest.main()
