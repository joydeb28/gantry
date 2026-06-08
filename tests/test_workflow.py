from pathlib import Path
import unittest

from support_weaver.models import ActionType, Ticket
from support_weaver.retrieval import TinyRetriever
from support_weaver.workflow import SupportWeaver


ROOT = Path(__file__).resolve().parents[1]


def build_weaver() -> SupportWeaver:
    return SupportWeaver(TinyRetriever.from_markdown_dir(ROOT / "examples" / "kb"))


class WorkflowTest(unittest.TestCase):
    def test_damaged_item_inside_policy_is_replaced(self) -> None:
        ticket = Ticket(
            id="T1",
            subject="Item arrived cracked",
            body="The lid is broken and the body is cracked.",
            order_value_usd=89,
            days_since_purchase=7,
        )

        resolution = build_weaver().resolve(ticket)

        self.assertEqual(resolution.final_action, ActionType.replace)
        self.assertTrue(resolution.verification.approved)
        self.assertEqual(resolution.evidence[0].title, "Damaged Item Replacement")

    def test_refund_outside_policy_escalates(self) -> None:
        ticket = Ticket(
            id="T2",
            subject="Refund request",
            body="I want a refund for an item I bought two months ago.",
            customer_tier="vip",
            order_value_usd=399,
            days_since_purchase=61,
        )

        resolution = build_weaver().resolve(ticket)

        self.assertEqual(resolution.final_action, ActionType.escalate)
        self.assertIn(ActionType.refund, resolution.policy.blocked_actions)
        self.assertTrue(resolution.verification.approved)

    def test_missing_order_value_asks_question(self) -> None:
        ticket = Ticket(
            id="T3",
            subject="Broken order",
            body="The product arrived broken.",
            days_since_purchase=5,
        )

        resolution = build_weaver().resolve(ticket)

        self.assertEqual(resolution.final_action, ActionType.ask_clarifying_question)
        self.assertEqual(resolution.triage.missing_fields, ("order_value_usd",))


if __name__ == "__main__":
    unittest.main()
