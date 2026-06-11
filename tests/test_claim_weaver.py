from pathlib import Path
import unittest

from gantry.models import Task
from gantry.scenarios import weaver_for


ROOT = Path(__file__).resolve().parents[1]


def run_task(use_case: str, task_name: str):
    task_path = ROOT / "examples" / use_case / task_name
    task = Task(**__import__("json").loads(task_path.read_text(encoding="utf-8")))
    return weaver_for(use_case, ROOT / "examples").run(task)


class ClaimWeaverTest(unittest.TestCase):
    def test_support_replacement(self) -> None:
        outcome = run_task("support", "task_replacement.json")

        self.assertEqual(outcome.final_action, "replace")
        self.assertTrue(outcome.verification.approved)

    def test_support_refund_escalates_when_policy_blocks(self) -> None:
        outcome = run_task("support", "task_refund_escalate.json")

        self.assertEqual(outcome.final_action, "escalate")
        self.assertIn("refund", outcome.policy.blocked_actions)

    def test_guardrail_redacts_pii(self) -> None:
        outcome = run_task("guardrail", "task_pii.json")

        self.assertEqual(outcome.final_action, "redact")
        self.assertIn("Data Safety Guardrail", outcome.draft.citations)

    def test_crm_schedules_hot_lead_demo(self) -> None:
        outcome = run_task("crm", "task_hot_lead.json")

        self.assertEqual(outcome.final_action, "schedule_demo")

    def test_research_builds_reading_list(self) -> None:
        outcome = run_task("research", "task_lit_scan.json")

        self.assertEqual(outcome.final_action, "build_reading_list")

    def test_coding_bugfix_writes_patch(self) -> None:
        outcome = run_task("coding", "task_bugfix.json")

        self.assertEqual(outcome.final_action, "write_patch")


if __name__ == "__main__":
    unittest.main()
