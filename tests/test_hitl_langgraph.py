import json
import unittest
from pathlib import Path
from langgraph.types import Command

from gantry.models import Task
from gantry.scenarios import weaver_for

ROOT = Path(__file__).resolve().parents[1]


class HITLLangGraphTest(unittest.TestCase):
    def test_hr_hitl_interrupt_and_resume(self) -> None:
        task_path = ROOT / "examples" / "hr" / "task_onboarding.json"
        task = Task(**json.loads(task_path.read_text(encoding="utf-8")))

        weaver = weaver_for("hr", ROOT / "examples")

        # First run: should trigger the interrupt and return a pending_approval outcome
        outcome1 = weaver.run(task)
        self.assertEqual(outcome1.final_action, "pending_approval")

        # Check graph state next is 'gate'
        config = {"configurable": {"thread_id": task.id}}
        state = weaver.graph.get_state(config)
        self.assertEqual(state.next, ("gate",))

        # Resume the graph with approved=True
        result2 = weaver.graph.invoke(Command(resume={"approved": True}), config=config)
        outcome2 = result2["outcome"]

        # The finalized run should complete the pipeline and take action
        self.assertEqual(outcome2.final_action, "send_welcome_kit")
        self.assertTrue(outcome2.verification.approved)
        self.assertIn("checkpoint:human_approved=true:resuming_pipeline", outcome2.audit_trail)


if __name__ == "__main__":
    unittest.main()
