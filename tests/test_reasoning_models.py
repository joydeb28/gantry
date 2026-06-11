import unittest
from unittest.mock import MagicMock

from gantry.llm import LangChainPlannerAgent, LangChainPlanner
from gantry.models import Evidence, PolicyDecision, Signal, Task, Plan


class ReasoningModelsTest(unittest.TestCase):
    def test_langchain_planner_agent(self) -> None:
        mock_planner = MagicMock(spec=LangChainPlanner)
        mock_planner.plan.return_value = Plan(
            action="write_patch",
            confidence=0.95,
            response="Patch generated.",
            internal_note="Mock note",
        )

        agent = LangChainPlannerAgent(mock_planner)
        plan = agent.run(
            task=Task(id="COD-1", use_case="coding", title="Bug", body="Fix it"),
            signal=Signal(intent="bugfix", risk=1),
            evidence=(),
            policy=PolicyDecision(allowed_actions=("write_patch", "explain")),
        )

        self.assertEqual(plan.action, "write_patch")
        self.assertEqual(plan.confidence, 0.95)
        mock_planner.plan.assert_called_once()


if __name__ == "__main__":
    unittest.main()
