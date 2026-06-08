import json
import unittest
from unittest.mock import patch

from agent_weaver.generic_agents import ReasoningPlannerAgent, TemplatePlannerAgent
from agent_weaver.llm import (
    OllamaReasoningClient,
    ReasoningResponse,
    SMALL_REASONING_MODELS,
    VLLMReasoningClient,
)
from agent_weaver.models import Evidence, PolicyDecision, Signal, Task


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeReasoningClient:
    def chat(self, messages):
        return ReasoningResponse(
            content='{"action": "write_patch", "confidence": 0.91, "response": "Patch and test the bug."}',
            thinking="Checked policy, evidence, and allowed actions.",
        )


class ReasoningModelsTest(unittest.TestCase):
    def test_ollama_client_requests_thinking(self) -> None:
        with patch("urllib.request.urlopen", return_value=FakeHTTPResponse({"message": {"content": "ok", "thinking": "trace"}})) as urlopen:
            response = OllamaReasoningClient().chat([{"role": "user", "content": "plan"}])

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertTrue(payload["think"])
        self.assertEqual(response.content, "ok")
        self.assertEqual(response.thinking, "trace")

    def test_vllm_client_uses_qwen3_reasoning_controls(self) -> None:
        profile = SMALL_REASONING_MODELS["vllm-qwen3-0.6b"]
        with patch(
            "urllib.request.urlopen",
            return_value=FakeHTTPResponse({"choices": [{"message": {"content": "ok", "reasoning": "trace"}}]}),
        ) as urlopen:
            response = VLLMReasoningClient(profile).chat([{"role": "user", "content": "plan"}])

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(payload["thinking_token_budget"], 256)
        self.assertEqual(response.thinking, "trace")

    def test_reasoning_planner_clamps_model_to_allowed_actions(self) -> None:
        planner = ReasoningPlannerAgent(
            client=FakeReasoningClient(),
            fallback=TemplatePlannerAgent({}, default_action="explain", default_response="Explain only."),
        )
        plan = planner.run(
            task=Task(id="COD-1", use_case="coding", title="Bug", body="Fix failing test"),
            signal=Signal(intent="bugfix", risk=1),
            evidence=(Evidence("kb.md", "Coding Rules", "Patch and test.", 1.0),),
            policy=PolicyDecision(allowed_actions=("escalate", "explain"), blocked_actions=("write_patch",), reason="No patch allowed."),
        )

        self.assertEqual(plan.action, "escalate")
        self.assertIn("reasoning_model=enabled", plan.internal_note)


if __name__ == "__main__":
    unittest.main()
