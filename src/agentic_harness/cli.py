from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .generic_agents import ReasoningPlannerAgent
from .llm import SMALL_REASONING_MODELS, client_from_profile
from .models import Task
from .scenarios import weaver_for

ALL_USE_CASES = [
    "coding",
    "crm",
    "finance",
    "guardrail",
    "hr",
    "it",
    "legal",
    "research",
    "support",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agentic Harness — run any use case with its native agentic pattern.\n\n"
            "Patterns by use case:\n"
            "  support, crm, research, coding -> Pipeline (sequential claim chain)\n"
            "  guardrail                       -> Orchestrator + sub-agents\n"
            "  it                              -> Router + specialist dispatch\n"
            "  legal                           -> Reflection / critic loop\n"
            "  hr                              -> Human-in-the-loop\n"
            "  finance                         -> Plan + execute\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--use-case", required=True, choices=ALL_USE_CASES)
    parser.add_argument("--task", required=True, help="Path to a task JSON file.")
    parser.add_argument("--examples-root", default="examples", help="Directory containing use-case KBs.")
    parser.add_argument("--planner", choices=["template", "ollama", "vllm"], default="template")
    parser.add_argument("--model-profile", choices=sorted(SMALL_REASONING_MODELS), default=None)
    parser.add_argument("--base-url", default=None, help="Override Ollama or vLLM base URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_data = json.loads(Path(args.task).read_text(encoding="utf-8"))
    task = Task(**task_data)

    weaver = weaver_for(args.use_case, kb_root=args.examples_root)

    # Optionally swap in a reasoning model planner for pipeline-pattern use cases
    if args.planner != "template" and hasattr(weaver, "recipe"):
        profile_name = args.model_profile or f"{args.planner}-qwen3-4b"
        client = client_from_profile(profile_name, args.base_url)
        weaver.recipe = replace(
            weaver.recipe,
            planner_agent=ReasoningPlannerAgent(client=client, fallback=weaver.recipe.planner_agent),
        )

    outcome = weaver.run(task)
    print(json.dumps(asdict(outcome), indent=2, default=str))


if __name__ == "__main__":
    main()
