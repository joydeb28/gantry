from __future__ import annotations

import argparse
import json
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path

from .generic_agents import ReasoningPlannerAgent
from .llm import SMALL_REASONING_MODELS, client_from_profile
from .models import Task
from .pattern import ClaimWeaver
from .retrieval import TinyRetriever
from .scenarios import recipe_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reusable claim-weaving agentic workflow.")
    parser.add_argument("--use-case", required=True, choices=["support", "guardrail", "crm", "research", "coding"])
    parser.add_argument("--task", required=True, help="Path to a task JSON file.")
    parser.add_argument("--examples-root", default="examples", help="Directory containing use-case examples.")
    parser.add_argument("--planner", choices=["template", "ollama", "vllm"], default="template")
    parser.add_argument("--model-profile", choices=sorted(SMALL_REASONING_MODELS), default=None)
    parser.add_argument("--base-url", default=None, help="Override Ollama or vLLM base URL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_data = json.loads(Path(args.task).read_text(encoding="utf-8"))
    task = Task(**task_data)
    recipe = recipe_for(args.use_case)
    if args.planner != "template":
        profile_name = args.model_profile or f"{args.planner}-qwen3-4b"
        client = client_from_profile(profile_name, args.base_url)
        recipe = replace(recipe, planner_agent=ReasoningPlannerAgent(client=client, fallback=recipe.planner_agent))

    kb_path = Path(args.examples_root) / args.use_case / "kb"
    weaver = ClaimWeaver(recipe, TinyRetriever.from_markdown_dir(kb_path))
    outcome = weaver.run(task)
    print(json.dumps(asdict(outcome), indent=2, default=str))


if __name__ == "__main__":
    main()
