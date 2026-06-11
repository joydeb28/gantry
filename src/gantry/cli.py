from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from langgraph.types import Command
from .models import Task
from .scenarios import weaver_for

ALL_USE_CASES = [
    "coding",
    "crm",
    "finance",
    "fraud",
    "guardrail",
    "hr",
    "it",
    "legal",
    "research",
    "support",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentic Harness — run any use case with its native agentic pattern.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--use-case", required=True, choices=ALL_USE_CASES)
    parser.add_argument("--task", help="Path to a task JSON file. Required unless resuming.")
    parser.add_argument("--examples-root", default="examples", help="Directory containing use-case KBs.")
    parser.add_argument("--planner", choices=["template", "ollama", "vllm"], default="template")
    parser.add_argument("--model-profile", default=None, help="Model name (e.g. qwen3:4b) for LangChain planner.")
    parser.add_argument("--base-url", default=None, help="Override Ollama base URL.")
    parser.add_argument("--stream", action="store_true", help="Stream LangGraph node executions.")
    parser.add_argument("--resume", help="Resume an interrupted HITL workflow with this thread ID.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    weaver = weaver_for(
        args.use_case,
        kb_root=args.examples_root,
        planner_type=args.planner,
        model_name=args.model_profile,
        base_url=args.base_url,
    )

    if args.resume:
        config = {"configurable": {"thread_id": args.resume}}
        print(f"Resuming HITL workflow for thread {args.resume}...")
        result = weaver.graph.invoke(Command(resume={"approved": True}), config=config)
        outcome = result.get("outcome")
        if outcome:
            print(json.dumps(outcome.model_dump(), indent=2, default=str))
        else:
            print("Resume completed, but no outcome found.")
        return

    if not args.task:
        raise ValueError("--task is required unless resuming via --resume.")

    task_data = json.loads(Path(args.task).read_text(encoding="utf-8"))
    task = Task(**task_data)

    if args.stream:
        config = {"configurable": {"thread_id": task.id}}
        print(f"Streaming execution for task {task.id}...")
        initial_audit = [
            f"task:{task.id}:received",
            f"use_case:{task.use_case}",
            "pattern:streaming",
        ]
        for event in weaver.graph.stream(
            {"task": task, "audit_trail": initial_audit, "findings": []},
            config=config
        ):
            for node, state in event.items():
                print(f"\n--- Node Executed: {node} ---")
                for key in ("signal", "policy", "draft", "verification", "outcome"):
                    if key in state and state[key]:
                        val = state[key]
                        if hasattr(val, "model_dump"):
                            print(f"{key.capitalize()}: {json.dumps(val.model_dump(), indent=2, default=str)}")
                        else:
                            print(f"{key.capitalize()}: {val}")
        return

    outcome = weaver.run(task)
    print(json.dumps(outcome.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    main()
