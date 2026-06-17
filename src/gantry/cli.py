"""Gantry CLI — run any use case with its native agentic pattern.

Usage examples::

    # Run a use case
    gantry --use-case support --task examples/support/task_replacement.json

    # Stream node-by-node execution
    gantry --use-case support --task examples/support/task_replacement.json --stream

    # Use an OpenAI planner instead of the default template planner
    gantry --use-case support --task ... --planner openai --model-profile gpt-4o-mini

    # Enable LangSmith observability tracing
    gantry --use-case fraud --task ... --trace

    # Resume an interrupted HITL workflow
    gantry --use-case hr --resume HR-1001

    # Use a PostgreSQL checkpoint backend for HITL
    gantry --use-case hr --task ... --checkpoint-backend postgres
    # (reads DATABASE_URL from env, e.g. postgresql://user:pass@host/db)
"""

from __future__ import annotations

import argparse
import json
import os
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

ALL_PLANNERS = ["template", "ollama", "vllm", "openai", "gemini", "anthropic"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentic Harness — run any use case with its native agentic pattern.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--use-case", required=True, choices=ALL_USE_CASES)
    parser.add_argument("--task", help="Path to a task JSON file. Required unless resuming.")
    parser.add_argument("--examples-root", default="examples", help="Directory containing use-case KBs.")
    parser.add_argument(
        "--planner",
        choices=ALL_PLANNERS,
        default="template",
        help=(
            "Planning engine. 'template' (default) uses rule-based templates with no LLM. "
            "'ollama'/'vllm' call a local model server. "
            "'openai'/'gemini'/'anthropic' call hosted APIs "
            "(requires the matching package and API key env var)."
        ),
    )
    parser.add_argument("--model-profile", default=None, help="Model name (e.g. gpt-4o-mini, gemini-2.0-flash).")
    parser.add_argument("--base-url", default=None, help="Override the LLM provider API base URL.")
    parser.add_argument("--stream", action="store_true", help="Stream LangGraph node executions.")
    parser.add_argument("--resume", help="Resume an interrupted HITL workflow with this thread ID.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Enable LangSmith observability tracing. "
            "Requires LANGCHAIN_API_KEY env var. "
            "Sets LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT=gantry."
        ),
    )
    parser.add_argument(
        "--checkpoint-backend",
        choices=["sqlite", "postgres"],
        default="sqlite",
        help=(
            "Checkpoint backend for HITL workflows. "
            "'sqlite' (default) stores checkpoints in .gantry_checkpoints.db. "
            "'postgres' reads DATABASE_URL env var "
            "(e.g. postgresql://user:pass@localhost/gantry)."
        ),
    )
    return parser


def _enable_tracing(project: str = "gantry") -> None:
    """Set LangSmith env vars to enable observability tracing."""
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        print(
            "[gantry] WARNING: --trace requires LANGCHAIN_API_KEY to be set. "
            "Tracing may not work."
        )
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project
    print(f"[gantry] LangSmith tracing enabled (project='{project}').")


def main() -> None:
    args = build_parser().parse_args()

    if args.trace:
        _enable_tracing()

    weaver = weaver_for(
        args.use_case,
        kb_root=args.examples_root,
        planner_type=args.planner,
        model_name=args.model_profile,
        base_url=args.base_url,
        checkpoint_backend=args.checkpoint_backend,
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
