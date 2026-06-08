from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .models import Task
from .scenarios import weaver_for


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reusable claim-weaving agentic workflow.")
    parser.add_argument("--use-case", required=True, choices=["support", "guardrail", "crm", "research", "coding"])
    parser.add_argument("--task", required=True, help="Path to a task JSON file.")
    parser.add_argument("--examples-root", default="examples", help="Directory containing use-case examples.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task_data = json.loads(Path(args.task).read_text(encoding="utf-8"))
    task = Task(**task_data)
    weaver = weaver_for(args.use_case, args.examples_root)
    outcome = weaver.run(task)
    print(json.dumps(asdict(outcome), indent=2, default=str))


if __name__ == "__main__":
    main()
