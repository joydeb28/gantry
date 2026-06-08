from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .models import Ticket
from .retrieval import TinyRetriever
from .workflow import SupportWeaver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve a support ticket with an offline-first agentic workflow.")
    parser.add_argument("--ticket", required=True, help="Path to a ticket JSON file.")
    parser.add_argument("--kb", default="examples/kb", help="Path to a markdown knowledge base directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ticket_data = json.loads(Path(args.ticket).read_text(encoding="utf-8"))
    ticket = Ticket(**ticket_data)
    weaver = SupportWeaver(TinyRetriever.from_markdown_dir(args.kb))
    resolution = weaver.resolve(ticket)
    print(json.dumps(asdict(resolution), indent=2, default=str))


if __name__ == "__main__":
    main()
