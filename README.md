# Agentic Support Weaver

Agentic Support Weaver is a small, offline-first reference implementation for customer support ticket resolution.

The novel pattern is **claim weaving**: each agent emits a tiny typed claim, then the workflow weaves those claims only when evidence, policy, and verification agree. This keeps the system simple enough to inspect while still looking like a real agentic workflow.

## Why This Exists

Most agent demos jump straight to a giant prompt. This repo shows a pattern teams can adopt incrementally:

- Triage agent classifies intent, urgency, and missing fields.
- Retriever agent finds support policy evidence from local markdown.
- Policy agent applies deterministic business rules.
- Resolution agent drafts an action and customer reply.
- Verifier agent blocks unsupported or policy-unsafe automation.
- Fallback path escalates instead of guessing.

The default implementation uses only the Python standard library. You can later swap individual agents for local LLM calls through Ollama, llama.cpp servers, vLLM, or any OpenAI-compatible local gateway.

## Quick Start

```bash
cd agentic-support-weaver
python -m venv .venv
source .venv/bin/activate
pip install -e .
support-weaver --ticket examples/ticket_damaged.json --kb examples/kb
```

Expected final action for the damaged-item example: `replace`.

Run tests with only the standard library:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Design Pattern: Claim Weaving

Instead of letting one agent decide everything, the workflow produces auditable intermediate objects:

```text
Ticket
  -> TriageClaim
  -> Evidence[]
  -> PolicyClaim
  -> ActionPlan
  -> Verification
  -> Resolution
```

This is useful for support automation because the hardest problem is not generating a nice reply. The hard problem is knowing when automation is allowed, where the answer came from, and when to escalate.

## Example Output

```json
{
  "ticket_id": "TCK-1001",
  "final_action": "replace",
  "customer_reply": "I am sorry the item arrived damaged. I can arrange a replacement right away under our 30-day damage policy."
}
```

## Extension Ideas

- Replace `TinyRetriever` with Chroma, LanceDB, Qdrant, or SQLite FTS.
- Add an Ollama-backed `ResolutionAgent` while keeping deterministic policy and verification.
- Emit `Resolution` objects to Zendesk, Freshdesk, GitHub Issues, or Slack.
- Add a human approval queue for verifier failures.
- Store audit trails as JSONL for later evaluation.

## Project Layout

```text
src/support_weaver/
  agents.py      # triage, resolution, verifier agents
  policy.py      # deterministic business policy
  retrieval.py   # dependency-free markdown retrieval
  workflow.py    # claim-weaving orchestration
  cli.py         # command-line interface
examples/
  kb/            # sample support knowledge base
  ticket_*.json  # sample tickets
tests/
```

## Free/Open-Source Stack

- Python standard library for the runnable baseline.
- Markdown files for a local knowledge base.
- Optional local LLM providers: Ollama, llama.cpp, vLLM, LocalAI.
- Optional vector stores: Chroma, Qdrant, LanceDB, SQLite FTS.

## License

MIT
