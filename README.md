# Agentic Pattern Weaver

Agentic Pattern Weaver is a small, offline-first reference implementation for building agentic AI workflows across many use cases.

The core pattern is **claim weaving**: each agent emits a small typed claim, then the workflow composes those claims only when evidence, policy, and verification agree. It is simple enough to inspect, but structured enough to adapt to production domains.

## Use Cases Included

- Guardrails
- Customer support automation
- Sales CRM management
- Research assistants
- Coding assistants

All examples use the same core workflow:

```text
Task
  -> Signal
  -> Evidence[]
  -> PolicyDecision
  -> Plan
  -> Verification
  -> Outcome
```

## Why This Is Different

Most agent demos start with one large prompt. This repo starts with an adoptable design pattern:

- **Signal agent** classifies intent, risk, missing fields, and tags.
- **Retriever** finds local evidence from markdown knowledge bases.
- **Policy agent** applies deterministic constraints before action.
- **Planner agent** proposes the next action and response.
- **Verifier agent** blocks unsupported, low-confidence, or policy-unsafe plans.
- **Fallback path** escalates instead of guessing.

The runnable baseline uses only the Python standard library. You can swap any agent for Ollama, llama.cpp, vLLM, LocalAI, or another free/open-source local model gateway.

## Quick Start

```bash
cd agentic-pattern-weaver
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run an example:

```bash
agent-weaver --use-case support --task examples/support/task_replacement.json
```

Without installing:

```bash
PYTHONPATH=src python3 -m agent_weaver.cli --use-case guardrail --task examples/guardrail/task_pii.json
```

## Example Commands

```bash
PYTHONPATH=src python3 -m agent_weaver.cli --use-case support --task examples/support/task_replacement.json
PYTHONPATH=src python3 -m agent_weaver.cli --use-case guardrail --task examples/guardrail/task_pii.json
PYTHONPATH=src python3 -m agent_weaver.cli --use-case crm --task examples/crm/task_hot_lead.json
PYTHONPATH=src python3 -m agent_weaver.cli --use-case research --task examples/research/task_lit_scan.json
PYTHONPATH=src python3 -m agent_weaver.cli --use-case coding --task examples/coding/task_bugfix.json
```

Expected final actions:

```text
support   -> replace
guardrail -> redact
crm       -> schedule_demo
research  -> build_reading_list
coding    -> write_patch
```

## Optional Small Reasoning Model

The default workflow does not require a model. For local reasoning, the recommended small profile is **Qwen3 4B**:

- Ollama: `qwen3:4b`
- vLLM: `Qwen/Qwen3-4B`
- Smaller fallback: `qwen3:1.7b` on Ollama or `Qwen/Qwen3-0.6B` on vLLM

Run with Ollama:

```bash
ollama pull qwen3:4b
PYTHONPATH=src python3 -m agent_weaver.cli \
  --use-case coding \
  --task examples/coding/task_bugfix.json \
  --planner ollama \
  --model-profile ollama-qwen3-4b
```

Run with vLLM:

```bash
vllm serve Qwen/Qwen3-4B --reasoning-parser qwen3
PYTHONPATH=src python3 -m agent_weaver.cli \
  --use-case research \
  --task examples/research/task_lit_scan.json \
  --planner vllm \
  --model-profile vllm-qwen3-4b
```

For very small machines:

```bash
ollama pull qwen3:1.7b
PYTHONPATH=src python3 -m agent_weaver.cli \
  --use-case guardrail \
  --task examples/guardrail/task_pii.json \
  --planner ollama \
  --model-profile ollama-qwen3-1.7b
```

The model planner can propose an action, but `PolicyDecision` and `Verification` still constrain the final result.

## Project Layout

```text
src/agent_weaver/
  models.py          # generic Task, Signal, PolicyDecision, Plan, Outcome
  pattern.py         # claim-weaving orchestration
  generic_agents.py  # reusable keyword, policy, planner, verifier agents
  scenarios.py       # recipes for support, guardrail, CRM, research, coding
  retrieval.py       # dependency-free markdown retrieval
  llm.py             # optional stdlib Ollama/vLLM reasoning clients
  cli.py             # command-line interface
examples/
  support/
  guardrail/
  crm/
  research/
  coding/
tests/
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Free/Open-Source Stack

- Python standard library for the default runnable implementation.
- Markdown files for local knowledge bases.
- Optional local LLM providers: Ollama, vLLM, llama.cpp, LocalAI.
- Optional retrieval stores: SQLite FTS, Chroma, Qdrant, LanceDB.

## Extension Ideas

- Add more model-backed planners while keeping deterministic policy and verification.
- Store `Outcome` audit trails as JSONL for evaluation.
- Connect actions to Zendesk, Salesforce, HubSpot, GitHub Issues, Jira, Slack, or email.
- Add a human approval queue for verifier failures.
- Replace keyword signals with embeddings or a small local classifier.

## License

MIT
