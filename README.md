# Gantry

> **Six production-grade agentic patterns. Nine real-world use cases. Orchestrated with LangGraph, LangChain, LlamaIndex, and Pydantic v2.**

`gantry` is a production-ready Python toolkit for building **agentic AI systems** using established orchestration patterns. Every use case runs with the pattern best suited to its domain — from a simple sequential pipeline to a cyclic reflection critic loop or human-in-the-loop workflow.

---

## Quick Start

```bash
git clone https://github.com/joydeb28/gantry.git
cd gantry

# Create virtual environment and install dependencies
python3 -m pip install uv
python3 -m uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[prod]"

# Run any use case
gantry --use-case support --task examples/support/task_replacement.json
```

---

## Agentic Patterns

All patterns are implemented as LangGraph `StateGraph` workflows compiled into weavers, maintaining the same clean `run(task) -> Outcome` interface.

| Pattern | LangGraph Implementation | Key Features | Use Cases |
|---|---|---|---|
| **Pipeline** | `PipelineWeaver` | Sequential node chain | Customer Support, Sales CRM, Research, Coding |
| **Orchestrator + Sub-agents** | `OrchestratorWeaver` | `Send` API parallel fan-out & aggregation | Guardrails |
| **Fraud Orchestrator** | `FraudOrchestratorWeaver` | 5 parallel sub-agents + composite scoring | Fraud Detection |
| **Router / Dispatcher** | `RouterWeaver` | `add_conditional_edges` dynamic specialist routing | IT Helpdesk |
| **Reflection / Critic Loop** | `ReflectionWeaver` | Cyclic node edge feedback loops | Legal & Compliance |
| **Human-in-the-Loop** | `HumanInTheLoopWeaver` | Native `interrupt()` state pausing and resuming | HR & Onboarding |
| **Plan → Execute** | `PlanExecuteWeaver` | Planner -> executor dynamic step execution loops | Finance & Accounting |

---

## Use Cases

| Key | Domain | Pattern | Example Task |
|---|---|---|---|
| `support` | Customer Support Automation | Pipeline | `examples/support/task_replacement.json` |
| `crm` | Sales CRM Management | Pipeline | `examples/crm/task_hot_lead.json` |
| `research` | Research Assistant | Pipeline | `examples/research/task_lit_scan.json` |
| `coding` | Coding Assistant | Pipeline | `examples/coding/task_bugfix.json` |
| `guardrail` | Guardrails | Orchestrator + Sub-agents | `examples/guardrail/task_pii.json` |
| `fraud` | Fraud Detection | Fraud Orchestrator (5 sub-agents) | `examples/fraud/task_card_fraud.json` |
| `it` | IT Helpdesk | Router | `examples/it/task_access_request.json` |
| `legal` | Legal & Compliance | Reflection / Critic Loop | `examples/legal/task_contract_review.json` |
| `hr` | HR & Onboarding | Human-in-the-Loop | `examples/hr/task_onboarding.json` |
| `finance` | Finance & Accounting | Plan → Execute | `examples/finance/task_invoice.json` |

---

## Project Structure

```
gantry/
├── src/gantry/
│   ├── models.py          # Pydantic v2 data models: Task, Signal, Evidence, Outcome, etc.
│   ├── generic_agents.py  # Agent components: KeywordSignalAgent, RulePolicyAgent, etc.
│   ├── retrieval.py       # KnowledgeBaseRetriever — Vector RAG backed by LlamaIndex + HuggingFace
│   ├── llm.py             # LangChain planners (ChatOllama + structured output)
│   ├── scenarios.py       # Registry/Builders mapping use cases to graph weavers
│   ├── cli.py             # CLI runner supporting --stream, --resume, and --planner
│   └── patterns/          # LangGraph StateGraph weavers
│       ├── pipeline.py        # PipelineWeaver
│       ├── orchestrator.py    # OrchestratorWeaver
│       ├── fraud.py           # FraudOrchestratorWeaver + 5 specialist sub-agents
│       ├── router.py          # RouterWeaver
│       ├── reflection.py      # ReflectionWeaver + CriticAgent
│       ├── hitl.py            # HumanInTheLoopWeaver (with SqliteSaver checkpointer)
│       └── plan_execute.py    # PlanExecuteWeaver
├── examples/              # Knowledge bases (*.md) and task scenarios (*.json)
├── tests/                 # Unit tests (test_claim_weaver.py, test_hitl_langgraph.py)
└── pyproject.toml         # Declared dependencies and build metadata
```

---

## Command Line Usage

### Running Scenarios
```bash
# Run a Customer Support Pipeline
gantry --use-case support --task examples/support/task_replacement.json

# Stream intermediate node execution states
gantry --use-case support --task examples/support/task_replacement.json --stream
```

### Human-in-the-Loop Workflow (Pause and Resume)
When running HITL workflows (like `hr`), execution will pause at the approval gate and return a `pending_approval` status:
```bash
# Run 1: Execution pauses at gate
gantry --use-case hr --task examples/hr/task_onboarding.json
```
This saves state checkpoints to SQLite. To resume the workflow:
```bash
# Run 2: Resume using the thread/task ID
gantry --use-case hr --resume HR-1001
```

### LLM-powered Planning (LangChain)
To swap the rule-based planner for a local Ollama reasoning model:
```bash
# Make sure Ollama is running with qwen3:4b
gantry --use-case support --task examples/support/task_replacement.json --planner ollama --model-profile qwen3:4b
```

---

## Running Tests

```bash
# Run unit tests and LangGraph integration tests
.venv/bin/pytest -v
```

---

## Design Principles

- **Framework Orchestration**: Leveraging LangGraph for clean state transition graphs and state management.
- **Type Safety**: Pydantic v2 enforces schemas and handles JSON serialization.
- **Semantic Retrieval**: LlamaIndex and local HuggingFace embeddings provide vector RAG out of the box.
- **State Audits**: Workflows compile audit logs throughout execution.

---

## License

MIT
