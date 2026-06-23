# Gantry
<p align="center">
  <img src="assets/gantry-banner.png" alt="Gantry - Composable Agentic Patterns for Production AI" width="100%">
</p>

`gantry` is a Python toolkit for building **agentic AI systems** using established orchestration patterns. Every use case runs with the pattern best suited to its domain — from a simple sequential pipeline to a cyclic reflection critic loop or human-in-the-loop workflow.

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
| **Parallel Orchestrator** | `ParallelOrchestratorWeaver` | Parallel sub-agents + composite score aggregation | Fraud Detection |
| **Router / Dispatcher** | `RouterWeaver` | `add_conditional_edges` dynamic specialist routing | IT Helpdesk |
| **Reflection / Critic Loop** | `ReflectionWeaver` | Cyclic node edge feedback loops with tunable thresholds | Legal & Compliance |
| **Human-in-the-Loop** | `HumanInTheLoopWeaver` | Native `interrupt()` state pausing, resuming, and rejecting | HR & Onboarding |
| **Plan → Execute** | `PlanExecuteWeaver` | Planner → executor dynamic step execution loops | Finance & Accounting |

---

## Use Cases

| Key | Domain | Pattern | Example Task |
|---|---|---|---|
| `support` | Customer Support Automation | Pipeline | `examples/support/task_replacement.json` |
| `crm` | Sales CRM Management | Pipeline | `examples/crm/task_hot_lead.json` |
| `research` | Research Assistant | Pipeline | `examples/research/task_lit_scan.json` |
| `coding` | Coding Assistant | Pipeline | `examples/coding/task_bugfix.json` |
| `guardrail` | Guardrails | Orchestrator + Sub-agents | `examples/guardrail/task_pii.json` |
| `fraud` | Fraud Detection | Parallel Orchestrator (5 sub-agents) | `examples/fraud/task_card_fraud.json` |
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
│   │                      # Agent role Protocols: SignalAgentProtocol, PolicyAgentProtocol,
│   │                      # PlannerAgentProtocol, VerifierAgentProtocol
│   ├── generic_agents.py  # KeywordSignalAgent, SemanticSignalAgent, RulePolicyAgent,
│   │                      # TemplatePlannerAgent, BasicVerifierAgent, safe_node()
│   ├── retrieval.py       # BaseRetriever protocol, KnowledgeBaseRetriever (filesystem),
│   │                      # RemoteRetriever (in-memory, for external content sources)
│   ├── llm.py             # Multi-provider LangChain planner: ollama, openai, gemini,
│   │                      # anthropic, vllm — all via build_llm() factory.
│   │                      # Token budget guard + retry/backoff + async aplan()
│   ├── scenarios.py       # Registry/Builders mapping use cases to graph weavers
│   ├── cli.py             # CLI runner: --stream, --resume, --reject, --planner,
│   │                      # --trace, --checkpoint-backend, --reflection-threshold,
│   │                      # --reflection-turns
│   └── patterns/          # LangGraph StateGraph weavers
│       ├── pipeline.py            # PipelineWeaver
│       ├── orchestrator.py        # OrchestratorWeaver (Guardrails)
│       ├── parallel_orchestrator.py # ParallelOrchestratorWeaver (Fraud)
│       ├── router.py              # RouterWeaver
│       ├── reflection.py          # ReflectionWeaver + CriticAgent
│       ├── hitl.py                # HumanInTheLoopWeaver (SQLite + PostgreSQL checkpointer)
│       └── plan_execute.py        # PlanExecuteWeaver
├── examples/              # Knowledge bases (*.md) and task scenarios (*.json)
├── tests/                 # Unit + integration tests (53 tests, all passing)
└── pyproject.toml         # Dependencies and optional-dep groups per LLM provider
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

### Human-in-the-Loop Workflow (Pause, Resume, and Reject)
When running HITL workflows (like `hr`), execution will pause at the approval gate and return a `pending_approval` status:
```bash
# Run 1: Execution pauses at gate
gantry --use-case hr --task examples/hr/task_onboarding.json
```
This saves state checkpoints to SQLite. To approve and resume the workflow:
```bash
# Run 2a: Approve — resume from the pause point
gantry --use-case hr --resume HR-1001
```
To reject the pending request (routes to `finalize_rejected`):
```bash
# Run 2b: Reject — deny the pending approval request
gantry --use-case hr --reject HR-1001
```

### LLM-powered Planning
Swap the rule-based planner for any supported LLM provider:
```bash
# Local Ollama model (no API key required)
gantry --use-case support --task examples/support/task_replacement.json \
  --planner ollama --model-profile qwen3:4b

# OpenAI (requires OPENAI_API_KEY + pip install "gantry[openai]")
gantry --use-case support --task examples/support/task_replacement.json \
  --planner openai --model-profile gpt-4o-mini

# Gemini (requires GOOGLE_API_KEY + pip install "gantry[gemini]")
gantry --use-case fraud --task examples/fraud/task_card_fraud.json \
  --planner gemini --model-profile gemini-2.0-flash

# Anthropic (requires ANTHROPIC_API_KEY + pip install "gantry[anthrop]")
gantry --use-case legal --task examples/legal/task_contract_review.json \
  --planner anthropic
```

### Tuning the Reflection / Critic Loop
The Legal & Compliance use case runs a draft → critic feedback loop. You can tune it at runtime:
```bash
# Stricter threshold (default 0.7) — critic must score 0.9+ to accept the draft
gantry --use-case legal --task examples/legal/task_contract_review.json \
  --reflection-threshold 0.9

# Fewer revision turns (default 3) for faster turnaround
gantry --use-case legal --task examples/legal/task_contract_review.json \
  --reflection-turns 1

# Combine: aggressive single-pass with high bar
gantry --use-case legal --task examples/legal/task_contract_review.json \
  --planner openai --reflection-threshold 0.85 --reflection-turns 2
```

### Observability — LangSmith Tracing
```bash
export LANGCHAIN_API_KEY=lsv2_...
gantry --use-case fraud --task examples/fraud/task_card_fraud.json --trace
# Sets LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT=gantry automatically
```

### PostgreSQL Checkpointer (Production HITL)
```bash
export DATABASE_URL=postgresql://user:pass@localhost/gantry
gantry --use-case hr --task examples/hr/task_onboarding.json \
  --checkpoint-backend postgres
```

---

## Installing LLM Provider Extras

```bash
pip install "gantry[openai]"         # OpenAI (langchain-openai)
pip install "gantry[gemini]"         # Gemini (langchain-google-genai)
pip install "gantry[anthrop]"        # Anthropic (langchain-anthropic)
pip install "gantry[all-providers]"  # All three cloud providers
pip install "gantry[prod]"           # HITL persistence: SQLite + PostgreSQL
```

---

## Running Tests

```bash
# Run all 53 unit and integration tests
.venv/bin/pytest -v
```

---

## Design Principles

- **Framework Orchestration**: LangGraph `StateGraph` for clean state transitions; every node is wrapped with `safe_node()` for graceful error recovery.
- **Type Safety**: Pydantic v2 enforces schemas and handles JSON serialization across all models. `AgentRecipe` validates all four agent fields at construction time against typed `Protocol` classes — misconfiguration fails early.
- **Reliable LLM Calls**: `LangChainPlanner` applies a per-provider token budget to evidence, and retries up to 3 times with exponential backoff on transient failures. An async `aplan()` / `arun()` path is also available.
- **Efficient State**: All pattern weavers use `Annotated[list[str], operator.add]` for `audit_trail` — nodes return only their new entries and LangGraph merges them, eliminating O(n) copy-and-extend on every node.
- **Multi-Provider LLM**: `build_llm()` factory supports Ollama, OpenAI, Gemini, Anthropic, and vLLM — swap with a single flag.
- **Semantic Retrieval**: `KnowledgeBaseRetriever` (filesystem) and `RemoteRetriever` (in-memory) both satisfy the `BaseRetriever` protocol; plug in Confluence, Notion, or any content source. SHA-256 fingerprinting detects stale KB caches automatically.
- **Semantic Intent Detection**: `SemanticSignalAgent` uses `bge-small-en-v1.5` embeddings to handle natural-language paraphrasing — no keyword lists needed.
- **Extensible Policy**: `RulePolicyAgent` accepts `custom_conditions` so domain-specific blocking logic can be injected without subclassing.
- **Tunable Reflection**: `ReflectionWeaver` `approval_threshold` and `max_turns` are now CLI-configurable via `--reflection-threshold` and `--reflection-turns`.
- **Full HITL Lifecycle**: `--resume` approves a pending HITL workflow; `--reject` denies it — both via the same interrupt/resume mechanism.
- **Observability**: `--trace` enables zero-code LangSmith tracing; structured audit logs are written into every `Outcome`.
- **Production Persistence**: HITL workflows support SQLite (dev) and PostgreSQL (production) checkpointers via `--checkpoint-backend`.

---

## License

MIT
