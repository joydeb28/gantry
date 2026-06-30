# Gantry
<p align="center">
  <img src="assets/gantry-banner.png" alt="Gantry - Composable Agentic Patterns for Production AI" width="100%">
</p>

`gantry` is a Python toolkit for building **agentic AI systems** using established orchestration patterns. Every use case runs with the pattern best suited to its domain — from a simple sequential pipeline to a cyclic reflection critic loop or human-in-the-loop workflow.

> **v0.4.0** — Introduces the Production Harness: pluggable metrics emission, session-aware conversation memory, autonomous triggers (cron, polling, webhook), and dynamic step planning.

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
│   │                      # TemplatePlannerAgent, BasicVerifierAgent, ClarificationAgent,
│   │                      # safe_node()
│   ├── retrieval.py       # BaseRetriever protocol, KnowledgeBaseRetriever (filesystem),
│   │                      # RemoteRetriever (in-memory), KBWatcher (hot-reload)
│   ├── llm.py             # Multi-provider LangChain planner: ollama, openai, gemini,
│   │                      # anthropic, vllm — all via build_llm() factory.
│   │                      # Token budget guard + retry/backoff + async aplan()
│   │                      # DynamicStepPlanner — LLM-powered step generation
│   ├── metrics.py         # MetricsEmitter protocol, NoOpEmitter, LoggingEmitter,
│   │                      # emit_outcome() helper (NEW in v0.4.0)
│   ├── memory.py          # ConversationBuffer, InMemoryBackend, Turn (NEW in v0.4.0)
│   ├── orchestration.py   # RetryOrchestrator — strategy escalation wrapper (NEW in v0.4.0)
│   ├── triggers.py        # ScheduledTrigger (cron), PollingTrigger, WebhookTrigger,
│   │                      # TriggerRunner — autonomous execution (NEW in v0.4.0)
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
│       └── plan_execute.py        # PlanExecuteWeaver + DynamicStepPlanner support
├── examples/              # Knowledge bases (*.md) and task scenarios (*.json)
├── tests/                 # Unit + integration tests
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

## Production Harness — v0.4.0

### Metrics Emission

Every pattern weaver's `run()` method now emits three standard metrics after each execution:

```python
from gantry.metrics import LoggingEmitter
from gantry.scenarios import weaver_for

# Dev: log metrics to Python logger
weaver = weaver_for("support", metrics=LoggingEmitter())

# Production: plug in Prometheus or Datadog
class PrometheusEmitter:
    def emit(self, event: str, labels: dict[str, str], value: float) -> None:
        self._counter.labels(**labels).inc(value)

weaver = weaver_for("support", metrics=PrometheusEmitter(...))
outcome = weaver.run(task)
# Emits: gantry.outcome.count, gantry.outcome.action, gantry.verification
```

Any object with `emit(event: str, labels: dict[str, str], value: float) -> None` satisfies the `MetricsEmitter` protocol — no inheritance required. Default is `NoOpEmitter` (zero overhead).

### Session Memory

Conversation history can be maintained across calls using `ConversationBuffer`:

```python
from gantry.memory import ConversationBuffer
from gantry.scenarios import weaver_for

buf = ConversationBuffer(session_id="user-42", max_turns=10)
weaver = weaver_for("support", memory=buf)

outcome1 = weaver.run(task1)   # context: empty
outcome2 = weaver.run(task2)   # context: previous turn injected into plan prompt
```

Backed by `InMemoryBackend` by default. Swap the backend for Redis or any persistent store by implementing the `MemoryBackend` protocol (`load(session_id) / save(session_id, turns)`).

### Autonomous Triggers

Run weavers without human input using three trigger types:

```python
from gantry.triggers import ScheduledTrigger, PollingTrigger, WebhookTrigger, TriggerRunner
import asyncio

# Cron-based: run every weekday at 08:00 (requires apscheduler, included in [prod])
schedule = ScheduledTrigger(
    cron="0 8 * * 1-5",
    task_factory=lambda: Task(id="daily-1", use_case="finance", ...),
    weaver=weaver_for("finance"),
)

# Interval polling: check a data source every 60 seconds
poller = PollingTrigger(
    interval_seconds=60,
    source=lambda: db.fetch_pending_invoices(),
    condition=lambda inv: inv["amount_usd"] > 10_000,
    task_factory=lambda inv: Task(id=inv["id"], use_case="finance", ...),
    weaver=weaver_for("finance"),
)

# Webhook: receive HTTP POST requests on a route
hook = WebhookTrigger(
    route="/hooks/zendesk",
    task_factory=lambda body: Task(id=body["ticket_id"], use_case="support", ...),
    weaver=weaver_for("support"),
    port=9000,
)

# Manage all triggers in one event loop with graceful SIGTERM/SIGINT shutdown
asyncio.run(TriggerRunner([schedule, poller, hook]).run())
```

### Strategy Retry (RetryOrchestrator)

Escalate across multiple weaver strategies when verification fails:

```python
from gantry.orchestration import RetryOrchestrator
from gantry.scenarios import weaver_for

orchestrator = RetryOrchestrator(
    strategies=[
        lambda: weaver_for("support"),                        # fast template planner
        lambda: weaver_for("support", planner="openai"),      # LLM for harder cases
    ]
)
outcome = orchestrator.run(task)  # same run(task) → Outcome interface
```

### Dynamic Step Planning

Replace the static `step_map` in `PlanExecuteWeaver` with an LLM-powered planner:

```python
from gantry.llm import DynamicStepPlanner
from gantry.scenarios import weaver_for

weaver = weaver_for("finance", dynamic_step_planner=DynamicStepPlanner(provider="openai"))
# Steps are now generated per-task by the LLM instead of looked up in step_map
```

### Installing Production Extras

```bash
pip install "gantry[prod]"           # HITL persistence + ScheduledTrigger (APScheduler)
pip install "gantry[openai]"         # OpenAI (langchain-openai)
pip install "gantry[gemini]"         # Gemini (langchain-google-genai)
pip install "gantry[anthrop]"        # Anthropic (langchain-anthropic)
pip install "gantry[all-providers]"  # All three cloud providers
```

---

## Running Tests

```bash
# Run all unit and integration tests
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
- **Live KB Hot-Reload**: `KBWatcher` rebuilds the vector index in a background thread whenever KB files change — zero-downtime knowledge base updates in long-running services.
- **Semantic Intent Detection**: `SemanticSignalAgent` uses `bge-small-en-v1.5` embeddings to handle natural-language paraphrasing — no keyword lists needed.
- **Extensible Policy**: `RulePolicyAgent` accepts `custom_conditions` so domain-specific blocking logic can be injected without subclassing.
- **Tunable Reflection**: `ReflectionWeaver` `approval_threshold` and `max_turns` are now CLI-configurable via `--reflection-threshold` and `--reflection-turns`.
- **Full HITL Lifecycle**: `--resume` approves a pending HITL workflow; `--reject` denies it — both via the same interrupt/resume mechanism.
- **Observability**: `--trace` enables zero-code LangSmith tracing; structured audit logs are written into every `Outcome`. Every `run()` call emits three standard metrics via the pluggable `MetricsEmitter` (v0.4.0).
- **Production Persistence**: HITL workflows support SQLite (dev) and PostgreSQL (production) checkpointers via `--checkpoint-backend`.
- **Session Memory**: `ConversationBuffer` carries multi-turn context across calls with pluggable backends (v0.4.0).
- **Autonomous Execution**: `ScheduledTrigger`, `PollingTrigger`, and `WebhookTrigger` run weavers proactively from external events — all managed by `TriggerRunner` with graceful shutdown (v0.4.0).
- **Strategy Resilience**: `RetryOrchestrator` escalates across weaver strategies when verification fails — same `run(task) → Outcome` interface (v0.4.0).
- **Dynamic Planning**: `DynamicStepPlanner` replaces static `step_map` lookups in `PlanExecuteWeaver` with LLM-generated step sequences (v0.4.0).

---

## License

MIT
