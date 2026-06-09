# Agentic Harness

> **Six production-grade agentic patterns. Nine real-world use cases. Zero external dependencies.**

`agentic-harness` is a batteries-included Python toolkit for building **agentic AI systems** using established orchestration patterns. Every use case runs with the pattern best suited to its domain — from a simple sequential pipeline to a full orchestrator + sub-agents fan-out.

The goal is simple: **anyone should be able to clone this repo and have a working, extensible agentic AI system in minutes.**

---

## Quick Start

```bash
git clone https://github.com/youruser/agentic-harness.git
cd agentic-harness

# Run any use case — no installs, no API keys
PYTHONPATH=src python3 -m agentic_harness.cli \
  --use-case guardrail \
  --task examples/guardrail/task_pii.json
```

---

## Agentic Patterns

Six patterns are implemented, each as a standalone `Weaver` class with a single `run(task) -> Outcome` interface.

| Pattern | Weaver Class | Flow | Use Cases |
|---|---|---|---|
| **Pipeline** | `PipelineWeaver` | Task → Signal → Evidence → Policy → Plan → Verify → Outcome | Customer Support, Sales CRM, Research, Coding |
| **Orchestrator + Sub-agents** | `OrchestratorWeaver` | Orchestrator fans out to N specialist sub-agents → aggregates → Pipeline | Guardrails |
| **Router / Dispatcher** | `RouterWeaver` | Classify intent → dispatch to specialist agent → Pipeline | IT Helpdesk |
| **Reflection / Critic Loop** | `ReflectionWeaver` | Draft → Critic scores → revise (up to N turns) → Verify | Legal & Compliance |
| **Human-in-the-Loop** | `HumanInTheLoopWeaver` | Pipeline → checkpoint (pending) → resume on approval | HR & Onboarding |
| **Plan → Execute** | `PlanExecuteWeaver` | Pipeline → multi-step execution (each step verified) | Finance & Accounting |

---

## Use Cases

| Key | Domain | Pattern | Example Task |
|---|---|---|---|
| `support` | Customer Support Automation | Pipeline | `examples/support/task_replacement.json` |
| `crm` | Sales CRM Management | Pipeline | `examples/crm/task_hot_lead.json` |
| `research` | Research Assistant | Pipeline | `examples/research/task_lit_scan.json` |
| `coding` | Coding Assistant | Pipeline | `examples/coding/task_bugfix.json` |
| `guardrail` | Guardrails | Orchestrator + Sub-agents | `examples/guardrail/task_pii.json` |
| `it` | IT Helpdesk | Router | `examples/it/task_access_request.json` |
| `legal` | Legal & Compliance | Reflection / Critic Loop | `examples/legal/task_contract_review.json` |
| `hr` | HR & Onboarding | Human-in-the-Loop | `examples/hr/task_onboarding.json` |
| `finance` | Finance & Accounting | Plan → Execute | `examples/finance/task_invoice.json` |

---

## Project Structure

```
agentic-harness/
├── src/agentic_harness/
│   ├── models.py          # Task, Signal, Evidence, PolicyDecision, Plan, Verification, Outcome
│   ├── generic_agents.py  # KeywordSignalAgent, RulePolicyAgent, TemplatePlannerAgent, BasicVerifierAgent
│   ├── retrieval.py       # BM25 TinyRetriever — load KB from .md files, no vector DB needed
│   ├── pattern.py         # ClaimWeaver — the Pipeline pattern (base for all others)
│   ├── llm.py             # Optional: Ollama / vLLM reasoning client (Qwen3 etc.)
│   ├── scenarios.py       # weaver_for(use_case) — the pattern registry
│   ├── cli.py             # Entry point: python -m agentic_harness.cli
│   └── patterns/          # Pattern library
│       ├── pipeline.py        # PipelineWeaver (re-export of ClaimWeaver)
│       ├── orchestrator.py    # OrchestratorWeaver + sub-agents (PII, Safety, ExternalSend, Tone)
│       ├── router.py          # RouterWeaver + SpecialistAgent
│       ├── reflection.py      # ReflectionWeaver + CriticAgent
│       ├── hitl.py            # HumanInTheLoopWeaver
│       └── plan_execute.py    # PlanExecuteWeaver + ExecutionStep
├── examples/
│   ├── support/   crm/   research/   coding/    # Pipeline use cases
│   ├── guardrail/                               # Orchestrator use case
│   ├── it/                                      # Router use case
│   ├── legal/                                   # Reflection use case
│   ├── hr/                                      # HITL use case
│   └── finance/                                 # Plan-Execute use case
│       └── each has: kb/*.md + task_*.json
├── tests/
│   ├── test_claim_weaver.py     # Unit tests for pipeline use cases
│   └── test_reasoning_models.py # Optional: reasoning model integration tests
└── pyproject.toml
```

---

## Running Examples

```bash
# Customer Support — Pipeline pattern
PYTHONPATH=src python3 -m agentic_harness.cli --use-case support --task examples/support/task_replacement.json

# Guardrails — Orchestrator pattern (PII + jailbreak sub-agents)
PYTHONPATH=src python3 -m agentic_harness.cli --use-case guardrail --task examples/guardrail/task_pii.json
PYTHONPATH=src python3 -m agentic_harness.cli --use-case guardrail --task examples/guardrail/task_jailbreak.json

# IT Helpdesk — Router pattern (classifies + dispatches to specialist)
PYTHONPATH=src python3 -m agentic_harness.cli --use-case it --task examples/it/task_access_request.json

# Legal — Reflection / Critic Loop (draft iterates until critic approves)
PYTHONPATH=src python3 -m agentic_harness.cli --use-case legal --task examples/legal/task_contract_review.json

# HR — Human-in-the-Loop (first run pauses for approval)
PYTHONPATH=src python3 -m agentic_harness.cli --use-case hr --task examples/hr/task_onboarding.json
# Second run (approved)
PYTHONPATH=src python3 -m agentic_harness.cli --use-case hr --task examples/hr/task_onboarding_approved.json

# Finance — Plan → Execute (multi-step: validate → check_budget → approve_payment)
PYTHONPATH=src python3 -m agentic_harness.cli --use-case finance --task examples/finance/task_invoice.json
```

---

## Writing a Task

All tasks are plain JSON files matching the `Task` model:

```json
{
  "id": "TASK-001",
  "use_case": "support",
  "title": "Replacement request",
  "body": "My laptop screen cracked. It is only 2 weeks old.",
  "metadata": {
    "value_usd": 120,
    "days_since_event": 14
  }
}
```

`metadata` is used by policy rules (e.g., `value_usd > 250` blocks auto-approval, `human_approved: true` resumes an HITL workflow).

---

## Adding a New Use Case

### Pipeline pattern (simplest)

1. Add a `my_use_case_recipe()` function to `scenarios.py`
2. Register it in the `weaver_for()` builders dict and `recipe_for()` dict
3. Add `examples/my_use_case/kb/` with `.md` knowledge-base files
4. Add an example `task_*.json`

### Non-pipeline patterns

Use `weaver_for()` which returns the correct weaver type. To add a new IT specialist, for example:

```python
# In scenarios.py — it_harness()
my_specialist = SpecialistAgent(
    name="my_specialist",
    policy_agent=RulePolicyAgent(...),
    planner_agent=TemplatePlannerAgent(...),
)
# Add to routes and router_keywords
```

To add a new guardrail sub-agent:

```python
@dataclass(frozen=True)
class MySubAgent:
    name: str = "my_checker"
    keywords: tuple[str, ...] = ("bad word", "another bad")

    def run(self, task: Task) -> SubAgentFinding:
        matches = [k for k in self.keywords if k in task.text.lower()]
        return SubAgentFinding(
            name=self.name,
            triggered=bool(matches),
            reason=f"Triggered: {matches}" if matches else "Clean.",
            risk_delta=2 if matches else 0,
        )
# Add to sub_agents tuple in guardrail_harness()
```

---

## Optional: Reasoning Model Planner

For pipeline use cases you can swap the template planner for a local Qwen3 reasoning model:

```bash
# Requires Ollama running with qwen3:4b
PYTHONPATH=src python3 -m agentic_harness.cli \
  --use-case support \
  --task examples/support/task_replacement.json \
  --planner ollama \
  --model-profile ollama-qwen3-4b
```

Supported profiles: `ollama-qwen3-4b`, `ollama-qwen3-1.7b`, `vllm-qwen3-4b`, `vllm-qwen3-0.6b`

---

## Running Tests

```bash
PYTHONPATH=src python3 tests/test_claim_weaver.py -v
```

---

## Design Principles

- **One interface**: every weaver exposes `run(task: Task) -> Outcome` regardless of pattern
- **Typed claims**: data flows as immutable dataclasses — no dicts, no magic strings mid-pipeline
- **Auditable**: every step appends to `Outcome.audit_trail` for full traceability
- **No external dependencies**: works with the Python 3.10+ standard library only
- **Pluggable**: swap any agent (signal, policy, planner, verifier, critic, sub-agent) by implementing its protocol

---

## License

MIT
