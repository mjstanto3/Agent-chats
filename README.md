# Multi-Agent LLM Debate Framework

A modular Python framework for **deliberative multi-agent reasoning** where LLM agents debate a topic, challenge each other's arguments, and iteratively refine an answer through structured conversation.

---

## Overview

Instead of producing independent answers, agents in this framework:

- **Challenge** each other's claims
- **Defend** positions with evidence
- **Critique** weak assumptions
- **Synthesise** a final, refined answer

A **Debate Blackboard** (structured state) is maintained alongside the conversation transcript, so the Moderator can route the next agent based on reasoning gaps—not just simple turn-taking.

---

## Project Structure

```
multi_agent_debate/
├── agents/
│   ├── base_agent.py          # Agent class with all reasoning methods
│   └── agent_factory.py       # Build agents from YAML config
├── moderator/
│   └── moderator_agent.py     # Blackboard-driven agent routing
├── conversation/
│   ├── conversation_manager.py  # Transcript storage + JSON/Markdown export
│   └── message_schema.py      # Message dataclass
├── blackboard/
│   └── debate_state.py        # Structured debate state (claims, objections, …)
├── orchestrator/
│   └── debate_orchestrator.py # Debate loop controller
├── synthesis/
│   └── synthesizer.py         # Final answer generator
├── config/
│   ├── agents.yaml            # Agent definitions
│   └── settings.yaml          # Global settings
├── prompts/
│   └── agent_prompts/         # Role-specific system prompt templates
│       ├── proponent.txt
│       ├── critic.txt
│       ├── researcher.txt
│       ├── analyst.txt
│       └── synthesizer.txt
└── outputs/                   # Generated transcripts (gitkeep)
main.py
requirements.txt
README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run in offline (mock) mode — no API key required

```bash
python main.py
```

This runs a demo debate using stub LLM responses so you can verify the entire pipeline end-to-end without an API key.

### 3. Run with OpenAI

```bash
export OPENAI_API_KEY=sk-...
python main.py --backend openai --topic "Should AI be regulated by governments?"
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--topic` | AI regulation question | The debate topic |
| `--backend` | `mock` | `mock` (offline) or `openai` (live) |
| `--rounds` | `6` | Maximum debate rounds |
| `--output-dir` | `outputs/` | Where to save transcripts |
| `--config` | `multi_agent_debate/config/agents.yaml` | Agent config file |
| `--prompts-dir` | `multi_agent_debate/prompts/agent_prompts` | Prompt template directory |
| `--log-level` | `INFO` | Logging verbosity |

---

## Architecture

### Debate Blackboard

The central shared state updated after every agent turn:

```json
{
  "topic": "...",
  "claims": [],
  "objections": [],
  "open_questions": [],
  "consensus_points": [],
  "candidate_answer": ""
}
```

### Agent Message Schema

Agents output a structured JSON envelope alongside their message text:

```json
{
  "message": "...",
  "state_update": {
    "new_claims": [],
    "new_objections": [],
    "new_questions": [],
    "new_consensus": [],
    "candidate_answer": ""
  }
}
```

### Moderator Routing Logic

The Moderator reads the blackboard and routes the next agent:

| Blackboard condition | Next agent |
|----------------------|-----------|
| Unresolved objections | Critic |
| Open questions / missing evidence | Researcher |
| Conflicting claims | Analyst |
| Consensus emerging | Synthesizer |
| Default / opening | Proponent |

### Transcript Export

After every debate the orchestrator saves:

- `outputs/<topic>_transcript.json` — full structured transcript
- `outputs/<topic>_transcript.md` — human-readable Markdown
- `outputs/<topic>_final_answer.txt` — the synthesised conclusion

---

## Configuration

### `config/agents.yaml`

Define the debate participants:

```yaml
agents:
  - name: Proponent
    role: proponent
    model: gpt-4o
    temperature: 0.7
  - name: Critic
    role: critic
    model: gpt-4o
    temperature: 0.7
  # …
```

### `config/settings.yaml`

```yaml
debate:
  max_rounds: 10
  backend: openai
  output_dir: outputs
llm:
  default_model: gpt-4o
  moderator_model: gpt-4o
  synthesizer_model: gpt-4o
```

---

## Extending the Framework

The architecture is designed to support future capabilities:

- **Tool-using agents** — override `_llm()` in `Agent` to call tools before responding
- **Retrieval-augmented debate** — inject retrieved documents into `_build_context()`
- **Evaluation / judge agents** — add a new role (`judge`) and routing rule in `ModeratorAgent`
- **Human moderation** — replace `ModeratorAgent.decide()` with an interactive CLI prompt
- **Debate visualisation** — parse the exported JSON transcript into a graph or timeline

---

## Running Tests

```bash
pytest tests/
```