# Agent Apps

Demo applications built on top of the agent learning scripts.

## Apps

| App | Description |
|-----|-------------|
| `streamlit_phase1.py` | Phase 1: Tool-calling agent UI |
| `streamlit_phase2.py` | Phase 2: ReAct agent with reasoning steps |
| `streamlit_phase3.py` | Phase 3: Planning agent with plan & execution |
| `streamlit_phase4.py` | Phase 4: Multi-agent system (Debate, Router, Orchestrator, Combined) |

## Run locally

```bash
cd apps
uv sync
uv run streamlit run streamlit_phase4.py
```

Opens at http://localhost:8501

## Phase 4 Features

- 4 selectable multi-agent patterns via sidebar
- **Debate**: Proposer/Critic iterate until consensus
- **Specialist Router**: Routes to domain-specific agents
- **Orchestrator/Worker**: Decomposes, delegates, synthesizes
- **Combined System**: Classifier + routing + optional critic
- Agent Flow expander shows each agent step with icons and metadata
