# Agent Exploration Project

A hands-on learning project exploring AI agents, from simple tool-calling to multi-agent systems.

## Project Structure

```
agent/
├── lib/       # Shared library (agent_core)
├── phase_1/   # Learning scripts (tool-calling agent)
├── phase_2/   # React agent patterns
├── phase_3/   # Planning agent patterns
├── phase_4/   # Multi-agent system patterns
├── phase_5/   # Evaluation & guardrails
├── apps/      # Demo applications (Streamlit UI, etc.)
└── ...
```

## Phases

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Tool-calling agent with Gemini | Complete |
| 2 | React agent with Gemini        | Complete |
| 3 | Planning agent with Gemini     | Complete |
| 4 | Multi-agent systems with Gemini | Complete |
| 5 | Evaluation & guardrails with Gemini | Complete |

## Apps

| App | Description | Run |
|-----|-------------|-----|
| Phase 1 Agent | Tool-calling UI | `cd apps && uv run streamlit run streamlit_phase1.py` |
| Phase 2 Agent | ReAct reasoning UI | `cd apps && uv run streamlit run streamlit_phase2.py` |
| Phase 3 Agent | Planning agent UI | `cd apps && uv run streamlit run streamlit_phase3.py` |
| Phase 4 Agent | Multi-agent system UI | `cd apps && uv run streamlit run streamlit_phase4.py` |

## Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies (from phase directory)
cd phase_1
uv sync
```

## Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-key-here
```
