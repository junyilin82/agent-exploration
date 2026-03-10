# Agent Exploration Project

A hands-on learning project exploring AI agents, from simple tool-calling to multi-agent systems.

## Project Structure

```
agent/
├── lib/       # Shared library (agent_core)
├── phase_1/   # Learning scripts (tool-calling agent)
├── phase_2/   # (Future) Advanced agent patterns
├── apps/      # Demo applications (Streamlit UI, etc.)
└── ...
```

## Phases

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Tool-calling agent with Gemini | Complete |
| 2 | TBD | Planned |

## Apps

| App | Description | Run |
|-----|-------------|-----|
| Streamlit Agent | Web UI demo | `cd apps && uv run streamlit run streamlit_phase1.py` |

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
