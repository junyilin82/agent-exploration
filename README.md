# Agent Exploration Project

A hands-on learning project exploring AI agents, from simple tool-calling to multi-agent systems.

## Project Structure

```
agent/
├── phase_1/   # Tool-calling agent (Gemini function calling)
├── phase_2/   # (Future) ReAct pattern / reasoning agents
├── phase_3/   # (Future) Multi-agent systems
└── ...
```

## Phases

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Tool-calling agent with Gemini | In Progress |
| 2 | TBD | Planned |

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
