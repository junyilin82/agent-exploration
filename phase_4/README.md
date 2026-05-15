# Phase 4: Multi-Agent Systems

Multiple agents working together to solve complex problems.

## Scripts

| Script | Pattern | Description |
|--------|---------|-------------|
| 01_multi_agent_concept.py | Concepts | Overview of multi-agent patterns |
| 02_debate_agents.py | Debate | Two agents critique each other |
| 03_specialist_agents.py | Router | Route to domain specialists |
| 04_orchestrator_worker.py | Orchestrator | Break down and delegate tasks |
| 05_multi_agent_system.py | Combined | Full system with all patterns |

## Patterns

### 1. Debate/Critique
- Agent A proposes
- Agent B critiques
- Iterate until consensus

### 2. Specialist/Router
- Router classifies the request
- Delegates to appropriate specialist

### 3. Orchestrator/Worker
- Orchestrator decomposes task
- Workers execute subtasks
- Orchestrator synthesizes results

## Running

```bash
cd phase_4
uv sync
uv run python 01_multi_agent_concept.py
```
