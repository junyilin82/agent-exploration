# Phase 2: ReAct Pattern

Learn the ReAct (Reasoning + Acting) pattern - a foundational agent architecture.

## What is ReAct?

ReAct makes the agent's reasoning explicit by forcing a **Thought → Action → Observation** loop.

```
User: What is 15% of 200 plus the temperature in Tokyo?

Thought: I need to calculate 15% of 200 first.
Action: calculator("0.15 * 200")
Observation: 30.0

Thought: Now I need to get the temperature in Tokyo.
Action: get_weather("Tokyo")
Observation: {"temperature_f": 68, "condition": "clear"}

Thought: I have both values. 30 + 68 = 98. I can answer now.
Action: finish("15% of 200 is 30, plus Tokyo's temperature of 68°F equals 98.")
```

## Why ReAct?

| Benefit | Description |
|---------|-------------|
| **Transparency** | See the agent's reasoning, not just actions |
| **Accuracy** | Explicit reasoning improves decision-making |
| **Debugging** | Easy to trace where things went wrong |
| **Control** | Can validate thoughts before actions |

## Scripts (Progressive Learning)

| Script | Purpose |
|--------|---------|
| `01_react_concept.py` | Understand the Thought/Action/Observation format |
| `02_react_prompt.py` | Design the ReAct prompt |
| `03_react_parser.py` | Parse structured output from model |
| `04_react_loop.py` | Complete ReAct agent loop |
| `05_react_vs_basic.py` | Compare ReAct vs basic tool-calling |

## Key Difference from Phase 1

**Phase 1 (Tool Calling):**
- Model returns structured `function_call` objects
- We execute and feed back `function_response`
- Reasoning is implicit (hidden in model's weights)

**Phase 2 (ReAct):**
- Model outputs text in Thought/Action format
- We parse the text, execute actions, append observations
- Reasoning is explicit (visible in output)

## Setup

```bash
cd phase_2
uv sync
```
