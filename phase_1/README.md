# Phase 1: Tool-Calling Agent

Learn how LLM-based tool-calling agents work by building one from scratch.

## What You'll Learn

1. **What is a "tool"?** - A function with a JSON schema describing its inputs
2. **How does the model choose?** - It returns structured `tool_calls` instead of text
3. **The agent loop** - Model → tool request → execute → feed result back → repeat
4. **When does it stop?** - Model returns text (no tool calls) when task is complete

## Scripts (Progressive Learning)

| Script | Purpose |
|--------|---------|
| `01_basic_completion.py` | Baseline: simple OpenAI chat completion |
| `02_function_schema.py` | Define tools as JSON schemas |
| `03_tool_calling.py` | Let the model request tool calls |
| `04_execution_loop.py` | Execute tools and feed results back |
| `05_full_agent.py` | Complete agent with logging |

## The Agent Loop (Core Concept)

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  User Query: "What's 25 * 4 plus the temp in NYC?"      │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────┐            │
│  │            LLM (GPT-4o)                 │            │
│  │  "I need to call calculator first"      │            │
│  └─────────────────────────────────────────┘            │
│                          │                              │
│                          ▼                              │
│           tool_calls: [{calculator: 25*4}]              │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────┐            │
│  │       Execute: calculator(25*4)         │            │
│  │       Result: 100                       │            │
│  └─────────────────────────────────────────┘            │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────┐            │
│  │            LLM (GPT-4o)                 │            │
│  │  "Now I need weather for NYC"           │            │
│  └─────────────────────────────────────────┘            │
│                          │                              │
│                          ▼                              │
│           tool_calls: [{weather: "NYC"}]                │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────┐            │
│  │       Execute: weather("NYC")           │            │
│  │       Result: 72°F                      │            │
│  └─────────────────────────────────────────┘            │
│                          │                              │
│                          ▼                              │
│  ┌─────────────────────────────────────────┐            │
│  │            LLM (GPT-4o)                 │            │
│  │  "I have all the info now"              │            │
│  └─────────────────────────────────────────┘            │
│                          │                              │
│                          ▼                              │
│        Response: "25*4=100, NYC is 72°F"                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Key Insight

The "agent" isn't magic. It's a loop:
1. Send messages to LLM
2. If response contains `tool_calls` → execute them, add results to messages, goto 1
3. If response is text → return to user

## Tools

- **Calculator**: Evaluates math expressions
- **Weather**: Returns weather for a city (mock API for learning)

## Setup

```bash
uv sync
cp ../.env.example .env  # Add your OpenAI API key
```

## Run

```bash
# Run each script in order to learn progressively
python 01_basic_completion.py
python 02_function_schema.py
# ... etc
```
