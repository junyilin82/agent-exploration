# Phase 3: Planning Agents

Planning agents create a complete plan before executing, unlike ReAct which decides step by step.

## Key Difference from ReAct

| Aspect | ReAct (Phase 2) | Planning (Phase 3) |
|--------|-----------------|-------------------|
| Decision timing | One step at a time | Full plan upfront |
| Visibility | See each thought | See entire plan |
| Dependencies | Implicit | Explicit |
| Best for | Simple tasks | Complex multi-step tasks |

## Scripts

1. **01_planning_concept.py** - Understand planning vs ReAct
2. **02_planning_prompt.py** - Design prompts for plan generation
3. **03_plan_parser.py** - Parse plans into structured data
4. **04_plan_executor.py** - Execute plans step by step
5. **05_planning_agent.py** - Complete planning agent

## Run

```bash
cd phase_3
uv sync
uv run python 01_planning_concept.py
```

## The Planning Loop

```
User Request
    ↓
[PLANNING PHASE]
    Generate Plan → Parse Plan
    ↓
[EXECUTION PHASE]
    Execute Step 1 → Result
    Execute Step 2 → Result (may use Step 1 result)
    ...
    Execute finish() → Final Answer
```

## Plan Format

```
PLAN:
[1] Description of step
    Tool: tool_name("argument")
    Expect: What we expect

[2] Next step (can reference {step_1})
    Tool: tool_name("{step_1} + 10")
    Expect: Result using previous step

[3] Final answer
    Tool: finish("Answer using {step_1} and {step_2}")
    Expect: Complete answer

END_PLAN
```
