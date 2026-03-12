"""
01_react_concept.py - Understanding the ReAct Pattern

PURPOSE:
    Understand the difference between basic tool-calling and ReAct.
    ReAct = Reasoning + Acting (explicit thought process)

PHASE 1 (Basic Tool Calling):
    User: "What is 15% of 200?"
    Model returns: function_call(calculator, "0.15 * 200")
    We execute: 30.0
    Model returns: "15% of 200 is 30"

    Problem: We don't see WHY the model chose that action.

PHASE 2 (ReAct):
    User: "What is 15% of 200?"
    Model returns:
        Thought: I need to calculate 15% of 200. 15% as decimal is 0.15.
        Action: calculator("0.15 * 200")
    We execute: 30.0
    We append: Observation: 30.0
    Model returns:
        Thought: The calculation is complete. 15% of 200 equals 30.
        Action: finish("15% of 200 is 30")

    Benefit: We see the reasoning!

KEY INSIGHT:
    ReAct doesn't use Gemini's built-in function_call feature.
    Instead, we prompt the model to output structured TEXT that we parse.

RUN:
    uv run python 01_react_concept.py
"""

# This script is conceptual - showing the format difference

PHASE_1_EXAMPLE = """
=== PHASE 1: Basic Tool Calling ===

User: What is 15% of 200 plus the temperature in Tokyo?

[API Call 1]
Model returns: function_call(calculator, {"expression": "0.15 * 200"})
               function_call(get_weather, {"city": "Tokyo"})

[We execute both tools]
Results: calculator -> 30.0, get_weather -> {"temperature_f": 68}

[API Call 2]
Model returns: "15% of 200 is 30. Tokyo is 68°F. Total: 98"

OBSERVATION:
- Fast (parallel tool calls)
- But no visibility into reasoning
- We don't know WHY it chose those tools
"""

PHASE_2_EXAMPLE = """
=== PHASE 2: ReAct Pattern ===

User: What is 15% of 200 plus the temperature in Tokyo?

[API Call 1]
Model returns:
    Thought: I need to break this into steps. First, calculate 15% of 200.
    Action: calculator("0.15 * 200")

[We parse and execute]
Observation: 30.0

[API Call 2]
Model returns:
    Thought: 15% of 200 is 30. Now I need Tokyo's temperature.
    Action: get_weather("Tokyo")

[We parse and execute]
Observation: {"temperature_f": 68, "condition": "clear"}

[API Call 3]
Model returns:
    Thought: I have both values: 30 and 68. Adding them: 30 + 68 = 98.
    Action: finish("15% of 200 is 30, plus Tokyo's temperature of 68°F equals 98.")

OBSERVATION:
- Slower (sequential, one action at a time)
- But full visibility into reasoning
- We see exactly WHY each action was taken
- Easier to debug and validate
"""

COMPARISON = """
=== COMPARISON ===

| Aspect              | Phase 1 (Tool Calling) | Phase 2 (ReAct)        |
|---------------------|------------------------|------------------------|
| Model output        | Structured function_call | Text with format       |
| Reasoning visible   | No (implicit)          | Yes (explicit)         |
| Tool calls          | Can be parallel        | Sequential             |
| Speed               | Faster                 | Slower                 |
| Debugging           | Harder                 | Easier                 |
| Implementation      | Built-in API feature   | Prompt engineering     |

WHEN TO USE EACH:

Tool Calling (Phase 1):
- Simple, straightforward tasks
- Speed matters
- Tools are independent

ReAct (Phase 2):
- Complex multi-step reasoning
- Need to validate/audit decisions
- Tasks require careful planning
- Debugging and transparency matter
"""


def main():
    print(PHASE_1_EXAMPLE)
    print(PHASE_2_EXAMPLE)
    print(COMPARISON)

    print("\n" + "=" * 60)
    print("NEXT: In 02_react_prompt.py, we'll design the prompt")
    print("that makes the model output in Thought/Action format.")
    print("=" * 60)


if __name__ == "__main__":
    main()
