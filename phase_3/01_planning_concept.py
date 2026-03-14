"""
01_planning_concept.py - Understanding Planning Agents

PURPOSE:
    Understand the difference between ReAct and Planning agents.
    Planning = Think ahead, create a full plan, then execute.

REACT (Phase 2):
    User: "Book a flight to Tokyo and find a hotel near Shibuya"

    [Step 1]
    Thought: I need to search for flights to Tokyo.
    Action: search_flights("Tokyo")
    Observation: Found 5 flights...

    [Step 2]
    Thought: Now I need to find hotels near Shibuya.
    Action: search_hotels("Shibuya")
    Observation: Found 3 hotels...

    [Step 3]
    Thought: I have both results. Let me present them.
    Action: finish("Here are your options...")

    OBSERVATION:
    - Decides ONE step at a time
    - No upfront plan
    - Good for simple tasks
    - Can get stuck or go in circles

PLANNING (Phase 3):
    User: "Book a flight to Tokyo and find a hotel near Shibuya"

    [Planning Phase]
    Plan:
    1. Search for flights to Tokyo
    2. Search for hotels near Shibuya
    3. Compare options and present to user

    [Execution Phase]
    Executing step 1: search_flights("Tokyo") → Found 5 flights
    Executing step 2: search_hotels("Shibuya") → Found 3 hotels
    Executing step 3: finish("Here are your options...")

    OBSERVATION:
    - Creates FULL PLAN upfront
    - Then executes step by step
    - Better for complex multi-step tasks
    - Can revise plan if something fails

KEY INSIGHT:
    Planning agents separate THINKING from DOING.
    - First: Think through the entire problem
    - Then: Execute the plan
    - If needed: Revise and re-plan

RUN:
    uv run python 01_planning_concept.py
"""

REACT_EXAMPLE = """
=== REACT APPROACH (Phase 2) ===

User: Calculate 20% of my monthly salary ($5000), then find the weather
      in Tokyo, and tell me if I can afford a $900 umbrella with that 20%.

[Step 1]
Thought: I need to calculate 20% of $5000 first.
Action: calculator("0.20 * 5000")
Observation: 1000.0

[Step 2]
Thought: Now I need to get Tokyo weather to understand why they need an umbrella.
Action: get_weather("Tokyo")
Observation: 68°F, rainy

[Step 3]
Thought: 20% of salary is $1000. The umbrella costs $900. $1000 > $900, so yes.
Action: finish("Yes! 20% of $5000 is $1000, which covers the $900 umbrella.
        And you might need it - Tokyo is currently rainy!")

CHARACTERISTICS:
- Makes decisions step-by-step
- No advance planning
- Works well for this simple case
"""

PLANNING_EXAMPLE = """
=== PLANNING APPROACH (Phase 3) ===

User: Calculate 20% of my monthly salary ($5000), then find the weather
      in Tokyo, and tell me if I can afford a $900 umbrella with that 20%.

[PLANNING PHASE]
Let me create a plan:

Plan:
  Step 1: Calculate 20% of $5000
          Tool: calculator("0.20 * 5000")
          Expected: A number representing 20% of salary

  Step 2: Get Tokyo weather
          Tool: get_weather("Tokyo")
          Expected: Weather conditions

  Step 3: Compare savings ($from step 1) with umbrella cost ($900)
          Tool: calculator("result_from_step_1 - 900")
          Expected: Positive = can afford, Negative = cannot afford

  Step 4: Compile final answer
          Tool: finish()
          Expected: Complete answer with all information

[EXECUTION PHASE]
Executing Step 1: calculator("0.20 * 5000") → 1000.0 ✓
Executing Step 2: get_weather("Tokyo") → 68°F, rainy ✓
Executing Step 3: calculator("1000 - 900") → 100.0 ✓
Executing Step 4: finish("Yes, you can afford it!...") ✓

CHARACTERISTICS:
- Creates full plan BEFORE any action
- Can identify dependencies between steps
- Easier to track progress
- Can revise plan if step fails
"""

WHEN_TO_USE = """
=== WHEN TO USE EACH APPROACH ===

Use REACT when:
- Task is simple (1-3 steps)
- Steps are independent
- You need quick, adaptive responses
- The path forward is obvious

Use PLANNING when:
- Task is complex (4+ steps)
- Steps have dependencies (step 3 needs result of step 1)
- You need to track progress
- Failure recovery is important
- You want to show the user the plan before executing

REAL-WORLD EXAMPLES:

React is better for:
- "What's the weather in Paris?"
- "Calculate 15% tip on $50"
- Quick Q&A

Planning is better for:
- "Research competitors, analyze their pricing, and draft a report"
- "Debug this code: first reproduce the bug, then identify the cause, then fix it"
- "Plan a trip: check flights, hotels, and create an itinerary"
"""

PLANNING_COMPONENTS = """
=== COMPONENTS OF A PLANNING AGENT ===

1. PLAN GENERATOR
   - Takes user request
   - Outputs a structured plan (list of steps)
   - Each step has: description, tool, expected result

2. PLAN PARSER
   - Converts model's text plan into structured data
   - Similar to ReAct parser, but for plans

3. PLAN EXECUTOR
   - Executes steps one by one
   - Tracks completed vs pending steps
   - Handles dependencies

4. PLAN REVISER (optional)
   - When a step fails, creates a new plan
   - Incorporates what was learned from failure
"""


def main():
    print(REACT_EXAMPLE)
    print(PLANNING_EXAMPLE)
    print(WHEN_TO_USE)
    print(PLANNING_COMPONENTS)

    print("\n" + "=" * 60)
    print("NEXT: In 02_planning_prompt.py, we'll design the prompt")
    print("that makes the model output a structured plan.")
    print("=" * 60)


if __name__ == "__main__":
    main()
