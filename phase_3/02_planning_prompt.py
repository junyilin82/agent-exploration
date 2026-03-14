"""
02_planning_prompt.py - Designing the Planning Prompt

PURPOSE:
    Create a prompt that makes the model output a structured plan.
    The plan must be parseable so we can execute it step by step.

PLAN FORMAT:
    We need the model to output something like:

    Plan:
    1. [Description] | Tool: [tool_name("arg")] | Expected: [what we expect]
    2. [Description] | Tool: [tool_name("arg")] | Expected: [what we expect]
    ...

    Or in a more structured format that's easier to parse.

RUN:
    uv run python 02_planning_prompt.py
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


# =============================================================================
# THE PLANNING PROMPT
# =============================================================================

PLANNING_PROMPT = '''You are a planning assistant that creates step-by-step plans.

## Your Task

Given a user request, create a detailed plan to accomplish it.
Do NOT execute the plan - only create it.

## Plan Format

You must output your plan in this EXACT format:

PLAN:
[1] <description of step 1>
    Tool: <tool_name>("argument")
    Expect: <what result you expect>

[2] <description of step 2>
    Tool: <tool_name>("argument")
    Expect: <what result you expect>

[3] ... and so on

END_PLAN

## Available Tools

1. calculator(expression) - Evaluate a math expression
   Example: calculator("15 * 20 + 5")

2. get_weather(city) - Get weather for a city
   Example: get_weather("Tokyo")

3. get_time(timezone) - Get current time in a timezone
   Example: get_time("EST")

4. search(query) - Search for information
   Example: search("best restaurants in Paris")

## Rules

- Create a complete plan before any execution
- Each step should have exactly one Tool
- Steps should be in logical order
- If a step depends on a previous result, note it in the description
- The final step should compile results into an answer
- Use "finish" as the tool for the final step

## Example

User: What is 30% of 200, and is that enough to buy a $50 item?

PLAN:
[1] Calculate 30% of 200
    Tool: calculator("0.30 * 200")
    Expect: A number representing 30% of 200

[2] Compare the result with $50
    Tool: calculator("result_from_step_1 - 50")
    Expect: Positive if enough, negative if not enough

[3] Provide final answer based on comparison
    Tool: finish("answer based on calculations")
    Expect: Clear yes/no answer with explanation

END_PLAN

Now create a plan for the following request:

'''


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    print("=" * 60)
    print("PLANNING PROMPT DEMO")
    print("=" * 60)

    print("\n[THE PROMPT]")
    print("-" * 40)
    print(PLANNING_PROMPT)

    print("\n[TESTING THE PROMPT]")
    print("-" * 40)

    # Test with a multi-step question
    user_request = "What is 15% of my $3000 savings, and what's the weather like in London? Tell me both."

    full_prompt = f"{PLANNING_PROMPT}\n\nUser: {user_request}"

    print(f"User: {user_request}\n")
    print("Model response:")

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=GenerateContentConfig(temperature=0),
    )

    print(response.text)

    print("\n" + "=" * 60)
    print("OBSERVATION:")
    print("  The model outputs a structured PLAN, not actions.")
    print("  We need to PARSE this plan to extract steps.")
    print("  That's what we'll build in 03_plan_parser.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
