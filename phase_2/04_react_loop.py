"""
04_react_loop.py - Complete ReAct Agent Loop

PURPOSE:
    Build the full ReAct agent that:
    1. Sends query + prompt to model
    2. Parses Thought/Action from response
    3. Executes action (or returns if finish)
    4. Appends Observation to context
    5. Repeats until finish() is called

THE LOOP:
    context = prompt + user_query
    while True:
        response = model(context)
        parsed = parse(response)
        if parsed.action == "finish":
            return parsed.action_arg
        result = execute(parsed.action)
        context += f"\nObservation: {result}\n"

RUN:
    uv run python 04_react_loop.py
"""

import os
import re
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


# =============================================================================
# REACT PROMPT
# =============================================================================

REACT_PROMPT = '''You are a helpful assistant that solves problems step by step.

## How to Respond

You must ALWAYS respond in this exact format:

Thought: [Your reasoning about what to do next]
Action: [tool_name("argument")]

After each action, you will see an Observation with the result.
Then continue with another Thought/Action until you have the final answer.

When you have the final answer, use:
Thought: [Your final reasoning]
Action: finish("[Your complete answer to the user]")

You MUST include Thought before EVERY Action. Never skip it.

## Available Tools

1. calculator(expression) - Evaluate a math expression
   Example: calculator("15 * 20 + 5")

2. get_weather(city) - Get weather for a city
   Example: get_weather("Tokyo")

3. get_time(timezone) - Get current time in a timezone (UTC, EST, PST, JST)
   Example: get_time("EST")

4. finish(answer) - Return the final answer to the user
   Example: finish("The answer is 42")

## Rules

- Always start with a Thought explaining your reasoning
- Only use ONE Action per response
- Wait for the Observation before continuing
- Use finish() only when you have the complete answer

## Example

User: What is 25% of 80?

Thought: I need to calculate 25% of 80. 25% as a decimal is 0.25, so I multiply 0.25 * 80.
Action: calculator("0.25 * 80")

[Then you would see: Observation: 20.0]

Thought: The calculation shows 25% of 80 is 20. I have the answer.
Action: finish("25% of 80 is 20")

Now solve the following:

'''


# =============================================================================
# TOOLS
# =============================================================================

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Get weather for a city (mock)."""
    mock_data = {
        "new york": "72°F, sunny",
        "london": "55°F, cloudy",
        "tokyo": "68°F, clear",
        "paris": "63°F, rainy",
    }
    return mock_data.get(city.lower(), "70°F, unknown conditions")


def get_time(timezone: str) -> str:
    """Get current time (mock)."""
    mock_times = {
        "utc": "14:30 UTC",
        "est": "09:30 EST",
        "pst": "06:30 PST",
        "jst": "23:30 JST",
    }
    return mock_times.get(timezone.lower(), "12:00 (unknown timezone)")


TOOLS: dict[str, Callable] = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_time": get_time,
}


# =============================================================================
# PARSER
# =============================================================================

@dataclass
class ParsedAction:
    thought: str | None
    action_name: str
    action_arg: str


def parse_react_output(text: str) -> ParsedAction:
    """Parse Thought/Action from model output."""
    # Extract thought
    thought = None
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action
    action_match = re.search(r"Action:\s*(\w+)\s*\(\s*[\"'](.+?)[\"']\s*\)", text, re.DOTALL)
    if not action_match:
        action_match = re.search(r"Action:\s*(\w+)\s*\(\s*(.+?)\s*\)", text, re.DOTALL)

    if not action_match:
        raise ValueError(f"Could not parse action from: {text}")

    return ParsedAction(
        thought=thought,
        action_name=action_match.group(1),
        action_arg=action_match.group(2).strip().strip("\"'"),
    )


# =============================================================================
# REACT AGENT
# =============================================================================

def react_agent(
    client: genai.Client,
    user_query: str,
    model_name: str = "gemini-2.5-flash",
    max_steps: int = 10,
    verbose: bool = True,
) -> str:
    """
    Run the ReAct agent loop.

    Returns the final answer.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"User: {user_query}")
        print(f"{'='*60}")

    # Build initial context
    context = REACT_PROMPT + f"User: {user_query}\n"

    for step in range(max_steps):
        if verbose:
            print(f"\n[Step {step + 1}]")

        # Get model response
        response = client.models.generate_content(
            model=model_name,
            contents=context,
            config=GenerateContentConfig(temperature=0),
        )

        output = response.text.strip()

        if verbose:
            print(f"Model:\n{output}")

        # Parse the output
        try:
            parsed = parse_react_output(output)
        except ValueError as e:
            if verbose:
                print(f"Parse error: {e}")
            return f"Error: Could not parse model output"

        # Check if done
        if parsed.action_name == "finish":
            if verbose:
                print(f"\n{'─'*60}")
                print(f"Final Answer: {parsed.action_arg}")
                print(f"{'─'*60}")
            return parsed.action_arg

        # Execute the action
        if parsed.action_name not in TOOLS:
            observation = f"Error: Unknown tool '{parsed.action_name}'"
        else:
            tool = TOOLS[parsed.action_name]
            observation = tool(parsed.action_arg)

        if verbose:
            print(f"Observation: {observation}")

        # Append to context
        context += f"{output}\nObservation: {observation}\n\n"

    return "Error: Max steps reached"


# =============================================================================
# MAIN
# =============================================================================

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return

    client = genai.Client(api_key=api_key)

    # Test queries
    queries = [
        "What is 15% of 200?",
        "What is 25 * 4 plus the temperature in Tokyo (just the number)?",
    ]

    for query in queries:
        react_agent(client, query, verbose=True)
        print("\n")


if __name__ == "__main__":
    main()
