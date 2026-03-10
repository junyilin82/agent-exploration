"""
04_agent_loop.py - The Complete Agent Loop

PURPOSE:
    Handle multiple tool calls in a loop until the model gives a final answer.
    This is the core pattern for a tool-calling agent.

THE LOOP:
    while True:
        response = call_model(messages)
        if response is text:
            return response  # Done!
        else:
            for each function_call in response:
                result = execute(function_call)
                add result to messages
            continue loop

KEY CONCEPT:
    We keep looping until the model returns text (no more tool calls).
    The model decides when it has enough information to answer.

RUN:
    uv run python 04_agent_loop.py
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    Tool,
)

load_dotenv()


# =============================================================================
# Tool implementations
# =============================================================================

def calculator(expression: str) -> float:
    """Evaluate a math expression."""
    return float(eval(expression))


def get_weather(city: str) -> dict:
    """Get weather for a city (mock)."""
    mock_data = {
        "new york": {"temp_fahrenheit": 72, "condition": "sunny"},
        "london": {"temp_fahrenheit": 55, "condition": "cloudy"},
        "tokyo": {"temp_fahrenheit": 68, "condition": "clear"},
    }
    return mock_data.get(city.lower(), {"temp_fahrenheit": 70, "condition": "unknown"})


FUNCTION_MAP = {
    "calculator": calculator,
    "get_weather": get_weather,
}


# =============================================================================
# Tool schemas
# =============================================================================

tools = Tool(function_declarations=[
    FunctionDeclaration(
        name="calculator",
        description="Evaluate a mathematical expression. Use for any math.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g., '25 * 4 + 10'",
                },
            },
            "required": ["expression"],
        },
    ),
    FunctionDeclaration(
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g., 'Tokyo'",
                },
            },
            "required": ["city"],
        },
    ),
])


def run_agent(client: genai.Client, model_name: str, user_query: str) -> str:
    """
    Run the agent loop until we get a final text response.

    This is the core agent pattern:
    1. Send messages to model
    2. If model returns function_call(s), execute them and loop
    3. If model returns text, we're done
    """
    print(f"\n{'='*60}")
    print(f"USER QUERY: {user_query}")
    print(f"{'='*60}")

    # Initialize conversation
    contents = [Content(role="user", parts=[Part.from_text(text=user_query)])]

    loop_count = 0
    max_loops = 10  # Safety limit

    while loop_count < max_loops:
        loop_count += 1
        print(f"\n[LOOP {loop_count}] Calling model...")

        # Call the model
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=GenerateContentConfig(tools=[tools]),
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        # Check if any part is a function call
        function_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

        if not function_calls:
            # No function calls = model is done, return text
            final_text = response.text
            print(f"\n[DONE] Model returned text response")
            print(f"       Loops taken: {loop_count}")
            return final_text

        # Process function calls
        print(f"       Model requested {len(function_calls)} tool(s):")

        # Add model's response (with function calls) to history
        contents.append(Content(role="model", parts=parts))

        # Execute each function and collect results
        function_response_parts = []

        for part in function_calls:
            fc = part.function_call
            func_name = fc.name
            func_args = dict(fc.args)

            print(f"         - {func_name}({func_args})")

            # Execute the function
            func = FUNCTION_MAP[func_name]
            result = func(**func_args)

            print(f"           Result: {result}")

            # Create function response
            function_response_parts.append(
                Part.from_function_response(name=func_name, response={"result": result})
            )

        # Add all function results to history
        contents.append(Content(role="user", parts=function_response_parts))

    return "Error: Max loops reached"


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    # Test cases
    queries = [
        # Simple: one tool
        "What is 456 * 789?",

        # Complex: needs multiple tools
        "What is 100 * 5 plus the temperature in Tokyo?",
    ]

    for query in queries:
        result = run_agent(client, model_name, query)
        print(f"\n[FINAL ANSWER]: {result}")
        print("\n" + "="*60)

    # Show the loop pattern
    print("""
THE AGENT LOOP PATTERN:

    ┌─────────────────────────────────────────┐
    │         while True:                     │
    │                                         │
    │   response = model(messages)            │
    │                                         │
    │   if response.is_text:                  │
    │       return response   ──────────────────► DONE
    │                                         │
    │   for fc in response.function_calls:    │
    │       result = execute(fc)              │
    │       messages.append(result)           │
    │                                         │
    │   continue ─────────────────────────────┐
    │                                         │
    └─────────────────────────────────────────┘

KEY INSIGHT:
    The MODEL decides when to stop (by returning text instead of tool calls).
    We just keep executing whatever it asks for.
""")


if __name__ == "__main__":
    main()
