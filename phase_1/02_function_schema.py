"""
02_function_schema.py - Defining Tools for the Model

PURPOSE:
    Show how to define "tools" that the model can request to use.
    A tool is just: a Python function + a schema describing it.

KEY CONCEPT:
    The model doesn't call functions directly. Instead:
    1. We describe available tools to the model (via schemas)
    2. The model can REQUEST a tool call (returns structured data)
    3. WE execute the function and feed the result back

    The model never runs code - it just asks us to run it.

WHAT YOU'LL SEE:
    - How to define a tool schema (name, description, parameters)
    - How to pass tools to the API
    - What the model returns when it wants to use a tool

RUN:
    uv run python 02_function_schema.py
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Tool

load_dotenv()


# =============================================================================
# STEP 1: Define the actual Python functions
# =============================================================================
# These are real functions that do real work. The model will never see this code.

def calculator(expression: str) -> float:
    """Evaluate a math expression and return the result."""
    # In production, you'd want to sanitize this!
    # For learning, we'll use eval (don't do this with untrusted input)
    return float(eval(expression))


def get_weather(city: str) -> dict:
    """Get weather for a city. (Mock implementation for learning)"""
    # Fake weather data - in production you'd call a real API
    mock_data = {
        "new york": {"temp": 72, "condition": "sunny"},
        "london": {"temp": 55, "condition": "cloudy"},
        "tokyo": {"temp": 68, "condition": "clear"},
    }
    city_lower = city.lower()
    if city_lower in mock_data:
        return mock_data[city_lower]
    return {"temp": 70, "condition": "unknown"}


# =============================================================================
# STEP 2: Define schemas that DESCRIBE these functions to the model
# =============================================================================
# This is what the model sees. It's a description, not the actual code.

calculator_schema = FunctionDeclaration(
    name="calculator",
    description="Evaluate a mathematical expression. Use this for any math calculations.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g., '25 * 4' or '(10 + 5) / 3'",
            },
        },
        "required": ["expression"],
    },
)

weather_schema = FunctionDeclaration(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g., 'New York' or 'London'",
            },
        },
        "required": ["city"],
    },
)

# Bundle schemas into a Tool object
tools = Tool(function_declarations=[calculator_schema, weather_schema])


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    print("=" * 60)
    print("TOOL SCHEMA DEMO")
    print("=" * 60)

    # Show the schemas we defined
    print("\n[TOOL SCHEMAS WE DEFINED]")
    print("-" * 40)
    for func in tools.function_declarations:
        print(f"  Tool: {func.name}")
        print(f"  Description: {func.description}")
        print(f"  Parameters: {func.parameters}")
        print()

    # Now let's ask a question that requires a tool
    user_message = "What is 127 * 348?"

    print("[SENDING TO MODEL]")
    print(f"  Message: {user_message}")
    print(f"  Tools provided: calculator, get_weather")
    print()

    # Pass the tools to the model
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=GenerateContentConfig(tools=[tools]),
    )

    print("[MODEL RESPONSE]")
    print("-" * 40)

    # Check what the model returned
    candidate = response.candidates[0]
    part = candidate.content.parts[0]

    # The model can return either:
    # 1. A text response (if it doesn't need tools)
    # 2. A function_call (if it wants to use a tool)

    if hasattr(part, "function_call") and part.function_call:
        fc = part.function_call
        print(f"  Type: FUNCTION CALL (model wants to use a tool)")
        print(f"  Function name: {fc.name}")
        print(f"  Arguments: {dict(fc.args)}")
        print()
        print("  [!] The model is ASKING us to run this function.")
        print("  [!] It did NOT run the function itself.")
        print("  [!] In the next script, we'll execute it and feed the result back.")
    else:
        print(f"  Type: TEXT RESPONSE")
        print(f"  Content: {response.text}")

    print()
    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  1. Tools = Python functions + JSON schemas describing them")
    print("  2. The model sees ONLY the schema, not the code")
    print("  3. When the model needs a tool, it returns a function_call")
    print("  4. The function_call contains: name + arguments")
    print("  5. WE must execute the function - the model just requests it")
    print("=" * 60)


if __name__ == "__main__":
    main()
