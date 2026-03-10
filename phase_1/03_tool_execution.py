"""
03_tool_execution.py - Executing Tools and Feeding Results Back

PURPOSE:
    Complete the loop: execute the function the model requested,
    then send the result back so the model can respond.

THE FULL FLOW:
    1. User asks a question
    2. Model returns function_call (requests a tool)
    3. WE execute the function → get result
    4. WE send the result back to the model
    5. Model generates final text response

KEY CONCEPT:
    The "conversation" includes tool calls and results.
    We build up a history: user message → assistant function_call → tool result → assistant text

RUN:
    uv run python 03_tool_execution.py
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
# Tool implementations (same as before)
# =============================================================================

def calculator(expression: str) -> float:
    """Evaluate a math expression and return the result."""
    return float(eval(expression))


def get_weather(city: str) -> dict:
    """Get weather for a city. (Mock implementation)"""
    mock_data = {
        "new york": {"temp": 72, "condition": "sunny"},
        "london": {"temp": 55, "condition": "cloudy"},
        "tokyo": {"temp": 68, "condition": "clear"},
    }
    return mock_data.get(city.lower(), {"temp": 70, "condition": "unknown"})


# Map function names to actual functions
FUNCTION_MAP = {
    "calculator": calculator,
    "get_weather": get_weather,
}


# =============================================================================
# Tool schemas (same as before)
# =============================================================================

calculator_schema = FunctionDeclaration(
    name="calculator",
    description="Evaluate a mathematical expression. Use this for any math calculations.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g., '25 * 4'",
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

tools = Tool(function_declarations=[calculator_schema, weather_schema])


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    user_message = "What is 127 * 348?"

    print("=" * 60)
    print("TOOL EXECUTION DEMO")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: Send user message to model (with tools)
    # -------------------------------------------------------------------------
    print(f"\n[STEP 1] User message: {user_message}")

    # Start building conversation history
    contents = [Content(role="user", parts=[Part.from_text(text=user_message)])]

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=GenerateContentConfig(tools=[tools]),
    )

    # -------------------------------------------------------------------------
    # STEP 2: Model returns function_call
    # -------------------------------------------------------------------------
    assistant_part = response.candidates[0].content.parts[0]
    fc = assistant_part.function_call

    print(f"\n[STEP 2] Model requests tool:")
    print(f"         Function: {fc.name}")
    print(f"         Arguments: {dict(fc.args)}")

    # Add assistant's response to history
    contents.append(Content(role="model", parts=[assistant_part]))

    # -------------------------------------------------------------------------
    # STEP 3: WE execute the function
    # -------------------------------------------------------------------------
    func = FUNCTION_MAP[fc.name]
    args = dict(fc.args)
    result = func(**args)

    print(f"\n[STEP 3] We execute: {fc.name}({args})")
    print(f"         Result: {result}")

    # -------------------------------------------------------------------------
    # STEP 4: Send result back to model
    # -------------------------------------------------------------------------
    # Create a "function response" part
    function_response_part = Part.from_function_response(
        name=fc.name,
        response={"result": result},
    )

    # Add tool result to history
    contents.append(Content(role="user", parts=[function_response_part]))

    print(f"\n[STEP 4] Sending result back to model...")

    # Call model again with the full history
    response2 = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=GenerateContentConfig(tools=[tools]),
    )

    # -------------------------------------------------------------------------
    # STEP 5: Model generates final response
    # -------------------------------------------------------------------------
    final_response = response2.text

    print(f"\n[STEP 5] Model's final response:")
    print(f"         {final_response}")

    # -------------------------------------------------------------------------
    # Show the full conversation history
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FULL CONVERSATION HISTORY")
    print("=" * 60)
    print("""
    ┌─────────────────────────────────────────────────────┐
    │ 1. USER: "What is 127 * 348?"                       │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │ 2. MODEL: function_call(calculator, "127 * 348")    │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │ 3. US: Execute calculator("127 * 348") → 44196      │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │ 4. US → MODEL: "Result is 44196"                    │
    └─────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────┐
    │ 5. MODEL: "127 * 348 = 44196"                       │
    └─────────────────────────────────────────────────────┘
    """)

    print("=" * 60)
    print("KEY TAKEAWAY:")
    print("  The 'agent loop' is just: call model → execute tool → feed back")
    print("  We made TWO API calls to complete one user request.")
    print("  Next: What if the model needs MULTIPLE tools?")
    print("=" * 60)


if __name__ == "__main__":
    main()
