"""
02b_tools_comparison.py - What Makes the Model Output Structured Data?

PURPOSE:
    Show the exact difference between a regular call and a tool-enabled call.
    The ONLY difference is whether we pass tools= to the API.

RUN:
    uv run python 02b_tools_comparison.py
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Tool

load_dotenv()

# Define a simple tool
calculator_schema = FunctionDeclaration(
    name="calculator",
    description="Evaluate a mathematical expression.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate",
            },
        },
        "required": ["expression"],
    },
)
tools = Tool(function_declarations=[calculator_schema])


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    prompt = "What is 127 * 348?"

    print("=" * 60)
    print("SAME PROMPT, TWO DIFFERENT API CALLS")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print()

    # -----------------------------------------------------------------
    # Call 1: WITHOUT tools
    # -----------------------------------------------------------------
    print("[CALL 1: WITHOUT tools=]")
    print("-" * 40)

    response1 = client.models.generate_content(
        model=model_name,
        contents=prompt,
        # No tools parameter!
    )

    part1 = response1.candidates[0].content.parts[0]
    print(f"  Response type: TEXT")
    print(f"  Content: {response1.text.strip()}")
    print()

    # -----------------------------------------------------------------
    # Call 2: WITH tools
    # -----------------------------------------------------------------
    print("[CALL 2: WITH tools=]")
    print("-" * 40)

    response2 = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=GenerateContentConfig(tools=[tools]),  # <-- THE DIFFERENCE
    )

    part2 = response2.candidates[0].content.parts[0]

    if hasattr(part2, "function_call") and part2.function_call:
        fc = part2.function_call
        print(f"  Response type: FUNCTION_CALL")
        print(f"  Function: {fc.name}")
        print(f"  Arguments: {dict(fc.args)}")
    else:
        print(f"  Response type: TEXT")
        print(f"  Content: {response2.text.strip()}")

    print()
    print("=" * 60)
    print("THE DIFFERENCE:")
    print()
    print("  WITHOUT tools=")
    print("    -> Model predicts text (may be wrong for complex math)")
    print()
    print("  WITH tools=")
    print("    -> Model can choose to return a function_call instead")
    print("    -> It's trained to recognize when tools help")
    print("    -> The schema tells it WHAT tools exist and HOW to call them")
    print()
    print("  The model was fine-tuned to:")
    print("    1. See the tool schemas in the prompt")
    print("    2. Decide if a tool would help answer the question")
    print("    3. Output structured function_call data instead of text")
    print("=" * 60)


if __name__ == "__main__":
    main()
