"""
02_react_prompt.py - Designing the ReAct Prompt

PURPOSE:
    Create a prompt that makes the model output in Thought/Action format.
    This is the core of ReAct - prompt engineering, not API features.

THE PROMPT MUST:
    1. Explain the Thought/Action/Observation format
    2. List available tools and their usage
    3. Show examples of correct behavior
    4. Define when to use "finish" action

RUN:
    uv run python 02_react_prompt.py
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()

# =============================================================================
# THE REACT PROMPT
# =============================================================================

REACT_PROMPT = '''You are a helpful assistant that solves problems step by step.

## How to Respond

You must ALWAYS respond in this exact format:

Thought: [Your reasoning about what to do next]
Action: [tool_name("argument")]

After you see the result of your action, you'll get an Observation with the result.
Then continue with another Thought/Action until you have the final answer.

When you have the final answer, use:
Thought: [Your final reasoning]
Action: finish("[Your complete answer to the user]")

## Available Tools

1. calculator(expression) - Evaluate a math expression
   Example: calculator("15 * 20 + 5")

2. get_weather(city) - Get weather for a city
   Example: get_weather("Tokyo")

3. get_time(timezone) - Get current time in a timezone
   Example: get_time("EST")

4. finish(answer) - Return the final answer to the user
   Example: finish("The answer is 42")

## Rules

- Always start with a Thought explaining your reasoning
- Only use ONE Action per response
- Wait for the Observation before your next Thought
- Use finish() only when you have the complete answer
- If a tool returns an error, acknowledge it and try a different approach

## Example

User: What is 25% of 80?

Thought: I need to calculate 25% of 80. 25% as a decimal is 0.25, so I multiply 0.25 * 80.
Action: calculator("0.25 * 80")

[Then you would see: Observation: 20.0]

Thought: The calculation shows 25% of 80 is 20. I have the answer.
Action: finish("25% of 80 is 20")
'''


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    print("=" * 60)
    print("REACT PROMPT DEMO")
    print("=" * 60)

    print("\n[THE PROMPT]")
    print("-" * 40)
    print(REACT_PROMPT)

    print("\n[TESTING THE PROMPT]")
    print("-" * 40)

    # Test with a simple question
    user_query = "What is 15% of 200?"

    # Combine system prompt with user query
    full_prompt = f"{REACT_PROMPT}\n\nUser: {user_query}"

    print(f"User: {user_query}\n")
    print("Model response:")

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=GenerateContentConfig(temperature=0),  # Deterministic for testing
    )

    print(response.text)

    print("\n" + "=" * 60)
    print("OBSERVATION:")
    print("  The model outputs 'Thought:' and 'Action:' as text.")
    print("  We need to PARSE this text to extract the action.")
    print("  That's what we'll build in 03_react_parser.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
