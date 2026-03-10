"""
01_basic_completion.py - Understanding the Baseline

PURPOSE:
    Before we add tools, let's understand how a basic LLM call works.
    This is the foundation everything else builds on.

WHAT HAPPENS:
    1. We send a message to Gemini
    2. Gemini processes it and returns text
    3. That's it - no tools, no loops, just a single request/response

KEY CONCEPT:
    An LLM is essentially a function: input text → output text
    Everything else (tools, agents, etc.) is built on top of this.

RUN:
    uv run python 01_basic_completion.py
"""

import os

from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()


def main() -> None:
    # Step 1: Create a client with API key
    # The API key authenticates us with Google's servers
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        print("Create a .env file with: GOOGLE_API_KEY=your-key-here")
        return

    client = genai.Client(api_key=api_key)

    # Step 2: Choose a model
    # gemini-2.0-flash is fast and good for learning
    # gemini-2.0-pro is more capable but slower
    model_name = "gemini-2.5-flash"

    # Step 3: Send a simple message
    user_message = "What is 25 * 4? Just give me the number."

    print("=" * 60)
    print("BASIC COMPLETION EXAMPLE")
    print("=" * 60)
    print(f"\n[SENDING TO GEMINI]")
    print(f"   Model: {model_name}")
    print(f"   Message: {user_message}")
    print()

    # This is the core API call - everything else builds on this
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
    )

    print(f"[RECEIVED FROM GEMINI]")
    print(f"   Response: {response.text}")
    print()

    # Let's peek under the hood at what we actually got back
    print("[UNDER THE HOOD]")
    print(f"   Response type: {type(response).__name__}")
    print(f"   Number of candidates: {len(response.candidates)}")

    # The response contains "candidates" - possible completions
    # Usually there's just one, but the API supports multiple
    candidate = response.candidates[0]
    print(f"   Finish reason: {candidate.finish_reason}")

    # The actual content is in parts (could be text, could be other things)
    print(f"   Number of parts: {len(candidate.content.parts)}")
    print(f"   Part type: {type(candidate.content.parts[0]).__name__}")

    print()
    print("=" * 60)
    print("KEY TAKEAWAY:")
    print("  This is a single request/response. The model can't 'do' anything")
    print("  except generate text. In the next script, we'll see how to give")
    print("  the model 'tools' it can ask us to call.")
    print("=" * 60)


if __name__ == "__main__":
    main()
