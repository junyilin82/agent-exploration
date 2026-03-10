"""
01b_sampling_demo.py - Understanding Sampling and Candidates

PURPOSE:
    Demonstrate that LLM outputs are sampled, not deterministic.
    Show how temperature and candidates work.

KEY CONCEPTS:
    - "candidates" = number of parallel completions to generate
    - "temperature" = randomness (0 = deterministic, higher = more random)
    - Same prompt can produce different outputs

RUN:
    uv run python 01b_sampling_demo.py
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        return

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-flash"

    # A prompt designed to show variation
    prompt = "Complete this sentence with one word: The sky is ___"

    print("=" * 60)
    print("DEMO: Multiple candidates in one request")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print()

    # Demo 1: High temperature (more random)
    print("With temperature=1.5 (high randomness), 4 candidates:")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=GenerateContentConfig(
            candidate_count=4,
            temperature=1.5,
        ),
    )
    for i, candidate in enumerate(response.candidates):
        text = candidate.content.parts[0].text.strip()
        print(f"  Candidate {i + 1}: {text}")

    print()

    # Demo 2: Temperature = 0 (deterministic)
    print("With temperature=0 (deterministic), 4 candidates:")
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=GenerateContentConfig(
            candidate_count=4,
            temperature=0.0,
        ),
    )
    for i, candidate in enumerate(response.candidates):
        text = candidate.content.parts[0].text.strip()
        print(f"  Candidate {i + 1}: {text}")

    print()
    print("=" * 60)
    print("KEY TAKEAWAYS:")
    print("  - High temperature = more variety in responses")
    print("  - Temperature 0 = all candidates are identical (greedy)")
    print("  - 'Candidates' lets you get multiple options in one API call")
    print("=" * 60)


if __name__ == "__main__":
    main()
