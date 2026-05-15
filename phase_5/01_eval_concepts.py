"""
01_eval_concepts.py - Introduction to Agent Evaluation & Guardrails

PURPOSE:
    Understand WHY evaluating agents is hard and WHAT tools exist.

KEY QUESTION:
    How do you know your agent is actually working correctly?

THE EVAL PROBLEM:

    Traditional software:
        input → function → output
        assert output == expected   ← deterministic!

    LLM agents:
        input → agent → output
        assert output ≈ expected?   ← non-deterministic!

    Challenges:
    - Same question can have many valid answers
    - Output format varies run to run
    - Agent may take different paths to same result
    - "Correct" is often subjective

TYPES OF EVALUATION:

    1. UNIT-LEVEL EVAL
       ┌──────────────┐
       │ Test single  │
       │ component    │
       └──────────────┘
       - Does the parser extract the right action?
       - Does the tool return expected results?
       - Does the router pick the right specialist?
       - Deterministic, fast, cheap

    2. END-TO-END EVAL
       ┌──────────┐     ┌──────────┐     ┌──────────┐
       │  Input   │ ──→ │  Agent   │ ──→ │  Check   │
       │ Dataset  │     │  (full)  │     │  Output  │
       └──────────┘     └──────────┘     └──────────┘
       - Run agent on a dataset of (question, expected_answer) pairs
       - Check if output matches expected answer
       - Slower, more expensive, but tests the real system

    3. LLM-AS-JUDGE
       ┌──────────┐     ┌──────────┐
       │  Agent   │ ──→ │  Judge   │ ──→ Score (1-5)
       │ Response │     │  (LLM)   │
       └──────────┘     └──────────┘
       - Use another LLM call to evaluate quality
       - Can score on multiple dimensions (accuracy, helpfulness)
       - Scalable alternative to human eval
       - But: judge can have its own biases

    4. HUMAN EVAL
       ┌──────────┐     ┌──────────┐
       │  Agent   │ ──→ │  Human   │ ──→ Score
       │ Response │     │ Reviewer │
       └──────────┘     └──────────┘
       - Gold standard for quality
       - Expensive, slow, doesn't scale
       - Use for spot-checking and calibrating LLM judges

GUARDRAILS:

    Input Guardrails (BEFORE agent runs):
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  User    │ ──→ │ Validate │ ──→ │  Agent   │
    │  Input   │     │  Input   │     │          │
    └──────────┘     └──────────┘     └──────────┘
                      │ REJECT if:
                      │ - Prompt injection
                      │ - Off-topic
                      │ - Unsafe content

    Output Guardrails (AFTER agent responds):
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Agent   │ ──→ │ Validate │ ──→ │  User    │
    │  Output  │     │  Output  │     │          │
    └──────────┘     └──────────┘     └──────────┘
                      │ BLOCK if:
                      │ - Hallucination detected
                      │ - Wrong format
                      │ - Unsafe content

WHY BOTH?
    - Input guardrails: prevent the agent from wasting time on bad requests
    - Output guardrails: catch problems the agent creates
    - Defense in depth: neither alone is sufficient

RUN:
    uv run python 01_eval_concepts.py
"""


def main() -> None:
    print("=" * 60)
    print("AGENT EVALUATION & GUARDRAILS - CONCEPT OVERVIEW")
    print("=" * 60)

    print("""
This script is conceptual - no code to run.

THE EVAL PROBLEM
================
Traditional tests use assert output == expected.
But agents are non-deterministic. Ask "What causes rain?" twice:

  Answer A: "Rain is caused by water vapor condensing in clouds."
  Answer B: "When warm moist air rises, it cools and water
             droplets form, falling as precipitation."

Both are correct! Simple string matching fails here.

TYPES OF EVALUATION
===================

  Type           | Cost    | Speed   | Best for
  ---------------+---------+---------+---------------------------
  Unit-level     | Free    | Fast    | Parsers, tools, routers
  End-to-end     | $$      | Slow    | Full agent regression tests
  LLM-as-Judge   | $       | Medium  | Scalable quality scoring
  Human eval     | $$$     | Slowest | Calibration, edge cases

GUARDRAILS
==========

  Input guardrails:  Block bad requests BEFORE the agent runs.
  Output guardrails: Block bad responses BEFORE the user sees them.
  Defense in depth:  Use BOTH for reliable systems.

PUTTING IT TOGETHER
===================

  Evaluation tells you: "Is my agent good enough?"
  Guardrails ensure:    "My agent won't misbehave in production."

  Together they give you confidence to deploy.

NEXT: In 02_llm_as_judge.py, we'll build an LLM-as-Judge
that scores answers on accuracy, relevance, and helpfulness.
""")

    print("=" * 60)
    print("Continue to 02_llm_as_judge.py when ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
