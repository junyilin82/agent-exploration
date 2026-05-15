"""
03_specialist_agents.py - Specialist/Router Pattern

PURPOSE:
    A router agent analyzes incoming requests and delegates
    to the most appropriate specialist agent.

WHY SPECIALISTS?
    - Each agent has focused expertise
    - Smaller, more precise prompts
    - Better performance on domain-specific tasks
    - Avoids "jack of all trades, master of none"

FLOW:
    User Question
         │
         ▼
    ┌─────────────┐
    │   Router    │──→ Analyzes: "What type of question is this?"
    └──────┬──────┘
           │
    ┌──────┼──────┬──────┐
    ▼      ▼      ▼      ▼
  ┌────┐┌────┐┌─────┐┌───────┐
  │Math││Code││Write││General│
  └────┘└────┘└─────┘└───────┘
           │
           ▼
    Specialist Answer

RUN:
    uv run python 03_specialist_agents.py
"""

import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================


class SpecialistType(Enum):
    """Types of specialist agents."""

    MATH = "math"
    CODE = "code"
    WRITING = "writing"
    GENERAL = "general"


@dataclass
class RouterConfig:
    """Configuration for the router system."""

    model_name: str = "gemini-2.5-flash"
    verbose: bool = True


# =============================================================================
# AGENT PROMPTS
# =============================================================================


ROUTER_PROMPT = """You are a routing assistant that classifies questions.

Analyze the user's question and decide which specialist should handle it.

Available specialists:
- MATH: For calculations, equations, statistics, mathematical reasoning
- CODE: For programming, debugging, code explanation, algorithms
- WRITING: For essays, creative writing, editing, summarization
- GENERAL: For factual questions, advice, explanations of concepts

Respond with ONLY one word: MATH, CODE, WRITING, or GENERAL

Question: {question}

Specialist:"""


MATH_SPECIALIST_PROMPT = """You are a mathematics expert.

You excel at:
- Calculations and arithmetic
- Algebra and equations
- Statistics and probability
- Mathematical reasoning
- Word problems

Provide clear, step-by-step solutions when appropriate.
Show your work so the user can follow along.

Question: {question}

Your answer:"""


CODE_SPECIALIST_PROMPT = """You are a programming expert.

You excel at:
- Writing clean, efficient code
- Debugging and fixing errors
- Explaining algorithms and data structures
- Code review and best practices
- Multiple programming languages

When providing code, include comments and explanations.
Consider edge cases and error handling.

Question: {question}

Your answer:"""


WRITING_SPECIALIST_PROMPT = """You are a writing expert.

You excel at:
- Clear and engaging prose
- Essay structure and argumentation
- Creative writing and storytelling
- Editing and improving text
- Summarization and paraphrasing

Focus on clarity, flow, and impact.
Adapt your style to the user's needs.

Question: {question}

Your answer:"""


GENERAL_SPECIALIST_PROMPT = """You are a knowledgeable general assistant.

You excel at:
- Explaining concepts clearly
- Providing factual information
- Giving balanced perspectives
- Answering diverse questions

Be helpful, accurate, and thorough.

Question: {question}

Your answer:"""


SPECIALIST_PROMPTS = {
    SpecialistType.MATH: MATH_SPECIALIST_PROMPT,
    SpecialistType.CODE: CODE_SPECIALIST_PROMPT,
    SpecialistType.WRITING: WRITING_SPECIALIST_PROMPT,
    SpecialistType.GENERAL: GENERAL_SPECIALIST_PROMPT,
}


# =============================================================================
# ROUTER SYSTEM
# =============================================================================


class RouterSystem:
    """Routes questions to specialist agents."""

    def __init__(
        self,
        client: genai.Client,
        config: RouterConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or RouterConfig()
        self.last_route: SpecialistType | None = None

    def _log(self, message: str) -> None:
        """Print if verbose mode is on."""
        if self.config.verbose:
            print(message)

    def _call_model(self, prompt: str, temperature: float = 0) -> str:
        """Make a single model call."""
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=GenerateContentConfig(temperature=temperature),
        )
        return response.text.strip() if response.text else ""

    def _route(self, question: str) -> SpecialistType:
        """Determine which specialist should handle the question."""
        prompt = ROUTER_PROMPT.format(question=question)
        response = self._call_model(prompt, temperature=0)

        # Parse the response
        response_upper = response.upper().strip()

        if "MATH" in response_upper:
            return SpecialistType.MATH
        elif "CODE" in response_upper:
            return SpecialistType.CODE
        elif "WRITING" in response_upper:
            return SpecialistType.WRITING
        else:
            return SpecialistType.GENERAL

    def _call_specialist(self, specialist: SpecialistType, question: str) -> str:
        """Call the appropriate specialist agent."""
        prompt_template = SPECIALIST_PROMPTS[specialist]
        prompt = prompt_template.format(question=question)
        return self._call_model(prompt, temperature=0.7)

    def run(self, question: str) -> str:
        """Route the question and get a specialist answer."""
        self._log(f"\n{'='*60}")
        self._log(f"Question: {question}")
        self._log(f"{'='*60}")

        # Route to specialist
        self._log("\n[ROUTER] Analyzing question...")
        specialist = self._route(question)
        self.last_route = specialist
        self._log(f"[ROUTER] Selected: {specialist.value.upper()} specialist")

        # Get specialist answer
        self._log(f"\n[{specialist.value.upper()} SPECIALIST] Answering...")
        answer = self._call_specialist(specialist, question)
        self._log(f"\n{answer}")

        return answer

    def get_last_route(self) -> SpecialistType | None:
        """Get the last routing decision."""
        return self.last_route


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return

    client = genai.Client(api_key=api_key)
    config = RouterConfig(verbose=True)
    router = RouterSystem(client, config)

    print("=" * 60)
    print("SPECIALIST ROUTER DEMO")
    print("=" * 60)

    # Test with different types of questions
    test_questions = [
        "What is 15% of 340, and then add 50 to that result?",
        "Write a Python function to reverse a string.",
        "Write a haiku about programming.",
        "What is the capital of France?",
    ]

    for question in test_questions:
        router.run(question)
        route = router.get_last_route()
        print(f"\n[Routed to: {route.value if route else 'unknown'}]")
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
