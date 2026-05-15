"""
02_debate_agents.py - Debate/Critique Pattern

PURPOSE:
    Two agents improve an answer through debate:
    - Agent A: Proposes an answer
    - Agent B: Critiques and suggests improvements
    - Agent A: Revises based on critique
    - Repeat until satisfied or max rounds

WHY DEBATE?
    - Catches errors through adversarial checking
    - Reduces hallucination (critic challenges claims)
    - Produces more thorough, considered answers
    - Simulates peer review process

FLOW:
    User Question
         │
         ▼
    ┌─────────┐
    │Proposer │──→ Initial Answer
    └─────────┘
         │
         ▼
    ┌─────────┐
    │ Critic  │──→ Critique + Suggestions
    └─────────┘
         │
         ▼
    ┌─────────┐
    │Proposer │──→ Revised Answer
    └─────────┘
         │
         ▼
    (Repeat until consensus or max rounds)

RUN:
    uv run python 02_debate_agents.py
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DebateConfig:
    """Configuration for the debate system."""

    model_name: str = "gemini-2.5-flash"
    max_rounds: int = 3
    verbose: bool = True


# =============================================================================
# AGENT PROMPTS
# =============================================================================


PROPOSER_INITIAL_PROMPT = """You are a helpful assistant that provides thorough, accurate answers.

Given a question, provide a clear and comprehensive answer.

Be specific and include relevant details. If you're uncertain about something, say so.

Question: {question}

Your answer:"""


PROPOSER_REVISE_PROMPT = """You are a helpful assistant revising your answer based on feedback.

Original question: {question}

Your previous answer:
{previous_answer}

Critic's feedback:
{critique}

Please revise your answer to address the valid points in the critique.
If you disagree with any criticism, explain why.

Your revised answer:"""


CRITIC_PROMPT = """You are a critical reviewer who checks answers for accuracy and completeness.

Your job is to:
1. Identify any factual errors or unsupported claims
2. Point out missing important information
3. Suggest improvements

Be constructive but thorough. If the answer is already excellent, say "APPROVED" and briefly explain why.

Question being answered: {question}

Answer to review:
{answer}

Your critique (or "APPROVED" if the answer is excellent):"""


# =============================================================================
# DEBATE AGENTS
# =============================================================================


class DebateSystem:
    """A system where two agents debate to improve answers."""

    def __init__(
        self,
        client: genai.Client,
        config: DebateConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or DebateConfig()
        self.debate_history: list[dict] = []

    def _log(self, message: str) -> None:
        """Print if verbose mode is on."""
        if self.config.verbose:
            print(message)

    def _call_model(self, prompt: str) -> str:
        """Make a single model call."""
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=GenerateContentConfig(temperature=0.7),
        )
        return response.text.strip() if response.text else ""

    def _propose_initial(self, question: str) -> str:
        """Proposer generates initial answer."""
        prompt = PROPOSER_INITIAL_PROMPT.format(question=question)
        return self._call_model(prompt)

    def _critique(self, question: str, answer: str) -> str:
        """Critic reviews the answer."""
        prompt = CRITIC_PROMPT.format(question=question, answer=answer)
        return self._call_model(prompt)

    def _revise(self, question: str, previous_answer: str, critique: str) -> str:
        """Proposer revises based on critique."""
        prompt = PROPOSER_REVISE_PROMPT.format(
            question=question,
            previous_answer=previous_answer,
            critique=critique,
        )
        return self._call_model(prompt)

    def run(self, question: str) -> str:
        """Run the debate process."""
        self._log(f"\n{'='*60}")
        self._log(f"Question: {question}")
        self._log(f"{'='*60}")

        self.debate_history = []

        # Initial proposal
        self._log("\n[PROPOSER] Generating initial answer...")
        current_answer = self._propose_initial(question)
        self._log(f"\n{current_answer}")

        self.debate_history.append({
            "round": 0,
            "role": "proposer",
            "content": current_answer,
        })

        # Debate rounds
        for round_num in range(1, self.config.max_rounds + 1):
            self._log(f"\n{'─'*60}")
            self._log(f"[ROUND {round_num}]")

            # Critic reviews
            self._log("\n[CRITIC] Reviewing...")
            critique = self._critique(question, current_answer)
            self._log(f"\n{critique}")

            self.debate_history.append({
                "round": round_num,
                "role": "critic",
                "content": critique,
            })

            # Check if approved
            if "APPROVED" in critique.upper():
                self._log("\n[CONSENSUS REACHED]")
                break

            # Proposer revises
            self._log("\n[PROPOSER] Revising...")
            current_answer = self._revise(question, current_answer, critique)
            self._log(f"\n{current_answer}")

            self.debate_history.append({
                "round": round_num,
                "role": "proposer",
                "content": current_answer,
            })

        self._log(f"\n{'='*60}")
        self._log("FINAL ANSWER:")
        self._log(f"{'='*60}")
        self._log(current_answer)

        return current_answer

    def get_history(self) -> list[dict]:
        """Get the full debate history."""
        return self.debate_history


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return

    client = genai.Client(api_key=api_key)
    config = DebateConfig(max_rounds=2, verbose=True)
    debate = DebateSystem(client, config)

    print("=" * 60)
    print("DEBATE AGENTS DEMO")
    print("=" * 60)

    # Test with a question that might benefit from critique
    # test_questions = [
    #     "What are the pros and cons of remote work?",
    #     # "Explain how photosynthesis works.",
    # ]

    test_questions = [
        "Are AI threatening young graduates' job prospects? Given many companies are claiming to use AI to replace junior roles.",
    ]

    for question in test_questions:
        debate.run(question)

        history = debate.get_history()
        print(f"\n[Stats: {len(history)} exchanges over {config.max_rounds} max rounds]")
        print("\n")


if __name__ == "__main__":
    main()
