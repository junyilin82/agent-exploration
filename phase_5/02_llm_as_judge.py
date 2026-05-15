"""
02_llm_as_judge.py - LLM-as-Judge Evaluation

PURPOSE:
    Use one LLM call to generate an answer, then another LLM call
    to score it on multiple quality dimensions.

WHY LLM-AS-JUDGE?
    - Human eval is expensive and slow
    - Simple string matching misses valid alternative answers
    - LLMs can assess nuanced qualities like helpfulness and clarity
    - Scalable: can evaluate thousands of responses automatically

FLOW:
    Question
         │
         ▼
    ┌─────────────┐
    │  Generator  │──→ Generate answer
    │   (LLM)     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Judge     │──→ Score on dimensions:
    │   (LLM)     │    - Accuracy (1-5)
    │             │    - Relevance (1-5)
    │             │    - Helpfulness (1-5)
    └──────┬──────┘
           │
           ▼
    Structured Scores + Reasoning

RUN:
    uv run python 02_llm_as_judge.py
"""

import json
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class JudgeConfig:
    """Configuration for the LLM-as-Judge system."""

    model_name: str = "gemini-2.5-flash"
    verbose: bool = True


@dataclass
class EvalScore:
    """Evaluation scores from the judge."""

    accuracy: int
    relevance: int
    helpfulness: int
    reasoning: str

    @property
    def average(self) -> float:
        """Average score across all dimensions."""
        return (self.accuracy + self.relevance + self.helpfulness) / 3

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy}/5 | Relevance: {self.relevance}/5 | "
            f"Helpfulness: {self.helpfulness}/5 | Avg: {self.average:.1f}/5\n"
            f"Reasoning: {self.reasoning}"
        )


# =============================================================================
# PROMPTS
# =============================================================================


GENERATOR_PROMPT = """You are a helpful assistant. Answer the following question clearly and concisely.

Question: {question}

Your answer:"""


JUDGE_PROMPT = """You are an expert evaluator. Score the following answer on three dimensions.

Question: {question}
Answer: {answer}
Reference answer: {reference}

Score each dimension from 1 (worst) to 5 (best):

1. ACCURACY: Does the answer contain correct information? Does it match the reference?
   1 = Completely wrong
   3 = Partially correct
   5 = Fully accurate

2. RELEVANCE: Does the answer address the question asked?
   1 = Completely off-topic
   3 = Partially relevant
   5 = Directly addresses the question

3. HELPFULNESS: Is the answer clear, well-structured, and useful?
   1 = Confusing and unhelpful
   3 = Adequate
   5 = Exceptionally clear and useful

Respond in this exact JSON format:
{{"accuracy": <score>, "relevance": <score>, "helpfulness": <score>, "reasoning": "<brief explanation>"}}

Your evaluation:"""


JUDGE_PROMPT_NO_REFERENCE = """You are an expert evaluator. Score the following answer on three dimensions.

Question: {question}
Answer: {answer}

Score each dimension from 1 (worst) to 5 (best):

1. ACCURACY: Does the answer appear factually correct?
   1 = Likely wrong  3 = Uncertain  5 = Clearly correct

2. RELEVANCE: Does the answer address the question asked?
   1 = Off-topic  3 = Partially relevant  5 = Directly addresses question

3. HELPFULNESS: Is the answer clear and useful?
   1 = Confusing  3 = Adequate  5 = Exceptionally clear

Respond in this exact JSON format:
{{"accuracy": <score>, "relevance": <score>, "helpfulness": <score>, "reasoning": "<brief explanation>"}}

Your evaluation:"""


# =============================================================================
# LLM JUDGE
# =============================================================================


class LLMJudge:
    """Uses an LLM to evaluate the quality of generated answers."""

    def __init__(self, client: genai.Client, config: JudgeConfig | None = None) -> None:
        self.client = client
        self.config = config or JudgeConfig()

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

    def _parse_scores(self, response: str) -> EvalScore:
        """Parse JSON scores from judge response."""
        json_match = re.search(r"\{.*?\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not find JSON in judge response: {response}")

        data = json.loads(json_match.group())
        return EvalScore(
            accuracy=int(data.get("accuracy", 0)),
            relevance=int(data.get("relevance", 0)),
            helpfulness=int(data.get("helpfulness", 0)),
            reasoning=data.get("reasoning", "No reasoning provided"),
        )

    def generate(self, question: str) -> str:
        """Generate an answer to a question."""
        prompt = GENERATOR_PROMPT.format(question=question)
        return self._call_model(prompt, temperature=0.7)

    def judge(self, question: str, answer: str, reference: str | None = None) -> EvalScore:
        """Judge an answer's quality."""
        if reference:
            prompt = JUDGE_PROMPT.format(question=question, answer=answer, reference=reference)
        else:
            prompt = JUDGE_PROMPT_NO_REFERENCE.format(question=question, answer=answer)

        response = self._call_model(prompt, temperature=0)
        return self._parse_scores(response)

    def generate_and_judge(
        self, question: str, reference: str | None = None
    ) -> tuple[str, EvalScore]:
        """Generate an answer and then judge it."""
        self._log(f"\n{'='*60}")
        self._log(f"Question: {question}")
        self._log(f"{'='*60}")

        # Generate
        self._log("\n[GENERATOR] Producing answer...")
        answer = self.generate(question)
        self._log(f"\n{answer}")

        # Judge
        self._log("\n[JUDGE] Evaluating...")
        score = self.judge(question, answer, reference)
        self._log(f"\n{score}")

        return answer, score


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return

    client = genai.Client(api_key=api_key)
    config = JudgeConfig(verbose=True)
    judge = LLMJudge(client, config)

    print("=" * 60)
    print("LLM-AS-JUDGE DEMO")
    print("=" * 60)

    # Demo 1: Generate and judge with reference
    print("\n--- Demo 1: Judge with reference answer ---")
    test_cases = [
        {
            "question": "What is the capital of France?",
            "reference": "The capital of France is Paris.",
        },
        {
            "question": "Explain photosynthesis in one sentence.",
            "reference": (
                "Photosynthesis is the process by which plants convert sunlight, "
                "water, and carbon dioxide into glucose and oxygen."
            ),
        },
    ]

    for case in test_cases:
        judge.generate_and_judge(case["question"], case["reference"])

    # Demo 2: Judge a deliberately bad answer
    print("\n--- Demo 2: Judge a bad answer ---")
    bad_answer = "Photosynthesis is when plants eat dirt to make energy."
    reference = (
        "Photosynthesis is the process by which plants convert sunlight, "
        "water, and carbon dioxide into glucose and oxygen."
    )
    print(f"\nQuestion: Explain photosynthesis in one sentence.")
    print(f"Bad answer: {bad_answer}")
    score = judge.judge(
        "Explain photosynthesis in one sentence.",
        bad_answer,
        reference,
    )
    print(f"\nScores: {score}")

    # Demo 3: Judge without reference (open-ended)
    print("\n--- Demo 3: Judge without reference (open-ended) ---")
    judge.generate_and_judge("What are three tips for learning a new programming language?")


if __name__ == "__main__":
    main()
