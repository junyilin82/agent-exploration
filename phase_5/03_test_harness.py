"""
03_test_harness.py - Agent Test Harness

PURPOSE:
    Run an agent against a dataset of test cases and evaluate performance.
    Combines a simple Q&A agent with LLM-as-Judge scoring.

WHY A TEST HARNESS?
    - Systematic evaluation across many test cases
    - Reproducible results (same dataset, same metrics)
    - Catch regressions when you change your agent
    - Quantify quality with aggregate scores

FLOW:
    Test Dataset
    ┌──────────────────────────────────┐
    │ Q1: "What is 2+2?"  A1: "4"    │
    │ Q2: "Capital of UK?" A2: "London│
    │ Q3: ...              A3: ...    │
    └──────────┬───────────────────────┘
               │
               ▼ For each test case:
        ┌─────────────┐
        │   Agent     │──→ Generate answer
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │   Judge     │──→ Score answer
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │  Results    │──→ Pass/Fail + Scores
        └─────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │         Summary Table            │
    │  Total: 6  Pass: 5  Fail: 1     │
    │  Avg Accuracy: 4.2/5            │
    └──────────────────────────────────┘

RUN:
    uv run python 03_test_harness.py
"""

import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class HarnessConfig:
    """Configuration for the test harness."""

    model_name: str = "gemini-2.5-flash"
    pass_threshold: float = 3.5  # average score >= this to pass
    verbose: bool = True


@dataclass
class TestCase:
    """A single test case: question + expected answer."""

    question: str
    expected_answer: str
    category: str = "general"


@dataclass
class EvalScore:
    """Evaluation scores from the judge."""

    accuracy: int
    relevance: int
    helpfulness: int
    reasoning: str

    @property
    def average(self) -> float:
        return (self.accuracy + self.relevance + self.helpfulness) / 3


@dataclass
class TestResult:
    """Result for a single test case."""

    test_case: TestCase
    agent_answer: str
    score: EvalScore
    passed: bool


@dataclass
class TestSummary:
    """Aggregate summary of test results."""

    total: int
    passed: int
    failed: int
    avg_accuracy: float
    avg_relevance: float
    avg_helpfulness: float
    avg_overall: float
    results: list[TestResult] = field(default_factory=list)


# =============================================================================
# PROMPTS
# =============================================================================


AGENT_PROMPT = """You are a helpful assistant. Answer the following question clearly and concisely.

Question: {question}

Your answer:"""


JUDGE_PROMPT = """You are an expert evaluator. Score the following answer on three dimensions.

Question: {question}
Agent's Answer: {answer}
Reference Answer: {reference}

Score each dimension from 1 (worst) to 5 (best):
1. ACCURACY: Does the agent's answer match the reference in factual content?
2. RELEVANCE: Does the answer address the question?
3. HELPFULNESS: Is the answer clear and useful?

Respond in this exact JSON format:
{{"accuracy": <score>, "relevance": <score>, "helpfulness": <score>, "reasoning": "<brief explanation>"}}

Your evaluation:"""


# =============================================================================
# TEST HARNESS
# =============================================================================


class TestHarness:
    """Runs an agent against test cases and evaluates with LLM-as-Judge."""

    def __init__(self, client: genai.Client, config: HarnessConfig | None = None) -> None:
        self.client = client
        self.config = config or HarnessConfig()

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

    def _run_agent(self, question: str) -> str:
        """Run the agent on a single question."""
        prompt = AGENT_PROMPT.format(question=question)
        return self._call_model(prompt, temperature=0.7)

    def _judge(self, question: str, answer: str, reference: str) -> EvalScore:
        """Judge an answer against a reference."""
        prompt = JUDGE_PROMPT.format(question=question, answer=answer, reference=reference)
        response = self._call_model(prompt, temperature=0)

        json_match = re.search(r"\{.*?\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"Could not parse judge response: {response}")

        data = json.loads(json_match.group())
        return EvalScore(
            accuracy=int(data.get("accuracy", 0)),
            relevance=int(data.get("relevance", 0)),
            helpfulness=int(data.get("helpfulness", 0)),
            reasoning=data.get("reasoning", ""),
        )

    def run_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        self._log(f"\n  Question: {test_case.question}")

        answer = self._run_agent(test_case.question)
        self._log(f"  Agent: {answer[:100]}...")

        score = self._judge(test_case.question, answer, test_case.expected_answer)
        passed = score.average >= self.config.pass_threshold

        status = "PASS" if passed else "FAIL"
        self._log(
            f"  Score: Acc={score.accuracy} Rel={score.relevance} Help={score.helpfulness} "
            f"Avg={score.average:.1f} {status}"
        )

        return TestResult(test_case=test_case, agent_answer=answer, score=score, passed=passed)

    def run_suite(self, test_cases: list[TestCase]) -> TestSummary:
        """Run all test cases and compute summary."""
        self._log(f"\n{'='*60}")
        self._log(f"Running {len(test_cases)} test cases...")
        self._log(f"Pass threshold: {self.config.pass_threshold}/5 average")
        self._log(f"{'='*60}")

        results = []
        for i, tc in enumerate(test_cases, 1):
            self._log(f"\n[Test {i}/{len(test_cases)}] ({tc.category})")
            result = self.run_test(tc)
            results.append(result)

        # Compute summary
        passed = sum(1 for r in results if r.passed)
        avg_acc = sum(r.score.accuracy for r in results) / len(results)
        avg_rel = sum(r.score.relevance for r in results) / len(results)
        avg_help = sum(r.score.helpfulness for r in results) / len(results)
        avg_overall = sum(r.score.average for r in results) / len(results)

        summary = TestSummary(
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            avg_accuracy=avg_acc,
            avg_relevance=avg_rel,
            avg_helpfulness=avg_help,
            avg_overall=avg_overall,
            results=results,
        )

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: TestSummary) -> None:
        """Print a formatted summary table."""
        print(f"\n{'='*80}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        print(
            f" {'#':>2} | {'Category':<12} | {'Question':<30} | "
            f"{'Acc':>3} | {'Rel':>3} | {'Help':>4} | {'Avg':>4} | Result"
        )
        print(f"{'-'*2}-+-{'-'*12}-+-{'-'*30}-+-{'-'*3}-+-{'-'*3}-+-{'-'*4}-+-{'-'*4}-+-------")

        for i, r in enumerate(summary.results, 1):
            q = r.test_case.question[:30].ljust(30)
            status = "PASS" if r.passed else "FAIL"
            print(
                f" {i:>2} | {r.test_case.category:<12} | {q} | "
                f" {r.score.accuracy:>2} |  {r.score.relevance:>2} |   {r.score.helpfulness:>2} | "
                f"{r.score.average:>4.1f} | {status}"
            )

        print(f"{'-'*2}-+-{'-'*12}-+-{'-'*30}-+-{'-'*3}-+-{'-'*3}-+-{'-'*4}-+-{'-'*4}-+-------")
        print(
            f" TOTAL: {summary.total} tests | "
            f"{summary.passed} passed | {summary.failed} failed | "
            f"Avg: {summary.avg_overall:.1f}/5"
        )
        print(f"{'='*80}")


# =============================================================================
# MAIN
# =============================================================================


TEST_CASES = [
    TestCase("What is the capital of France?", "Paris", "geography"),
    TestCase("What is 15% of 200?", "30", "math"),
    TestCase("Who wrote Romeo and Juliet?", "William Shakespeare", "literature"),
    TestCase("What is the chemical formula for water?", "H2O", "science"),
    TestCase("What programming language is known for data science?", "Python", "technology"),
    TestCase("How many continents are there?", "7", "geography"),
]


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return

    client = genai.Client(api_key=api_key)
    config = HarnessConfig(verbose=True)
    harness = TestHarness(client, config)

    print("=" * 60)
    print("AGENT TEST HARNESS DEMO")
    print("=" * 60)

    harness.run_suite(TEST_CASES)


if __name__ == "__main__":
    main()
