"""
04_orchestrator_worker.py - Orchestrator/Worker Pattern

PURPOSE:
    One orchestrator agent breaks down a complex task into subtasks,
    worker agents execute each subtask, and the orchestrator combines results.

WHY ORCHESTRATOR/WORKER?
    - Complex tasks become manageable subtasks
    - Workers can be specialized or generic
    - Orchestrator maintains the big picture
    - Natural division of planning vs execution

FLOW:
    User Request
         │
         ▼
    ┌─────────────┐
    │ Orchestrator│──→ "Break this into subtasks"
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Subtasks   │──→ [Task 1, Task 2, Task 3]
    └──────┬──────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
  ┌────┐┌────┐┌────┐
  │ W1 ││ W2 ││ W3 │──→ Execute each task
  └────┘└────┘└────┘
           │
           ▼
    ┌─────────────┐
    │ Orchestrator│──→ "Combine these results"
    └─────────────┘
           │
           ▼
      Final Answer

RUN:
    uv run python 04_orchestrator_worker.py
"""

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
class OrchestratorConfig:
    """Configuration for the orchestrator system."""

    model_name: str = "gemini-2.5-flash"
    max_subtasks: int = 5
    verbose: bool = True


@dataclass
class Subtask:
    """A subtask to be executed by a worker."""

    number: int
    description: str
    result: str = ""


# =============================================================================
# AGENT PROMPTS
# =============================================================================


DECOMPOSE_PROMPT = """You are a task decomposition expert.

Given a complex request, break it down into simple, independent subtasks.

Rules:
- Each subtask should be self-contained
- Subtasks should be in logical order
- Keep it to 2-5 subtasks (no more than {max_subtasks})
- Each subtask should be something a simple assistant can do

Format your response as:
SUBTASKS:
[1] First subtask description
[2] Second subtask description
[3] Third subtask description
END_SUBTASKS

User request: {request}

Your decomposition:"""


WORKER_PROMPT = """You are a helpful assistant completing a specific subtask.

Complete the following task thoroughly and concisely.
Provide only the result - no need to explain what you're doing.

Task: {task}

Your result:"""


SYNTHESIZE_PROMPT = """You are an expert at combining information into coherent answers.

The user asked: {original_request}

Here are the results from completing each subtask:

{subtask_results}

Synthesize these results into a single, coherent response for the user.
Make it flow naturally - don't just list the subtask results.

Your synthesized answer:"""


# =============================================================================
# ORCHESTRATOR SYSTEM
# =============================================================================


@dataclass
class OrchestratorStats:
    """Statistics from orchestrator run."""

    subtasks: list[Subtask] = field(default_factory=list)
    orchestrator_calls: int = 0
    worker_calls: int = 0


class OrchestratorSystem:
    """Orchestrates complex tasks by breaking them into subtasks."""

    def __init__(
        self,
        client: genai.Client,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or OrchestratorConfig()
        self.stats = OrchestratorStats()

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

    def _decompose(self, request: str) -> list[Subtask]:
        """Break down the request into subtasks."""
        prompt = DECOMPOSE_PROMPT.format(
            request=request,
            max_subtasks=self.config.max_subtasks,
        )

        self.stats.orchestrator_calls += 1
        response = self._call_model(prompt)

        # Parse subtasks
        subtasks = []
        pattern = r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|END_SUBTASKS|$)"
        matches = re.findall(pattern, response, re.DOTALL)

        for num_str, description in matches:
            subtasks.append(Subtask(
                number=int(num_str),
                description=description.strip(),
            ))

        return subtasks

    def _execute_subtask(self, subtask: Subtask) -> str:
        """Have a worker execute a single subtask."""
        prompt = WORKER_PROMPT.format(task=subtask.description)
        self.stats.worker_calls += 1
        return self._call_model(prompt, temperature=0.7)

    def _synthesize(self, original_request: str, subtasks: list[Subtask]) -> str:
        """Combine subtask results into a final answer."""
        subtask_results = "\n\n".join([
            f"[Subtask {s.number}] {s.description}\nResult: {s.result}"
            for s in subtasks
        ])

        prompt = SYNTHESIZE_PROMPT.format(
            original_request=original_request,
            subtask_results=subtask_results,
        )

        self.stats.orchestrator_calls += 1
        return self._call_model(prompt, temperature=0.7)

    def run(self, request: str) -> str:
        """Run the orchestrator on a complex request."""
        self._log(f"\n{'='*60}")
        self._log(f"Request: {request}")
        self._log(f"{'='*60}")

        # Reset stats
        self.stats = OrchestratorStats()

        # Decompose into subtasks
        self._log("\n[ORCHESTRATOR] Decomposing task...")
        subtasks = self._decompose(request)
        self.stats.subtasks = subtasks

        self._log(f"\n[ORCHESTRATOR] Created {len(subtasks)} subtasks:")
        for st in subtasks:
            self._log(f"  [{st.number}] {st.description}")

        # Execute each subtask
        self._log("\n[WORKERS] Executing subtasks...")
        for subtask in subtasks:
            self._log(f"\n  [Worker {subtask.number}] Working on: {subtask.description[:50]}...")
            subtask.result = self._execute_subtask(subtask)
            self._log(f"  [Worker {subtask.number}] Done: {subtask.result[:100]}...")

        # Synthesize results
        self._log("\n[ORCHESTRATOR] Synthesizing results...")
        final_answer = self._synthesize(request, subtasks)

        self._log(f"\n{'='*60}")
        self._log("FINAL ANSWER:")
        self._log(f"{'='*60}")
        self._log(final_answer)

        return final_answer

    def get_stats(self) -> dict:
        """Get statistics from the last run."""
        return {
            "subtasks": self.stats.subtasks,
            "orchestrator_calls": self.stats.orchestrator_calls,
            "worker_calls": self.stats.worker_calls,
            "total_calls": self.stats.orchestrator_calls + self.stats.worker_calls,
        }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return

    client = genai.Client(api_key=api_key)
    config = OrchestratorConfig(verbose=True)
    orchestrator = OrchestratorSystem(client, config)

    print("=" * 60)
    print("ORCHESTRATOR/WORKER DEMO")
    print("=" * 60)

    # Test with a complex request
    complex_request = """
    I'm planning a weekend trip to San Francisco.
    I need recommendations for:
    - 2-3 must-see attractions
    - A good restaurant for seafood
    - Tips for getting around the city
    """

    orchestrator.run(complex_request)

    stats = orchestrator.get_stats()
    print(f"\n[Stats: {stats['orchestrator_calls']} orchestrator calls, "
          f"{stats['worker_calls']} worker calls, "
          f"{stats['total_calls']} total]")


if __name__ == "__main__":
    main()
