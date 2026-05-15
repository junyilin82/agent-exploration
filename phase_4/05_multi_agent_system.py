"""
05_multi_agent_system.py - Complete Multi-Agent System

PURPOSE:
    A production-style multi-agent system that combines patterns:
    - Router for task classification
    - Specialists for domain-specific work
    - Critic for quality assurance

ARCHITECTURE:
    User Request
         │
         ▼
    ┌─────────────┐
    │   Router    │──→ Classify task type
    └──────┬──────┘
           │
    ┌──────┴──────┐
    ▼             ▼
  Simple       Complex
    │             │
    ▼             ▼
  ┌────┐    ┌──────────┐
  │Spec│    │Orchestr- │
  │list│    │ator      │
  └──┬─┘    └────┬─────┘
     │           │
     │     ┌─────┼─────┐
     │     ▼     ▼     ▼
     │   ┌───┐┌───┐┌───┐
     │   │W1 ││W2 ││W3 │
     │   └─┬─┘└─┬─┘└─┬─┘
     │     └────┴────┘
     │           │
     └─────┬─────┘
           ▼
    ┌─────────────┐
    │   Critic    │──→ Quality check (optional)
    └──────┬──────┘
           │
           ▼
      Final Answer

RUN:
    uv run python 05_multi_agent_system.py
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================


class TaskComplexity(Enum):
    """Task complexity levels."""

    SIMPLE = "simple"
    COMPLEX = "complex"


class TaskDomain(Enum):
    """Task domain types."""

    MATH = "math"
    CODE = "code"
    WRITING = "writing"
    RESEARCH = "research"
    GENERAL = "general"


@dataclass
class AgentConfig:
    """Configuration for the multi-agent system."""

    model_name: str = "gemini-2.5-flash"
    enable_critic: bool = True
    verbose: bool = True


@dataclass
class AgentStats:
    """Statistics from the multi-agent run."""

    complexity: TaskComplexity | None = None
    domain: TaskDomain | None = None
    subtasks: list[str] = field(default_factory=list)
    critic_approved: bool = False
    total_calls: int = 0


# =============================================================================
# PROMPTS
# =============================================================================


CLASSIFIER_PROMPT = """Analyze this request and classify it.

1. COMPLEXITY: Is this SIMPLE (can be answered directly) or COMPLEX (needs multiple steps/subtasks)?
2. DOMAIN: What domain is this? MATH, CODE, WRITING, RESEARCH, or GENERAL?

Respond in this exact format:
COMPLEXITY: <SIMPLE or COMPLEX>
DOMAIN: <MATH, CODE, WRITING, RESEARCH, or GENERAL>

Request: {request}

Classification:"""


SPECIALIST_PROMPTS = {
    TaskDomain.MATH: """You are a mathematics expert. Solve problems step-by-step.
Question: {request}
Your solution:""",

    TaskDomain.CODE: """You are a programming expert. Write clean, well-documented code.
Request: {request}
Your code:""",

    TaskDomain.WRITING: """You are a writing expert. Create clear, engaging content.
Request: {request}
Your writing:""",

    TaskDomain.RESEARCH: """You are a research expert. Provide thorough, factual information.
Request: {request}
Your findings:""",

    TaskDomain.GENERAL: """You are a helpful assistant. Provide a clear, helpful response.
Request: {request}
Your response:""",
}


DECOMPOSE_PROMPT = """Break down this complex request into 2-4 simple subtasks.

Format:
SUBTASKS:
[1] First subtask
[2] Second subtask
END_SUBTASKS

Request: {request}

Decomposition:"""


SYNTHESIZE_PROMPT = """Combine these subtask results into one coherent answer.

Original request: {request}

Subtask results:
{results}

Synthesized answer:"""


CRITIC_PROMPT = """Review this answer for quality.

Request: {request}
Answer: {answer}

Check for:
1. Accuracy - Is it factually correct?
2. Completeness - Does it fully address the request?
3. Clarity - Is it easy to understand?

If acceptable, respond: APPROVED
If not, respond: NEEDS_REVISION: <brief explanation>

Your review:"""


# =============================================================================
# MULTI-AGENT SYSTEM
# =============================================================================


class MultiAgentSystem:
    """A complete multi-agent system with routing, specialists, and critic."""

    def __init__(
        self,
        client: genai.Client,
        config: AgentConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or AgentConfig()
        self.stats = AgentStats()

    def _log(self, message: str) -> None:
        """Print if verbose mode is on."""
        if self.config.verbose:
            print(message)

    def _call_model(self, prompt: str, temperature: float = 0) -> str:
        """Make a single model call."""
        self.stats.total_calls += 1
        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=GenerateContentConfig(temperature=temperature),
        )
        return response.text.strip() if response.text else ""

    def _classify(self, request: str) -> tuple[TaskComplexity, TaskDomain]:
        """Classify the request by complexity and domain."""
        prompt = CLASSIFIER_PROMPT.format(request=request)
        response = self._call_model(prompt)

        # Parse complexity
        complexity = TaskComplexity.SIMPLE
        if "COMPLEX" in response.upper():
            complexity = TaskComplexity.COMPLEX

        # Parse domain
        domain = TaskDomain.GENERAL
        for d in TaskDomain:
            if d.value.upper() in response.upper():
                domain = d
                break

        return complexity, domain

    def _handle_simple(self, request: str, domain: TaskDomain) -> str:
        """Handle a simple request with the appropriate specialist."""
        prompt = SPECIALIST_PROMPTS[domain].format(request=request)
        return self._call_model(prompt, temperature=0.7)

    def _handle_complex(self, request: str) -> str:
        """Handle a complex request with orchestrator pattern."""
        # Decompose
        decompose_prompt = DECOMPOSE_PROMPT.format(request=request)
        decomposition = self._call_model(decompose_prompt)

        # Parse subtasks
        subtasks = []
        pattern = r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|END_SUBTASKS|$)"
        matches = re.findall(pattern, decomposition, re.DOTALL)
        for _, desc in matches:
            subtasks.append(desc.strip())

        self.stats.subtasks = subtasks
        self._log(f"  Subtasks: {subtasks}")

        # Execute each subtask
        results = []
        for i, subtask in enumerate(subtasks, 1):
            self._log(f"  [Worker {i}] {subtask[:50]}...")
            worker_prompt = f"Complete this task:\n{subtask}\n\nYour result:"
            result = self._call_model(worker_prompt, temperature=0.7)
            results.append(f"[{i}] {subtask}\nResult: {result}")

        # Synthesize
        synthesize_prompt = SYNTHESIZE_PROMPT.format(
            request=request,
            results="\n\n".join(results),
        )
        return self._call_model(synthesize_prompt, temperature=0.7)

    def _critique(self, request: str, answer: str) -> tuple[bool, str]:
        """Have the critic review the answer."""
        prompt = CRITIC_PROMPT.format(request=request, answer=answer)
        response = self._call_model(prompt)

        approved = "APPROVED" in response.upper()
        return approved, response

    def run(self, request: str) -> str:
        """Run the multi-agent system on a request."""
        self._log(f"\n{'='*60}")
        self._log(f"Request: {request}")
        self._log(f"{'='*60}")

        # Reset stats
        self.stats = AgentStats()

        # Step 1: Classify
        self._log("\n[CLASSIFIER] Analyzing request...")
        complexity, domain = self._classify(request)
        self.stats.complexity = complexity
        self.stats.domain = domain
        self._log(f"  Complexity: {complexity.value}")
        self._log(f"  Domain: {domain.value}")

        # Step 2: Route and execute
        if complexity == TaskComplexity.SIMPLE:
            self._log(f"\n[{domain.value.upper()} SPECIALIST] Handling...")
            answer = self._handle_simple(request, domain)
        else:
            self._log("\n[ORCHESTRATOR] Breaking down task...")
            answer = self._handle_complex(request)

        self._log(f"\n[DRAFT ANSWER]\n{answer[:200]}...")

        # Step 3: Critic review (optional)
        if self.config.enable_critic:
            self._log("\n[CRITIC] Reviewing...")
            approved, critique = self._critique(request, answer)
            self.stats.critic_approved = approved

            if approved:
                self._log("  APPROVED")
            else:
                self._log(f"  Feedback: {critique[:100]}...")
                # In a full system, we might revise here

        self._log(f"\n{'='*60}")
        self._log("FINAL ANSWER:")
        self._log(f"{'='*60}")
        self._log(answer)

        return answer

    def get_stats(self) -> dict:
        """Get statistics from the last run."""
        return {
            "complexity": self.stats.complexity.value if self.stats.complexity else None,
            "domain": self.stats.domain.value if self.stats.domain else None,
            "subtasks": self.stats.subtasks,
            "critic_approved": self.stats.critic_approved,
            "total_calls": self.stats.total_calls,
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
    config = AgentConfig(enable_critic=True, verbose=True)
    system = MultiAgentSystem(client, config)

    print("=" * 60)
    print("MULTI-AGENT SYSTEM DEMO")
    print("=" * 60)

    # Test with different types of requests
    test_requests = [
        # Simple math
        "What is 25% of 180?",
        # Complex research
        # "Compare the pros and cons of electric vs hybrid cars, and give a recommendation for a city commuter.",
    ]

    for request in test_requests:
        system.run(request)
        stats = system.get_stats()
        print(f"\n[Stats: {stats['complexity']}, {stats['domain']}, "
              f"{stats['total_calls']} API calls, "
              f"critic: {'approved' if stats['critic_approved'] else 'not approved'}]")
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
