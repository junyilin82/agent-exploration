"""
05_planning_agent.py - Complete Planning Agent

PURPOSE:
    A full planning agent that:
    1. Takes a user request
    2. Generates a plan
    3. Parses the plan
    4. Executes the plan step by step
    5. Returns the final answer

OPTIONAL ENHANCEMENT:
    Re-planning when a step fails.

RUN:
    uv run python 05_planning_agent.py
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for the planning agent."""

    model_name: str = "gemini-2.5-flash"
    max_replan_attempts: int = 2
    verbose: bool = True


# =============================================================================
# TOOLS
# =============================================================================


def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Get weather for a city (mock)."""
    mock_data = {
        "new york": "72°F, sunny",
        "london": "55°F, cloudy",
        "tokyo": "68°F, clear",
        "paris": "63°F, rainy",
    }
    result = mock_data.get(city.lower())
    if result is None:
        return f"Error: Unknown city '{city}'. Available cities: New York, London, Tokyo, Paris"
    return result


def get_time(timezone: str) -> str:
    """Get current time (mock)."""
    mock_times = {
        "utc": "14:30 UTC",
        "est": "09:30 EST",
        "pst": "06:30 PST",
        "jst": "23:30 JST",
    }
    return mock_times.get(timezone.lower(), "12:00 unknown")


def search(query: str) -> str:
    """Search for information (mock)."""
    return f"Search results for '{query}': [mock results]"


TOOLS: dict[str, Callable] = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_time": get_time,
    "search": search,
}


# =============================================================================
# PLANNING PROMPT
# =============================================================================


PLANNING_PROMPT = '''You are a planning assistant that creates step-by-step plans.

## Your Task

Given a user request, create a detailed plan to accomplish it.
Do NOT execute the plan - only create it.

## Plan Format

You must output your plan in this EXACT format:

PLAN:
[1] <description of step 1>
    Tool: <tool_name>("argument")
    Expect: <what result you expect>

[2] <description of step 2>
    Tool: <tool_name>("argument")
    Expect: <what result you expect>

... continue for all steps ...

[N] Compile final answer
    Tool: finish("<answer template with {step_X} placeholders>")
    Expect: Complete answer to user

END_PLAN

## Available Tools

1. calculator(expression) - Evaluate a math expression
   Example: calculator("15 * 20 + 5")

2. get_weather(city) - Get weather for a city
   Example: get_weather("Tokyo")

3. get_time(timezone) - Get current time in a timezone (UTC, EST, PST, JST)
   Example: get_time("EST")

4. search(query) - Search for information
   Example: search("best restaurants in Paris")

5. finish(answer) - Compile and return the final answer
   Use {step_N} to reference result from step N
   Example: finish("The answer is {step_1} and the weather is {step_2}")

## Rules

- Create a complete plan before any execution
- Each step must have exactly one Tool
- Steps should be in logical order
- Use {step_N} in later steps to reference results from step N
- The LAST step must use finish() to compile the final answer
- Keep plans concise (typically 2-5 steps)

## Example

User: What is 25% of 80, and is that enough for a $15 lunch?

PLAN:
[1] Calculate 25% of 80
    Tool: calculator("0.25 * 80")
    Expect: The amount (should be 20)

[2] Calculate how much is left after $15 lunch
    Tool: calculator("{step_1} - 15")
    Expect: Remaining amount

[3] Compile final answer
    Tool: finish("25% of 80 is ${step_1}. After a $15 lunch, you have ${step_2} left. So yes, it's enough!")
    Expect: Clear answer with all details

END_PLAN

Now create a plan for:

'''


# =============================================================================
# PLAN PARSER
# =============================================================================


@dataclass
class PlanStep:
    """A single step in a plan."""

    number: int
    description: str
    tool: str
    arg: str
    expected: str


def parse_plan(text: str) -> list[PlanStep]:
    """Parse a plan from model output."""
    steps = []

    # Find the plan section
    plan_match = re.search(r"PLAN:(.*?)END_PLAN", text, re.DOTALL | re.IGNORECASE)
    if not plan_match:
        plan_match = re.search(r"PLAN:(.*?)$", text, re.DOTALL | re.IGNORECASE)

    if not plan_match:
        raise ValueError("Could not find PLAN section in text")

    plan_text = plan_match.group(1)

    # Pattern to match each step
    step_pattern = re.compile(
        r"\[(\d+)\]\s*(.+?)\n\s*Tool:\s*(\w+)\s*\(\s*[\"']?(.+?)[\"']?\s*\)\s*\n\s*Expect:\s*(.+?)(?=\n\s*\[\d+\]|\n\s*END_PLAN|$)",
        re.DOTALL | re.IGNORECASE,
    )

    for match in step_pattern.finditer(plan_text):
        step = PlanStep(
            number=int(match.group(1)),
            description=match.group(2).strip(),
            tool=match.group(3).strip(),
            arg=match.group(4).strip().strip("\"'"),
            expected=match.group(5).strip(),
        )
        steps.append(step)

    if not steps:
        raise ValueError("Could not parse any steps from plan")

    return steps


# =============================================================================
# PLAN EXECUTOR
# =============================================================================


@dataclass
class ExecutionResult:
    """Result of executing a step."""

    step_number: int
    tool: str
    arg: str
    result: str
    success: bool


@dataclass
class PlanExecutor:
    """Executes a plan step by step."""

    results: dict[int, str] = field(default_factory=dict)
    execution_log: list[ExecutionResult] = field(default_factory=list)

    def substitute_references(self, arg: str) -> str:
        """Replace {step_N} references with actual values."""

        def replace_ref(match: re.Match) -> str:
            step_num = int(match.group(1))
            if step_num in self.results:
                return self.results[step_num]
            return match.group(0)

        arg = re.sub(r"\{step_(\d+)\}", replace_ref, arg)
        arg = re.sub(r"result_from_step_(\d+)", replace_ref, arg, flags=re.IGNORECASE)

        return arg

    def execute_step(self, step: PlanStep) -> ExecutionResult:
        """Execute a single step."""
        actual_arg = self.substitute_references(step.arg)

        if step.tool == "finish":
            result = actual_arg  # finish just returns the substituted template
            success = True
        elif step.tool not in TOOLS:
            result = f"Error: Unknown tool '{step.tool}'"
            success = False
        else:
            try:
                result = TOOLS[step.tool](actual_arg)
                success = "Error" not in result
            except Exception as e:
                result = f"Error: {e}"
                success = False

        self.results[step.number] = result

        exec_result = ExecutionResult(
            step_number=step.number,
            tool=step.tool,
            arg=actual_arg,
            result=result,
            success=success,
        )
        self.execution_log.append(exec_result)

        return exec_result

    def execute_plan(self, plan: list[PlanStep], verbose: bool = True) -> tuple[str, bool]:
        """
        Execute all steps in a plan.

        Returns (final_answer, success).
        """
        if verbose:
            print(f"\n[EXECUTING PLAN] ({len(plan)} steps)")

        for step in plan:
            if verbose:
                print(f"\n  Step {step.number}: {step.description}")
                print(f"    Tool: {step.tool}({step.arg!r})")

            result = self.execute_step(step)

            if verbose:
                status = "✓" if result.success else "✗"
                print(f"    Result: {result.result} {status}")

            if step.tool == "finish":
                return result.result, True

            if not result.success:
                return f"Failed at step {step.number}: {result.result}", False

        return "Error: Plan did not include a finish step", False


# =============================================================================
# PLANNING AGENT
# =============================================================================


@dataclass
class AgentStats:
    """Statistics from agent run."""

    planning_calls: int = 0
    execution_steps: int = 0
    replan_count: int = 0


class PlanningAgent:
    """
    A planning agent that:
    1. Generates a plan
    2. Executes the plan
    3. Re-plans if needed
    """

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

    def _generate_plan(self, prompt: str) -> str:
        """Generate a plan from a prompt."""
        self.stats.planning_calls += 1

        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=GenerateContentConfig(temperature=0),
        )

        return response.text.strip() if response.text else ""

    def _build_replan_prompt(
        self,
        original_request: str,
        failed_step: ExecutionResult,
        completed_results: dict[int, str],
    ) -> str:
        """Build a prompt for re-planning after a failure."""
        completed_info = ""
        if completed_results:
            completed_lines = [
                f"  Step {num}: {result}"
                for num, result in completed_results.items()
                if num < failed_step.step_number
            ]
            if completed_lines:
                completed_info = "Completed steps:\n" + "\n".join(completed_lines)

        return f"""{PLANNING_PROMPT}

IMPORTANT: A previous plan failed. Here's what happened:

Original request: {original_request}

{completed_info}

Failed at step {failed_step.step_number}:
  Tool: {failed_step.tool}("{failed_step.arg}")
  Error: {failed_step.result}

Please create a NEW plan that avoids this error. You may need to:
- Use a different approach
- Handle the error case
- Ask for clarification in your finish() response

Original request: {original_request}
"""

    def run(self, user_request: str) -> str:
        """
        Run the planning agent on a user request.

        Returns the final answer.
        """
        self._log(f"\n{'='*60}")
        self._log(f"User: {user_request}")
        self._log(f"{'='*60}")

        # Reset stats
        self.stats = AgentStats()

        # Track context for potential re-planning
        current_prompt = PLANNING_PROMPT + user_request

        for attempt in range(self.config.max_replan_attempts + 1):
            if attempt > 0:
                self._log(f"\n[RE-PLANNING] Attempt {attempt + 1}")
                self.stats.replan_count += 1

            # Generate plan
            self._log("\n[GENERATING PLAN]")
            plan_text = self._generate_plan(current_prompt)
            self._log(f"\n{plan_text}")

            # Parse plan
            try:
                plan = parse_plan(plan_text)
                self._log(f"\n[PARSED] {len(plan)} steps")
            except ValueError as e:
                self._log(f"\n[ERROR] Could not parse plan: {e}")
                if attempt < self.config.max_replan_attempts:
                    # Try re-planning with parse error context
                    current_prompt = f"""{PLANNING_PROMPT}

IMPORTANT: Your previous plan could not be parsed. Error: {e}

Please create a valid plan following the EXACT format specified.

Original request: {user_request}
"""
                    continue
                return f"Error: Could not parse plan - {e}"

            # Execute plan
            executor = PlanExecutor()
            final_answer, success = executor.execute_plan(plan, verbose=self.config.verbose)

            self.stats.execution_steps += len(executor.execution_log)

            if success:
                self._log(f"\n{'─'*60}")
                self._log(f"FINAL ANSWER: {final_answer}")
                self._log(f"{'─'*60}")
                return final_answer

            # Execution failed - try re-planning
            self._log(f"\n[EXECUTION FAILED] {final_answer}")

            if attempt < self.config.max_replan_attempts:
                # Get the failed step from execution log
                failed_step = executor.execution_log[-1]
                current_prompt = self._build_replan_prompt(
                    original_request=user_request,
                    failed_step=failed_step,
                    completed_results=executor.results,
                )
            else:
                return final_answer

        return "Error: Max re-plan attempts reached"

    def get_stats(self) -> dict:
        """Get statistics from the last run."""
        return {
            "planning_calls": self.stats.planning_calls,
            "execution_steps": self.stats.execution_steps,
            "replan_count": self.stats.replan_count,
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
    config = AgentConfig(verbose=True)
    agent = PlanningAgent(client, config)

    print("=" * 60)
    print("PLANNING AGENT DEMO")
    print("=" * 60)

    # Test queries
    test_queries = [
        "What is 20% of 500, and what's the weather in Tokyo?",
        # This one will trigger re-planning (unknown city)
        "What is 10% of 200, and what's the weather in Mumbai?",
    ]

    for query in test_queries:
        agent.run(query)
        stats = agent.get_stats()
        print(f"\n[Stats: {stats['planning_calls']} plan(s), {stats['execution_steps']} steps executed]")
        print("\n")

    # Interactive mode
    print("=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            agent.run(user_input)
            stats = agent.get_stats()
            print(f"\n[Stats: {stats['planning_calls']} plan(s), {stats['execution_steps']} steps executed]")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
