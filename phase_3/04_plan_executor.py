"""
04_plan_executor.py - Executing Plans Step by Step

PURPOSE:
    Take a parsed plan and execute it step by step.
    Track results and handle the final "finish" step.

THE EXECUTION LOOP:
    for step in plan:
        if step.tool == "finish":
            return compile_final_answer(step, results)
        else:
            result = execute_tool(step.tool, step.arg)
            results[step.number] = result

KEY CHALLENGE:
    Steps may reference previous results!
    Example: "calculator('result_from_step_1 - 50')"
    We need to substitute actual values.

RUN:
    uv run python 04_plan_executor.py
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# =============================================================================
# PLAN STEP (from 03_plan_parser.py)
# =============================================================================


@dataclass
class PlanStep:
    """A single step in a plan."""

    number: int
    description: str
    tool: str
    arg: str
    expected: str


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
    return mock_data.get(city.lower(), "70°F, unknown")


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
        """
        Replace references to previous results.

        E.g., "result_from_step_1" → actual value from step 1
        """
        # Pattern: result_from_step_N or step_N_result or similar
        def replace_ref(match: re.Match) -> str:
            step_num = int(match.group(1))
            if step_num in self.results:
                return self.results[step_num]
            return match.group(0)  # Keep original if not found

        # Try multiple patterns
        arg = re.sub(r"result_from_step_(\d+)", replace_ref, arg, flags=re.IGNORECASE)
        arg = re.sub(r"step_(\d+)_result", replace_ref, arg, flags=re.IGNORECASE)
        arg = re.sub(r"\{step_(\d+)\}", replace_ref, arg)

        return arg

    def execute_step(self, step: PlanStep) -> ExecutionResult:
        """Execute a single step."""
        # Substitute any references to previous results
        actual_arg = self.substitute_references(step.arg)

        if step.tool == "finish":
            # For finish, compile all results into the answer
            result = self._compile_finish(actual_arg)
            success = True
        elif step.tool not in TOOLS:
            result = f"Error: Unknown tool '{step.tool}'"
            success = False
        else:
            try:
                tool_func = TOOLS[step.tool]
                result = tool_func(actual_arg)
                success = True
            except Exception as e:
                result = f"Error: {e}"
                success = False

        # Store result
        self.results[step.number] = result

        # Log execution
        exec_result = ExecutionResult(
            step_number=step.number,
            tool=step.tool,
            arg=actual_arg,
            result=result,
            success=success,
        )
        self.execution_log.append(exec_result)

        return exec_result

    def _compile_finish(self, template: str) -> str:
        """Compile the final answer, substituting all results."""
        answer = template
        for step_num, result in self.results.items():
            answer = answer.replace(f"{{step_{step_num}}}", result)
            answer = answer.replace(f"result_from_step_{step_num}", result)
        return answer

    def execute_plan(self, plan: list[PlanStep], verbose: bool = True) -> str:
        """
        Execute all steps in a plan.

        Returns the final answer.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"EXECUTING PLAN ({len(plan)} steps)")
            print(f"{'='*60}")

        for step in plan:
            if verbose:
                print(f"\n[Step {step.number}] {step.description}")
                print(f"  Tool: {step.tool}({step.arg!r})")

            result = self.execute_step(step)

            if verbose:
                status = "✓" if result.success else "✗"
                print(f"  Result: {result.result} {status}")

            if step.tool == "finish":
                if verbose:
                    print(f"\n{'─'*60}")
                    print(f"FINAL ANSWER: {result.result}")
                    print(f"{'─'*60}")
                return result.result

            if not result.success:
                if verbose:
                    print(f"\n  ⚠ Step failed! Plan execution stopped.")
                return f"Error at step {step.number}: {result.result}"

        return "Error: Plan did not include a finish step"


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("PLAN EXECUTOR DEMO")
    print("=" * 60)

    # Create a sample plan (normally this comes from the parser)
    plan = [
        PlanStep(
            number=1,
            description="Calculate 15% of $3000",
            tool="calculator",
            arg="0.15 * 3000",
            expected="The savings amount",
        ),
        PlanStep(
            number=2,
            description="Get London weather",
            tool="get_weather",
            arg="London",
            expected="Current weather",
        ),
        PlanStep(
            number=3,
            description="Check if savings covers $400 expense",
            tool="calculator",
            arg="result_from_step_1 - 400",
            expected="Positive means can afford",
        ),
        PlanStep(
            number=4,
            description="Compile final answer",
            tool="finish",
            arg="15% of $3000 is $result_from_step_1. London weather: result_from_step_2. Can afford $400? result_from_step_3 remaining.",
            expected="Complete answer",
        ),
    ]

    # Execute the plan
    executor = PlanExecutor()
    final_answer = executor.execute_plan(plan, verbose=True)

    # Show execution summary
    print("\n[EXECUTION SUMMARY]")
    for log in executor.execution_log:
        status = "✓" if log.success else "✗"
        print(f"  Step {log.step_number}: {log.tool}({log.arg!r}) → {log.result} {status}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("  Notice how 'result_from_step_1' was substituted with '450.0'")
    print("  This allows steps to reference previous results!")
    print("\n  NEXT: In 05_planning_agent.py, we'll combine everything:")
    print("  - Generate plan from user request")
    print("  - Parse the plan")
    print("  - Execute the plan")
    print("  - Handle failures with re-planning")
    print("=" * 60)


if __name__ == "__main__":
    main()
