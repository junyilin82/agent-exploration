"""
Planning Agent - Plan first, then execute.

The Planning pattern creates a full plan before executing,
unlike ReAct which decides step by step.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

from google import genai
from google.genai.types import GenerateContentConfig

from planning_agent.tools import Tool, DEFAULT_TOOLS


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for the Planning agent."""

    model_name: str = "gemini-2.5-flash"
    max_replan_attempts: int = 2
    verbose: bool = True


# =============================================================================
# PROMPT BUILDER
# =============================================================================


def build_planning_prompt(tools: list[Tool], extra_instructions: str = "") -> str:
    """Build the planning prompt dynamically from the tool list."""
    tool_descriptions = []
    for i, tool in enumerate(tools, 1):
        tool_descriptions.append(
            f"{i}. {tool.name}(argument) - {tool.description}\n"
            f"   Example: {tool.example}"
        )

    tools_section = "\n\n".join(tool_descriptions)

    extra_section = ""
    if extra_instructions:
        extra_section = f"\n\n## Additional Instructions\n\n{extra_instructions}"

    return f"""You are a planning assistant that creates step-by-step plans.

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
    Tool: finish("<answer template with {{step_X}} placeholders>")
    Expect: Complete answer to user

END_PLAN

## Available Tools

{tools_section}

{len(tools) + 1}. finish(answer) - Compile and return the final answer
   Use {{step_N}} to reference result from step N
   Example: finish("The answer is {{step_1}} and the weather is {{step_2}}")

## Rules

- Create a complete plan before any execution
- Each step must have exactly one Tool
- Steps should be in logical order
- Use {{step_N}} in later steps to reference results from step N
- The LAST step must use finish() to compile the final answer
- Keep plans concise (typically 2-5 steps)
{extra_section}
Now create a plan for:

"""


PLANNING_PROMPT = build_planning_prompt(DEFAULT_TOOLS)


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

    plan_match = re.search(r"PLAN:(.*?)END_PLAN", text, re.DOTALL | re.IGNORECASE)
    if not plan_match:
        plan_match = re.search(r"PLAN:(.*?)$", text, re.DOTALL | re.IGNORECASE)

    if not plan_match:
        raise ValueError("Could not find PLAN section in text")

    plan_text = plan_match.group(1)

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

    tools: dict[str, Callable] = field(default_factory=dict)
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
            result = actual_arg
            success = True
        elif step.tool not in self.tools:
            result = f"Error: Unknown tool '{step.tool}'"
            success = False
        else:
            try:
                result = self.tools[step.tool](actual_arg)
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

    def execute_plan(self, plan: list[PlanStep]) -> tuple[str, bool]:
        """Execute all steps in a plan. Returns (final_answer, success)."""
        for step in plan:
            result = self.execute_step(step)

            if step.tool == "finish":
                return result.result, True

            if not result.success:
                return f"Failed at step {step.number}: {result.result}", False

        return "Error: Plan did not include a finish step", False


# =============================================================================
# AGENT STATS
# =============================================================================


@dataclass
class PlanAttempt:
    """A single plan attempt with its execution results."""

    plan_steps: list[PlanStep]
    execution_log: list[ExecutionResult]
    success: bool


@dataclass
class AgentStats:
    """Statistics from agent run."""

    planning_calls: int = 0
    execution_steps: int = 0
    replan_count: int = 0
    attempts: list[PlanAttempt] = field(default_factory=list)


# =============================================================================
# THE AGENT
# =============================================================================


class PlanningAgent:
    """
    A planning agent that:
    1. Generates a full plan
    2. Executes the plan
    3. Re-plans if needed
    """

    def __init__(
        self,
        client: genai.Client,
        config: AgentConfig | None = None,
        tools: list[Tool] | None = None,
        extra_instructions: str = "",
    ) -> None:
        self.client = client
        self.config = config or AgentConfig()
        self.tools = tools or DEFAULT_TOOLS
        self.planning_prompt = build_planning_prompt(self.tools, extra_instructions)

        # Build tool lookup
        self._tool_funcs: dict[str, Callable] = {t.name: t.func for t in self.tools}

        # Stats for current/last run
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

        return f"""{self.planning_prompt}

IMPORTANT: A previous plan failed. Here's what happened:

Original request: {original_request}

{completed_info}

Failed at step {failed_step.step_number}:
  Tool: {failed_step.tool}("{failed_step.arg}")
  Error: {failed_step.result}

Please create a NEW plan that avoids this error.

Original request: {original_request}
"""

    def run(self, user_request: str) -> str:
        """Run the planning agent on a user request."""
        self._log(f"\n{'='*60}")
        self._log(f"User: {user_request}")
        self._log(f"{'='*60}")

        # Reset stats
        self.stats = AgentStats()

        current_prompt = self.planning_prompt + user_request

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
                    current_prompt = f"""{self.planning_prompt}

IMPORTANT: Your previous plan could not be parsed. Error: {e}
Please create a valid plan following the EXACT format specified.

Original request: {user_request}
"""
                    continue
                return f"Error: Could not parse plan - {e}"

            # Execute plan
            executor = PlanExecutor(tools=self._tool_funcs)
            final_answer, success = executor.execute_plan(plan)

            self.stats.execution_steps += len(executor.execution_log)

            # Record this attempt
            self.stats.attempts.append(PlanAttempt(
                plan_steps=plan,
                execution_log=executor.execution_log,
                success=success,
            ))

            if success:
                self._log(f"\n{'─'*60}")
                self._log(f"FINAL ANSWER: {final_answer}")
                self._log(f"{'─'*60}")
                return final_answer

            self._log(f"\n[EXECUTION FAILED] {final_answer}")

            if attempt < self.config.max_replan_attempts:
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
            "attempts": self.stats.attempts,
        }
