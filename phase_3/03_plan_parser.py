"""
03_plan_parser.py - Parsing Plans from Model Output

PURPOSE:
    Parse the model's plan output into structured data.
    Each step becomes an object we can execute.

EXPECTED INPUT:
    PLAN:
    [1] Calculate 30% of 200
        Tool: calculator("0.30 * 200")
        Expect: A number representing 30% of 200

    [2] Compare the result with $50
        Tool: calculator("60 - 50")
        Expect: Positive if enough

    END_PLAN

EXPECTED OUTPUT:
    [
        PlanStep(number=1, description="Calculate 30% of 200",
                 tool="calculator", arg="0.30 * 200",
                 expected="A number representing 30% of 200"),
        PlanStep(number=2, description="Compare the result with $50",
                 tool="calculator", arg="60 - 50",
                 expected="Positive if enough"),
    ]

RUN:
    uv run python 03_plan_parser.py
"""

import re
from dataclasses import dataclass


@dataclass
class PlanStep:
    """A single step in a plan."""

    number: int
    description: str
    tool: str
    arg: str
    expected: str

    def __repr__(self) -> str:
        return f"PlanStep({self.number}: {self.tool}({self.arg!r}))"


def parse_plan(text: str) -> list[PlanStep]:
    """
    Parse a plan from model output.

    Returns a list of PlanStep objects.
    """
    steps = []

    # Find the plan section
    plan_match = re.search(r"PLAN:(.*?)END_PLAN", text, re.DOTALL | re.IGNORECASE)
    if not plan_match:
        # Try without END_PLAN marker
        plan_match = re.search(r"PLAN:(.*?)$", text, re.DOTALL | re.IGNORECASE)

    if not plan_match:
        raise ValueError("Could not find PLAN section in text")

    plan_text = plan_match.group(1)

    # Pattern to match each step
    # [1] Description
    #     Tool: tool_name("argument")
    #     Expect: expectation
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


def main():
    print("=" * 60)
    print("PLAN PARSER DEMO")
    print("=" * 60)

    # Test cases
    test_cases = [
        # Basic plan
        """
PLAN:
[1] Calculate 30% of 200
    Tool: calculator("0.30 * 200")
    Expect: A number representing 30% of 200

[2] Compare the result with $50
    Tool: calculator("60 - 50")
    Expect: Positive if enough, negative if not

[3] Provide final answer
    Tool: finish("Based on calculations, yes you can afford it")
    Expect: Clear answer

END_PLAN
        """,
        # Multi-step with weather
        """
PLAN:
[1] Calculate 15% of $3000
    Tool: calculator("0.15 * 3000")
    Expect: The savings amount

[2] Get London weather
    Tool: get_weather("London")
    Expect: Current weather conditions

[3] Compile final answer with both pieces of information
    Tool: finish("Your 15% savings is X and London weather is Y")
    Expect: Complete answer

END_PLAN
        """,
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Input:\n{test}")

        try:
            steps = parse_plan(test)
            print(f"\nParsed {len(steps)} steps:")
            for step in steps:
                print(f"  {step}")
                print(f"    Description: {step.description}")
                print(f"    Expected: {step.expected}")
        except ValueError as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("NEXT: In 04_plan_executor.py, we'll execute plans step by step.")
    print("=" * 60)


if __name__ == "__main__":
    main()
