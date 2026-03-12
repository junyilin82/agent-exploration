"""
03_react_parser.py - Parsing ReAct Output

PURPOSE:
    Parse the model's text output to extract Thought and Action.
    This is needed because ReAct uses text formatting, not structured API responses.

PARSING CHALLENGE:
    Model output: "Thought: I need to calculate...\nAction: calculator("0.15 * 200")"
    We need to extract:
        - thought: "I need to calculate..."
        - action_name: "calculator"
        - action_arg: "0.15 * 200"

RUN:
    uv run python 03_react_parser.py
"""

import re
from dataclasses import dataclass


@dataclass
class ParsedAction:
    """Parsed result from ReAct output."""
    thought: str | None
    action_name: str
    action_arg: str

    def __repr__(self) -> str:
        return f"ParsedAction(thought={self.thought!r}, action={self.action_name}({self.action_arg!r}))"


def parse_react_output(text: str) -> ParsedAction:
    """
    Parse ReAct format output to extract thought and action.

    Expected formats:
        Thought: ...
        Action: tool_name("argument")

    Or just:
        Action: tool_name("argument")
    """
    # Extract thought (optional)
    thought = None
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action (required)
    # Match patterns like: calculator("0.15 * 200") or finish("answer here")
    action_match = re.search(r"Action:\s*(\w+)\s*\(\s*[\"'](.+?)[\"']\s*\)", text, re.DOTALL)

    if not action_match:
        # Try alternative format: Action: tool_name(argument) without quotes
        action_match = re.search(r"Action:\s*(\w+)\s*\(\s*(.+?)\s*\)", text, re.DOTALL)

    if not action_match:
        raise ValueError(f"Could not parse action from: {text}")

    action_name = action_match.group(1)
    action_arg = action_match.group(2).strip().strip("\"'")

    return ParsedAction(thought=thought, action_name=action_name, action_arg=action_arg)


def main():
    print("=" * 60)
    print("REACT PARSER DEMO")
    print("=" * 60)

    # Test cases
    test_cases = [
        # Full format
        '''Thought: I need to calculate 15% of 200. 15% as decimal is 0.15.
Action: calculator("0.15 * 200")''',

        # Without thought
        '''Action: calculator("0.15 * 200")''',

        # Finish action
        '''Thought: I have calculated the result. The answer is 30.
Action: finish("15% of 200 is 30")''',

        # Weather query
        '''Thought: The user wants to know the weather in Tokyo.
Action: get_weather("Tokyo")''',

        # Multi-line thought
        '''Thought: This is a complex problem. Let me break it down:
1. First I need to calculate the percentage
2. Then get the weather
I'll start with the calculation.
Action: calculator("100 * 0.15")''',

        # Without action (should raise error)
        '''Thought: I forgot to include the action. This should fail.''',
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Input:\n{test}")
        print()

        try:
            result = parse_react_output(test)
            print(f"Parsed: {result}")
        except ValueError as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("NEXT: In 04_react_loop.py, we'll build the complete loop:")
    print("  1. Get model output")
    print("  2. Parse Thought/Action")
    print("  3. Execute action")
    print("  4. Append Observation")
    print("  5. Repeat until finish()")
    print("=" * 60)


if __name__ == "__main__":
    main()
