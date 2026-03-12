"""
05_full_react_agent.py - Complete ReAct Agent

PURPOSE:
    A polished, production-style react agent with:
    - Clean logging showing each step
    - Error handling
    - Configurable tools
    - Interactive mode

This is everything from scripts 01-04 combined into a usable agent.

RUN:
    uv run python 05_full_react_agent.py
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    GenerateContentConfig,
    Part,
)

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for the agent."""
    model_name: str = "gemini-2.5-flash"
    max_steps: int = 10
    verbose: bool = True


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def calculator(expression: str) -> float|str:
    """Evaluate a math expression."""
    try:
        # WARNING: eval is dangerous with untrusted input!
        # In production, use a safe math parser
        result = float(eval(expression))
        return result
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> dict:
    """Get weather for a city (mock implementation)."""
    mock_data = {
        "new york": {"temperature_f": 72, "condition": "sunny", "humidity": 45},
        "london": {"temperature_f": 55, "condition": "cloudy", "humidity": 80},
        "tokyo": {"temperature_f": 68, "condition": "clear", "humidity": 60},
        "paris": {"temperature_f": 63, "condition": "rainy", "humidity": 75},
    }
    return mock_data.get(city.lower(), {"temperature_f": 70, "condition": "unknown"})


def get_time(timezone: str) -> str:
    """Get current time (mock implementation)."""
    # In production, use actual timezone library
    mock_times = {
        "utc": "14:30",
        "est": "09:30",
        "pst": "06:30",
        "jst": "23:30",
    }
    return mock_times.get(timezone.lower(), "12:00 (unknown timezone)")

TOOLS: dict[str, Callable] = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_time": get_time,
}

# =============================================================================
# REACT PROMPT
# =============================================================================

REACT_PROMPT = '''You are a helpful assistant that solves problems step by step.

## How to Respond

You must ALWAYS respond in this exact format:

Thought: [Your reasoning about what to do next]
Action: [tool_name("argument")]

After each action, you will see an Observation with the result.
Then continue with another Thought/Action until you have the final answer.

When you have the final answer, use:
Thought: [Your final reasoning]
Action: finish("[Your complete answer to the user]")

You MUST include Thought before EVERY Action. Never skip it.

## Available Tools

1. calculator(expression) - Evaluate a math expression
   Example: calculator("15 * 20 + 5")

2. get_weather(city) - Get weather for a city
   Example: get_weather("Tokyo")

3. get_time(timezone) - Get current time in a timezone (UTC, EST, PST, JST)
   Example: get_time("EST")

4. finish(answer) - Return the final answer to the user
   Example: finish("The answer is 42")

## Rules

- Always start with a Thought explaining your reasoning
- Only use ONE Action per response
- Wait for the Observation before continuing
- Use finish() only when you have the complete answer

## Example

User: What is 25% of 80?

Thought: I need to calculate 25% of 80. 25% as a decimal is 0.25, so I multiply 0.25 * 80.
Action: calculator("0.25 * 80")

[Then you would see: Observation: 20.0]

Thought: The calculation shows 25% of 80 is 20. I have the answer.
Action: finish("25% of 80 is 20")

Now solve the following:

'''


# =============================================================================
# PARSER
# =============================================================================

@dataclass
class ParsedAction:
    thought: str | None
    action_name: str
    action_arg: str


def parse_react_output(text: str) -> ParsedAction:
    """Parse Thought/Action from model output."""
    # Extract thought
    thought = None
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action
    action_match = re.search(r"Action:\s*(\w+)\s*\(\s*[\"'](.+?)[\"']\s*\)", text, re.DOTALL)
    if not action_match:
        action_match = re.search(r"Action:\s*(\w+)\s*\(\s*(.+?)\s*\)", text, re.DOTALL)

    if not action_match:
        raise ValueError(f"Could not parse action from: {text}")

    return ParsedAction(
        thought=thought,
        action_name=action_match.group(1),
        action_arg=action_match.group(2).strip().strip("\"'"),
    )


# =============================================================================
# LOGGER
# =============================================================================

class AgentLogger:
    """Simple logger for agent activity."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.history: list[dict] = []

    def log(self, event: str, data: Any = None) -> None:
        """Log an event."""
        entry = {"event": event, "data": data}
        self.history.append(entry)
        if self.verbose:
            self._print(event, data)

    def _print(self, event: str, data: Any) -> None:
        """Print formatted log entry."""
        if event == "user_query":
            print(f"\n{'='*60}")
            print(f"USER: {data}")
            print(f"{'='*60}")

        elif event == "step_start":
            print(f"\n[Step {data}]")

        elif event == "tool_request":
            name, args = data
            print(f"  → Model requests: {name}({args})")

        elif event == "tool_result":
            name, result = data
            print(f"  ← Result: {result}")

        elif event == "model_response":
            print(f"\n{'─'*60}")
            print(f"AGENT:\n{data}")
            print(f"{'─'*60}")

        elif event == "error":
            print(f"\n[ERROR] {data}")

        elif event == "summary":
            loops, tools_called = data
            print(f"\n[Stats: {loops} API calls, {tools_called} tool executions]")


# =============================================================================
# THE AGENT
# =============================================================================

class ReActAgent:
    """
    A react agent using Gemini.

    This implements the core agent loop:
    1. Send prompt + user query to model
    2. Parse model responses, execute them
    3. Send results back to model
    4. Repeat until model returns finish action or max steps reached
    """

    def __init__(
        self,
        client: genai.Client,
        config: AgentConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or AgentConfig()
        self.logger = AgentLogger(verbose=self.config.verbose)
        self.step_count = 0
        self.system_prompt = REACT_PROMPT

    def get_stats(self) -> dict:
        """Get current agent stats."""
        return {
            "api_calls": self.step_count,
        }

    def run(self, user_query: str) -> str:
        """
        Run the agent on a user query.

        Returns the final text response.
        """
        # Reset step count for each run
        self.step_count = 0

        # Initialize conversation with user query
        contents = [Content(role="user", parts=[Part.from_text(text=user_query)])]

        self.logger.log("user_query", user_query)

        while self.step_count < self.config.max_steps:
            self.step_count += 1
            self.logger.log("step_start", self.step_count)

            # Call the model
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=contents,
                    config=GenerateContentConfig(
                        temperature=0,
                        system_instruction=self.system_prompt,
                    ),
                )
            except Exception as e:
                self.logger.log("error", f"API call failed: {e}")
                return f"Error: {e}"

            candidate = response.candidates[0]
            parts = candidate.content.parts
            output = "".join(
                p.text for p in parts 
                if hasattr(p, "text") and p.text
            ).strip()

            self.logger.log("model_response", output)

            # Add model's response to history
            contents.append(Content(role="model", parts=parts))

            # Parse the output            
            try:
                parsed = parse_react_output(output)
            except Exception as e:
                self.logger.log("error", f"Failed to parse model output: {e}")
                return f"Error: Could not parse model output"
            
            # Check if done
            if parsed.action_name == "finish":
                self.logger.log("final_response", parsed.action_arg)
                return parsed.action_arg
            
            # Execute the action
            if parsed.action_name not in TOOLS:
                observation = f"Error: Unknown tool '{parsed.action_name}'"
            else:
                tool = TOOLS[parsed.action_name]
                try:
                    observation = tool(parsed.action_arg)
                except Exception as e:
                    observation = f"Error executing tool: {e}"

            # Add observation to history
            contents.append(Content(role="user", parts=[Part.from_text(text=f"Observation: {observation}")]))

        # Max steps reached
        self.logger.log("error", "Max steps reached")
        return "Error: Agent exceeded maximum iterations"
    

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # Setup
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return

    client = genai.Client(api_key=api_key)
    config = AgentConfig(verbose=True)
    agent = ReActAgent(client, config)

    # Test queries
    queries = [
        "What is 15% of 200?",
        "What is 25 * 4 plus the temperature in Tokyo (just the number)?",
    ]

    for query in queries:
        agent.run(query)
        print()

    # Interactive mode
    print("\n" + "=" * 60)
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
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()