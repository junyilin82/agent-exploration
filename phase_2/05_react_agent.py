"""
05_react_agent_v2.py - Complete ReAct Agent (Alternative Implementation)

PURPOSE:
    A polished, production-style ReAct agent with:
    - Clean logging showing each step
    - Error handling
    - Configurable tools (pass tools as objects, not hardcoded)
    - Interactive mode
    - Stats tracking

Key differences from 05_full_react_agent.py:
    - Tools are defined as dataclass objects, not just functions
    - Prompt is built dynamically from tool list
    - Stats track both API calls and tool calls
    - Cleaner separation of concerns

RUN:
    uv run python 05_react_agent_v2.py
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for the ReAct agent."""

    model_name: str = "gemini-2.5-flash"
    max_steps: int = 10
    verbose: bool = True


@dataclass
class Tool:
    """A tool that the agent can use."""

    name: str
    description: str
    example: str
    func: Callable[[str], Any]


# =============================================================================
# DEFAULT TOOLS
# =============================================================================


def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        # WARNING: eval is dangerous with untrusted input!
        # In production, use a safe math parser like numexpr
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Get weather for a city (mock implementation)."""
    mock_data = {
        "new york": "72°F, sunny, humidity 45%",
        "london": "55°F, cloudy, humidity 80%",
        "tokyo": "68°F, clear, humidity 60%",
        "paris": "63°F, rainy, humidity 75%",
    }
    return mock_data.get(city.lower(), "70°F, unknown conditions")


def get_time(timezone: str) -> str:
    """Get current time (mock implementation)."""
    mock_times = {
        "utc": "14:30 UTC",
        "est": "09:30 EST",
        "pst": "06:30 PST",
        "jst": "23:30 JST",
    }
    return mock_times.get(timezone.lower(), "12:00 (unknown timezone)")


DEFAULT_TOOLS = [
    Tool(
        name="calculator",
        description="Evaluate a math expression",
        example='calculator("15 * 20 + 5")',
        func=calculator,
    ),
    Tool(
        name="get_weather",
        description="Get weather for a city",
        example='get_weather("Tokyo")',
        func=get_weather,
    ),
    Tool(
        name="get_time",
        description="Get current time in a timezone (UTC, EST, PST, JST)",
        example='get_time("EST")',
        func=get_time,
    ),
]


# =============================================================================
# PROMPT BUILDER
# =============================================================================


def build_react_prompt(tools: list[Tool], extra_instructions: str = "") -> str:
    """Build the ReAct prompt dynamically from the tool list.

    Args:
        tools: List of Tool objects to include in the prompt.
        extra_instructions: Additional instructions to append to the prompt.
    """
    tool_descriptions = []
    for i, tool in enumerate(tools, 1):
        tool_descriptions.append(
            f"{i}. {tool.name}(argument) - {tool.description}\n"
            f"   Example: {tool.example}"
        )

    tools_section = "\n\n".join(tool_descriptions)

    # Add extra instructions section if provided
    extra_section = ""
    if extra_instructions:
        extra_section = f"\n\n## Additional Instructions\n\n{extra_instructions}"

    return f"""You are a helpful assistant that solves problems step by step.

## How to Respond

You must ALWAYS respond in this exact format:

Thought: [Your reasoning about what to do next]
Action: [tool_name("argument")]

After each action, you will see an Observation with the result.
Then continue with another Thought/Action until you have the final answer.

When you have the final answer, use:
Thought: [Your final reasoning]
Action: finish("[Your complete answer to the user]")

IMPORTANT: You MUST include a Thought before EVERY Action. Never skip the Thought.

## Available Tools

{tools_section}

{len(tools) + 1}. finish(answer) - Return the final answer to the user
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

Observation: 20.0

Thought: The calculation shows 25% of 80 is 20. I have the answer.
Action: finish("25% of 80 is 20")
{extra_section}
Now solve the following:

"""


# =============================================================================
# PARSER
# =============================================================================


@dataclass
class ParsedAction:
    """Parsed result from ReAct output."""

    thought: str | None
    action_name: str
    action_arg: str


def parse_react_output(text: str) -> ParsedAction:
    """Parse Thought/Action from model output."""
    # Extract thought (optional)
    thought = None
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action (required) - try quoted first, then unquoted
    action_match = re.search(
        r"Action:\s*(\w+)\s*\(\s*[\"'](.+?)[\"']\s*\)", text, re.DOTALL
    )
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
# AGENT STATS
# =============================================================================


@dataclass
class AgentStats:
    """Statistics from an agent run."""

    api_calls: int = 0
    tool_calls: int = 0
    steps: list[dict] = field(default_factory=list)

    def record_step(
        self,
        step_num: int,
        thought: str | None,
        action: str,
        action_arg: str,
        observation: str | None = None,
    ) -> None:
        """Record a step in the agent's execution."""
        self.steps.append(
            {
                "step": step_num,
                "thought": thought,
                "action": action,
                "action_arg": action_arg,
                "observation": observation,
            }
        )


# =============================================================================
# THE AGENT
# =============================================================================


class ReActAgent:
    """
    A ReAct agent using Gemini.

    The ReAct loop:
    1. Send prompt + user query to model
    2. Parse Thought/Action from response
    3. If finish(), return the answer
    4. Execute the tool, get observation
    5. Append observation to context
    6. Repeat until finish() or max steps
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
        self.system_prompt = build_react_prompt(self.tools, extra_instructions)

        # Build tool lookup
        self._tool_funcs: dict[str, Callable] = {t.name: t.func for t in self.tools}

        # Stats for current/last run
        self.stats = AgentStats()

    def _log(self, message: str) -> None:
        """Print if verbose mode is on."""
        if self.config.verbose:
            print(message)

    def _execute_tool(self, name: str, arg: str) -> str:
        """Execute a tool and return the result as a string."""
        if name not in self._tool_funcs:
            return f"Error: Unknown tool '{name}'"

        try:
            result = self._tool_funcs[name](arg)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {e}"

    def run(self, user_query: str) -> str:
        """
        Run the agent on a user query.

        Returns the final answer as a string.
        """
        # Reset stats for this run
        self.stats = AgentStats()

        self._log(f"\n{'='*60}")
        self._log(f"User: {user_query}")
        self._log(f"{'='*60}")

        # Initialize conversation
        contents: list[Content] = [
            Content(role="user", parts=[Part.from_text(text=user_query)])
        ]

        for step in range(1, self.config.max_steps + 1):
            self._log(f"\n[Step {step}]")
            self.stats.api_calls += 1

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
                self._log(f"Error: API call failed - {e}")
                return f"Error: API call failed - {e}"

            # Extract text from response
            output = response.text.strip() if response.text else ""
            self._log(f"Model:\n{output}")

            # Add model response to conversation history
            contents.append(
                Content(role="model", parts=[Part.from_text(text=output)])
            )

            # Parse the output
            try:
                parsed = parse_react_output(output)
            except ValueError as e:
                self._log(f"Error: {e}")
                return "Error: Could not parse model output"

            # Check if done
            if parsed.action_name == "finish":
                self.stats.record_step(
                    step, parsed.thought, "finish", parsed.action_arg
                )
                self._log(f"\n{'─'*60}")
                self._log(f"Final Answer: {parsed.action_arg}")
                self._log(f"{'─'*60}")
                return parsed.action_arg

            # Execute the tool
            self.stats.tool_calls += 1
            observation = self._execute_tool(parsed.action_name, parsed.action_arg)
            self._log(f"Observation: {observation}")

            # Record this step
            self.stats.record_step(
                step, parsed.thought, parsed.action_name, parsed.action_arg, observation
            )

            # Add observation to conversation
            contents.append(
                Content(
                    role="user",
                    parts=[Part.from_text(text=f"Observation: {observation}")],
                )
            )

        # Max steps reached
        self._log(f"\nError: Max steps ({self.config.max_steps}) reached")
        return "Error: Agent exceeded maximum iterations"

    def get_stats(self) -> dict:
        """Get statistics from the last run."""
        return {
            "api_calls": self.stats.api_calls,
            "tool_calls": self.stats.tool_calls,
            "steps": self.stats.steps,
        }


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
    print("=" * 60)
    print("REACT AGENT DEMO")
    print("=" * 60)

    test_queries = [
        "What is 15% of 200?",
        "What is 25 * 4 plus the temperature in Tokyo (just the number)?",
    ]

    for query in test_queries:
        agent.run(query)
        stats = agent.get_stats()
        print(f"\n[Stats: {stats['api_calls']} API calls, {stats['tool_calls']} tool executions]")
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
            print(f"\n[Stats: {stats['api_calls']} API calls, {stats['tool_calls']} tool executions]")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
