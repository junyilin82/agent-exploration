"""
05_full_agent.py - Complete Tool-Calling Agent

PURPOSE:
    A polished, production-style agent with:
    - Clean logging showing each step
    - Error handling
    - Configurable tools
    - Interactive mode

This is everything from scripts 01-04 combined into a usable agent.

RUN:
    uv run python 05_full_agent.py
"""

import os
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    Tool,
)

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for the agent."""
    model_name: str = "gemini-2.5-flash"
    max_loops: int = 10
    verbose: bool = True


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """
    Registry for tools. Holds both:
    - Function implementations (what to execute)
    - Function schemas (what to tell the model)
    """

    def __init__(self) -> None:
        self._functions: dict[str, Callable] = {}
        self._schemas: list[FunctionDeclaration] = []

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: Callable,
    ) -> None:
        """Register a tool with its schema and implementation."""
        self._functions[name] = func
        self._schemas.append(
            FunctionDeclaration(
                name=name,
                description=description,
                parameters=parameters,
            )
        )

    def get_function(self, name: str) -> Callable:
        """Get a function by name."""
        if name not in self._functions:
            raise ValueError(f"Unknown function: {name}")
        return self._functions[name]

    def get_tools(self) -> Tool:
        """Get Tool object for the API."""
        return Tool(function_declarations=self._schemas)

    def list_tools(self) -> list[str]:
        """List registered tool names."""
        return list(self._functions.keys())


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def calculator(expression: str) -> float:
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


# =============================================================================
# SETUP TOOLS
# =============================================================================

def create_default_tools() -> ToolRegistry:
    """Create registry with default tools."""
    registry = ToolRegistry()

    registry.register(
        name="calculator",
        description="Evaluate a mathematical expression. Use for any calculations.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression, e.g., '25 * 4 + 10' or 'sqrt(16)'",
                }
            },
            "required": ["expression"],
        },
        func=calculator,
    )

    registry.register(
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g., 'Tokyo' or 'New York'",
                }
            },
            "required": ["city"],
        },
        func=get_weather,
    )

    registry.register(
        name="get_time",
        description="Get current time in a timezone.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone code, e.g., 'UTC', 'EST', 'PST', 'JST'",
                }
            },
            "required": ["timezone"],
        },
        func=get_time,
    )

    # You can add more tools here as needed

    return registry


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

        elif event == "loop_start":
            print(f"\n[Loop {data}]")

        elif event == "tool_request":
            name, args = data
            print(f"  → Model requests: {name}({args})")

        elif event == "tool_result":
            name, result = data
            print(f"  ← Result: {result}")

        elif event == "final_response":
            print(f"\n{'─'*60}")
            print(f"AGENT: {data}")
            print(f"{'─'*60}")

        elif event == "error":
            print(f"\n[ERROR] {data}")

        elif event == "summary":
            loops, tools_called = data
            print(f"\n[Stats: {loops} API calls, {tools_called} tool executions]")


# =============================================================================
# THE AGENT
# =============================================================================

class ToolCallingAgent:
    """
    A tool-calling agent using Gemini.

    This implements the core agent loop:
    1. Send user query + tool schemas to model
    2. If model returns function_call(s), execute them
    3. Send results back to model
    4. Repeat until model returns text
    """

    def __init__(
        self,
        client: genai.Client,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.config = config or AgentConfig()
        self.logger = AgentLogger(verbose=self.config.verbose)
        self.loop_count = 0
        self.total_tool_calls = 0
        self._register_introspection_tool()

    def _register_introspection_tool(self) -> None:
        """Register a tool that allows the model to query agent stats."""
        self.tools.register(
            name="get_agent_stats",
            description="Get current agent stats like API calls and tool calls.",
            parameters={
                "type": "object",
                "properties": {
                    "api_calls": {
                        "type": "integer",
                        "description": "Current API calls, i.e., loop count of the agent.",
                    },
                    "tool_calls": {
                        "type": "integer",
                        "description": "Total number of tools called so far.",
                    },
                },
                "required": ["api_calls", "tool_calls"],
            },
            func=self.get_stats,
        )

    def get_stats(self) -> dict:
        """Get current agent stats."""
        return {
            "api_calls": self.loop_count,
            "tool_calls": self.total_tool_calls,
        }

    def run(self, user_query: str) -> str:
        """
        Run the agent on a user query.

        Returns the final text response.
        """
        self.logger.log("user_query", user_query)

        # Initialize conversation
        contents = [Content(role="user", parts=[Part.from_text(text=user_query)])]

        while self.loop_count < self.config.max_loops:
            self.loop_count += 1
            self.logger.log("loop_start", self.loop_count)

            # Call the model
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=contents,
                    config=GenerateContentConfig(tools=[self.tools.get_tools()]),
                )
            except Exception as e:
                self.logger.log("error", f"API call failed: {e}")
                return f"Error: {e}"

            candidate = response.candidates[0]
            parts = candidate.content.parts

            # Check for function calls
            function_calls = [
                p for p in parts
                if hasattr(p, "function_call") and p.function_call
            ]

            # No function calls = we're done
            if not function_calls:
                final_text = response.text
                self.logger.log("final_response", final_text)
                self.logger.log("summary", (self.loop_count, self.total_tool_calls))
                return final_text

            # Add model's response to history
            contents.append(Content(role="model", parts=parts))

            # Execute each function call
            function_response_parts = []

            for part in function_calls:
                fc = part.function_call
                func_name = fc.name
                func_args = dict(fc.args)

                self.logger.log("tool_request", (func_name, func_args))
                self.total_tool_calls += 1

                # Execute
                try:
                    func = self.tools.get_function(func_name)
                    result = func(**func_args)
                except Exception as e:
                    result = f"Error executing {func_name}: {e}"

                self.logger.log("tool_result", (func_name, result))

                # Create response part
                function_response_parts.append(
                    Part.from_function_response(
                        name=func_name,
                        response={"result": result},
                    )
                )

            # Add results to history
            contents.append(Content(role="user", parts=function_response_parts))

        # Max loops reached
        self.logger.log("error", "Max loops reached")
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
    tools = create_default_tools()
    config = AgentConfig(verbose=True)
    agent = ToolCallingAgent(client, tools, config)

    # Show available tools
    print("=" * 60)
    print("TOOL-CALLING AGENT")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Available tools: {', '.join(tools.list_tools())}")

    # Demo queries
    queries = [
        "What is 15% of 85?",
        "What's the weather in Paris and what time is it in Tokyo?",
        "If it's 72°F in New York and 55°F in London, what's the difference?",
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
