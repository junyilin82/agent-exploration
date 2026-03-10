"""
Agent implementations

Provides both:
- run_agent() - Simple function for quick use
- ToolCallingAgent - Full class with stats, logging, introspection
"""

from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai.types import Content, GenerateContentConfig, Part

from agent_core.registry import ToolRegistry


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_SYSTEM_INSTRUCTION = """You are a helpful assistant with access to tools.
Use tools when they are relevant to answering the question.
For general knowledge questions, answer directly without using tools."""


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    model_name: str = "gemini-2.5-flash"
    max_loops: int = 10
    verbose: bool = True
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION


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
# FULL AGENT CLASS
# =============================================================================

class ToolCallingAgent:
    """
    A tool-calling agent using Gemini.

    This implements the core agent loop:
    1. Send user query + tool schemas to model
    2. If model returns function_call(s), execute them
    3. Send results back to model
    4. Repeat until model returns text

    Features:
    - Stats tracking (api_calls, tool_calls)
    - Logging with history
    - Introspection tool (model can query its own stats)
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
        self.api_calls = 0
        self.tool_calls = 0
        self._register_introspection_tools()

    def _register_introspection_tools(self) -> None:
        """Register tools that allow the model to query agent state."""
        self.tools.register(
            name="get_agent_stats",
            description="Get statistics about this agent session (API calls made, tools executed).",
            parameters={
                "type": "object",
                "properties": {},
            },
            func=self.get_stats,
        )

    def get_stats(self) -> dict:
        """Get current agent statistics."""
        return {
            "api_calls": self.api_calls,
            "tool_calls": self.tool_calls,
        }

    def run(self, user_query: str) -> str:
        """
        Run the agent on a user query.

        Returns the final text response.
        """
        self.logger.log("user_query", user_query)

        # Initialize conversation
        contents = [Content(role="user", parts=[Part.from_text(text=user_query)])]

        while self.api_calls < self.config.max_loops:
            self.api_calls += 1
            self.logger.log("loop_start", self.api_calls)

            # Call the model
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=contents,
                    config=GenerateContentConfig(
                        tools=[self.tools.get_tools()],
                        system_instruction=self.config.system_instruction,
                    ),
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
                self.logger.log("summary", (self.api_calls, self.tool_calls))
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
                self.tool_calls += 1

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
# SIMPLE FUNCTION (for quick use)
# =============================================================================

def run_agent(
    client: genai.Client,
    tools: ToolRegistry,
    user_query: str,
    model_name: str = "gemini-2.5-flash",
    max_loops: int = 10,
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
) -> tuple[str, list[dict]]:
    """
    Run the agent loop until we get a final text response.

    This is a simpler alternative to ToolCallingAgent for quick use.

    Args:
        client: Gemini API client
        tools: Tool registry with available tools
        user_query: The user's question
        model_name: Model to use
        max_loops: Maximum iterations (safety limit)
        system_instruction: Instructions for the model

    Returns:
        Tuple of (response_text, execution_log)
        - response_text: The final text response from the model
        - execution_log: List of tool calls made [{tool, args, result}, ...]
    """
    contents = [Content(role="user", parts=[Part.from_text(text=user_query)])]
    log = []

    for _ in range(max_loops):
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=GenerateContentConfig(
                tools=[tools.get_tools()],
                system_instruction=system_instruction,
            ),
        )

        parts = response.candidates[0].content.parts
        function_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

        # No function calls = model is done
        if not function_calls:
            return response.text, log

        # Add model's response to history
        contents.append(Content(role="model", parts=parts))

        # Execute each function call
        function_response_parts = []

        for part in function_calls:
            fc = part.function_call
            func_name = fc.name
            func_args = dict(fc.args)

            # Execute
            try:
                func = tools.get_function(func_name)
                result = func(**func_args)
            except Exception as e:
                result = f"Error: {e}"

            log.append({"tool": func_name, "args": func_args, "result": result})

            function_response_parts.append(
                Part.from_function_response(name=func_name, response={"result": result})
            )

        # Add results to history
        contents.append(Content(role="user", parts=function_response_parts))

    return "Error: Max loops reached", log
