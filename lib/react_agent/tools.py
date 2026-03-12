"""
Tool definitions for the ReAct agent.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """A tool that the agent can use."""

    name: str
    description: str
    example: str
    func: Callable[[str], Any]


# =============================================================================
# DEFAULT TOOL IMPLEMENTATIONS
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


# =============================================================================
# DEFAULT TOOLS LIST
# =============================================================================


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


def create_default_tools() -> list[Tool]:
    """Create and return the default tools list."""
    return DEFAULT_TOOLS.copy()
