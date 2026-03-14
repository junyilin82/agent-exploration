"""
Tool definitions for the Planning agent.
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
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Get weather for a city (mock implementation)."""
    mock_data = {
        "new york": "72°F, sunny",
        "london": "55°F, cloudy",
        "tokyo": "68°F, clear",
        "paris": "63°F, rainy",
    }
    result = mock_data.get(city.lower())
    if result is None:
        return f"Error: Unknown city '{city}'. Available: New York, London, Tokyo, Paris"
    return result


def get_time(timezone: str) -> str:
    """Get current time (mock implementation)."""
    mock_times = {
        "utc": "14:30 UTC",
        "est": "09:30 EST",
        "pst": "06:30 PST",
        "jst": "23:30 JST",
    }
    return mock_times.get(timezone.lower(), "12:00 (unknown timezone)")


def search(query: str) -> str:
    """Search for information (mock)."""
    return f"Search results for '{query}': [mock results]"


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
    Tool(
        name="search",
        description="Search for information",
        example='search("best restaurants")',
        func=search,
    ),
]


def create_default_tools() -> list[Tool]:
    """Create and return the default tools list."""
    return DEFAULT_TOOLS.copy()
