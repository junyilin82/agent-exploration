"""
Default tool implementations
"""

from agent_core.registry import ToolRegistry


def calculator(expression: str) -> float:
    """Evaluate a math expression."""
    try:
        # WARNING: eval is dangerous with untrusted input!
        # In production, use a safe math parser
        return float(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> dict:
    """Get weather for a city (mock implementation)."""
    mock_data = {
        "new york": {"temperature_f": 72, "condition": "sunny", "humidity": 45},
        "london": {"temperature_f": 55, "condition": "cloudy", "humidity": 80},
        "tokyo": {"temperature_f": 68, "condition": "clear", "humidity": 60},
        "paris": {"temperature_f": 63, "condition": "rainy", "humidity": 75},
        "san francisco": {"temperature_f": 65, "condition": "foggy", "humidity": 70},
    }
    return mock_data.get(city.lower(), {"temperature_f": 70, "condition": "unknown"})


def get_time(timezone: str) -> str:
    """Get current time (mock implementation)."""
    mock_times = {
        "utc": "14:30",
        "est": "09:30",
        "pst": "06:30",
        "jst": "23:30",
    }
    return mock_times.get(timezone.lower(), "12:00 (unknown timezone)")


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
                    "description": "Math expression, e.g., '25 * 4 + 10'",
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
        description="Get current time in a timezone (UTC, EST, PST, JST).",
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

    return registry
