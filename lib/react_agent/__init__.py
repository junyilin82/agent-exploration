"""
react_agent - ReAct (Reasoning + Acting) agent library

The ReAct pattern makes the model's reasoning visible through
explicit Thought/Action/Observation steps.

Usage:
    from react_agent import ReActAgent, AgentConfig, Tool, DEFAULT_TOOLS

    client = genai.Client(api_key=api_key)
    config = AgentConfig(verbose=True)
    agent = ReActAgent(client, config)

    response = agent.run("What is 15% of 200?")
"""

from react_agent.tools import (
    Tool,
    DEFAULT_TOOLS,
    create_default_tools,
    calculator,
    get_weather,
    get_time,
)
from react_agent.agent import (
    ReActAgent,
    AgentConfig,
    AgentStats,
    ParsedAction,
    build_react_prompt,
    parse_react_output,
)

__all__ = [
    # Agent
    "ReActAgent",
    "AgentConfig",
    "AgentStats",
    # Tools
    "Tool",
    "DEFAULT_TOOLS",
    "create_default_tools",
    "calculator",
    "get_weather",
    "get_time",
    # Utilities
    "ParsedAction",
    "build_react_prompt",
    "parse_react_output",
]
