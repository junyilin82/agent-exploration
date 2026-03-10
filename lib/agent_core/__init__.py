"""
agent_core - Shared library for tool-calling agents

Usage:
    # Simple function approach
    from agent_core import ToolRegistry, run_agent, create_default_tools

    # Full class approach (with stats, logging, introspection)
    from agent_core import ToolCallingAgent, AgentConfig, AgentLogger
"""

from agent_core.registry import ToolRegistry
from agent_core.agent import (
    run_agent,
    ToolCallingAgent,
    AgentConfig,
    AgentLogger,
)
from agent_core.tools import (
    create_default_tools,
    calculator,
    get_weather,
    get_time,
)

__all__ = [
    # Registry
    "ToolRegistry",
    # Agent (simple function)
    "run_agent",
    # Agent (full class)
    "ToolCallingAgent",
    "AgentConfig",
    "AgentLogger",
    # Default tools
    "create_default_tools",
    "calculator",
    "get_weather",
    "get_time",
]
