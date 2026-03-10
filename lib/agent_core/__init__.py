"""
agent_core - Shared library for tool-calling agents

Usage:
    from agent_core import ToolRegistry, run_agent, default_tools
"""

from agent_core.registry import ToolRegistry
from agent_core.agent import run_agent
from agent_core.tools import create_default_tools, calculator, get_weather, get_time

__all__ = [
    "ToolRegistry",
    "run_agent",
    "create_default_tools",
    "calculator",
    "get_weather",
    "get_time",
]
