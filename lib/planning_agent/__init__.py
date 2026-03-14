"""
planning_agent - Planning agent library

The Planning pattern creates a full plan before executing,
unlike ReAct which decides step by step.

Usage:
    from planning_agent import PlanningAgent, AgentConfig, DEFAULT_TOOLS

    client = genai.Client(api_key=api_key)
    config = AgentConfig(verbose=True)
    agent = PlanningAgent(client, config)

    response = agent.run("What is 15% of 200 and the weather in Tokyo?")
"""

from planning_agent.tools import (
    Tool,
    DEFAULT_TOOLS,
    create_default_tools,
    calculator,
    get_weather,
    get_time,
    search,
)
from planning_agent.agent import (
    PlanningAgent,
    AgentConfig,
    AgentStats,
    PlanAttempt,
    PlanStep,
    ExecutionResult,
    PlanExecutor,
    build_planning_prompt,
    parse_plan,
)

__all__ = [
    # Agent
    "PlanningAgent",
    "AgentConfig",
    "AgentStats",
    "PlanAttempt",
    # Plan
    "PlanStep",
    "ExecutionResult",
    "PlanExecutor",
    # Tools
    "Tool",
    "DEFAULT_TOOLS",
    "create_default_tools",
    "calculator",
    "get_weather",
    "get_time",
    "search",
    # Utilities
    "build_planning_prompt",
    "parse_plan",
]
