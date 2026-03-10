"""
Tool Registry - manages tool schemas and implementations
"""

from typing import Callable

from google.genai.types import FunctionDeclaration, Tool


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
