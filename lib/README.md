# agent-core

Shared library for tool-calling agents.

## Components

- **ToolRegistry** - Register tools with schemas and implementations
- **run_agent** - The agent loop (call model → execute tools → repeat)
- **Default tools** - calculator, get_weather, get_time

## Usage

```python
from agent_core import ToolRegistry, run_agent, create_default_tools

# Use default tools
tools = create_default_tools()

# Or create custom registry
tools = ToolRegistry()
tools.register(
    name="my_tool",
    description="Does something useful",
    parameters={"type": "object", "properties": {...}},
    func=my_function,
)

# Run agent
client = genai.Client(api_key=api_key)
response, log = run_agent(client, tools, "What is 2 + 2?")
```

## Install as dependency

In your `pyproject.toml`:

```toml
[project]
dependencies = ["agent-core"]

[tool.uv.sources]
agent-core = { path = "../lib", editable = true }
```
