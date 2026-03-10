"""
04b_mcp_comparison.py - Comparing Our Code to MCP

This file shows how MCP standardizes what we built manually.
"""

# =============================================================================
# WHAT WE BUILT (Gemini-specific)
# =============================================================================

our_tool_definition = """
FunctionDeclaration(
    name="calculator",
    description="Evaluate a mathematical expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression, e.g., '25 * 4'"
            }
        },
        "required": ["expression"]
    }
)
"""

# =============================================================================
# MCP tools/list RESPONSE (Standardized)
# =============================================================================
# When a client calls the tools/list endpoint on an MCP server,
# this is the JSON response format:

mcp_tools_list_response = {
    "tools": [
        {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g., '25 * 4'"
                    }
                },
                "required": ["expression"]
            }
        },
        {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    ]
}

# =============================================================================
# MCP tools/call REQUEST (How you invoke a tool)
# =============================================================================

mcp_tools_call_request = {
    "method": "tools/call",
    "params": {
        "name": "calculator",
        "arguments": {
            "expression": "127 * 348"
        }
    }
}

# =============================================================================
# MCP tools/call RESPONSE (What you get back)
# =============================================================================

mcp_tools_call_response = {
    "content": [
        {
            "type": "text",
            "text": "44196"
        }
    ]
}

# =============================================================================
# COMPARISON
# =============================================================================

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("MCP vs OUR IMPLEMENTATION")
    print("=" * 70)

    print("\n[1] TOOL DEFINITION COMPARISON\n")
    print("Our Gemini FunctionDeclaration:")
    print("-" * 40)
    print(our_tool_definition)

    print("MCP tools/list response:")
    print("-" * 40)
    print(json.dumps(mcp_tools_list_response, indent=2))

    print("\n[2] TOOL INVOCATION\n")
    print("Our code:")
    print("-" * 40)
    print("""
    # Model returns:
    function_call.name = "calculator"
    function_call.args = {"expression": "127 * 348"}

    # We execute:
    result = FUNCTION_MAP["calculator"](expression="127 * 348")
    """)

    print("MCP tools/call:")
    print("-" * 40)
    print("Request:", json.dumps(mcp_tools_call_request, indent=2))
    print("\nResponse:", json.dumps(mcp_tools_call_response, indent=2))

    print("\n" + "=" * 70)
    print("KEY DIFFERENCES")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  Aspect          │ Our Code              │ MCP                  │
    ├──────────────────┼───────────────────────┼──────────────────────┤
    │  Tool schema     │ Gemini-specific       │ Standardized JSON    │
    │  Discovery       │ Hardcoded             │ tools/list endpoint  │
    │  Execution       │ Direct function call  │ tools/call endpoint  │
    │  Transport       │ Same process          │ stdio/HTTP/SSE       │
    │  Interop         │ Gemini only           │ Any LLM client       │
    └─────────────────────────────────────────────────────────────────┘

    MCP is essentially:
    1. A standard schema format (inputSchema instead of parameters)
    2. A standard discovery protocol (tools/list)
    3. A standard execution protocol (tools/call)
    4. A transport layer (so tools can run in separate processes)

    The CONCEPTS are identical to what we built. MCP just standardizes them.
    """)
