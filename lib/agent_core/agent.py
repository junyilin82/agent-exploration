"""
Agent loop implementation
"""

from google import genai
from google.genai.types import Content, GenerateContentConfig, Part

from agent_core.registry import ToolRegistry


def run_agent(
    client: genai.Client,
    tools: ToolRegistry,
    user_query: str,
    model_name: str = "gemini-2.5-flash",
    max_loops: int = 10,
) -> tuple[str, list[dict]]:
    """
    Run the agent loop until we get a final text response.

    Args:
        client: Gemini API client
        tools: Tool registry with available tools
        user_query: The user's question
        model_name: Model to use
        max_loops: Maximum iterations (safety limit)

    Returns:
        Tuple of (response_text, execution_log)
        - response_text: The final text response from the model
        - execution_log: List of tool calls made [{tool, args, result}, ...]
    """
    contents = [Content(role="user", parts=[Part.from_text(text=user_query)])]
    log = []

    for _ in range(max_loops):
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=GenerateContentConfig(tools=[tools.get_tools()]),
        )

        parts = response.candidates[0].content.parts
        function_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

        # No function calls = model is done
        if not function_calls:
            return response.text, log

        # Add model's response to history
        contents.append(Content(role="model", parts=parts))

        # Execute each function call
        function_response_parts = []

        for part in function_calls:
            fc = part.function_call
            func_name = fc.name
            func_args = dict(fc.args)

            # Execute
            try:
                func = tools.get_function(func_name)
                result = func(**func_args)
            except Exception as e:
                result = f"Error: {e}"

            log.append({"tool": func_name, "args": func_args, "result": result})

            function_response_parts.append(
                Part.from_function_response(name=func_name, response={"result": result})
            )

        # Add results to history
        contents.append(Content(role="user", parts=function_response_parts))

    return "Error: Max loops reached", log
