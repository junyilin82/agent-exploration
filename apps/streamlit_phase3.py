"""
Streamlit UI for Phase 3 Planning Agent

A web interface to demo the Planning agent, showing the plan generation and execution.

RUN:
    cd apps
    uv sync
    uv run streamlit run streamlit_phase3.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from planning_agent import PlanningAgent, AgentConfig, DEFAULT_TOOLS

load_dotenv()

# Extra instructions for general knowledge questions
EXTRA_INSTRUCTIONS = """
- For general knowledge questions that don't require tools, use finish() directly with the answer
- Be concise and helpful in your responses
"""


def main():
    st.set_page_config(page_title="Planning Agent", page_icon="📋", layout="wide")

    st.title("📋 Planning Agent Demo")
    st.markdown("An agent that **plans first**, then executes all steps.")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Create a .env file with your API key.")
        st.code("GOOGLE_API_KEY=your-key-here", language="bash")
        return

    # Initialize agent in session state (persists across reruns)
    if "agent" not in st.session_state:
        client = genai.Client(api_key=api_key)
        config = AgentConfig(verbose=False)  # Disable console logging
        st.session_state.agent = PlanningAgent(
            client,
            config,
            extra_instructions=EXTRA_INSTRUCTIONS,
        )

    agent = st.session_state.agent

    # Sidebar - show available tools and stats
    with st.sidebar:
        st.header("Available Tools")
        for tool in DEFAULT_TOOLS:
            st.markdown(f"- **{tool.name}** - {tool.description}")

        st.header("Example Queries")
        st.markdown("""
        - What is 15% of 250?
        - What's the weather in Tokyo?
        - What is 25 * 4 and the weather in London?
        - What time is it in Tokyo (JST)?
        """)

        st.divider()

        # Show live stats
        stats = agent.get_stats()
        st.metric("Planning Calls", stats["planning_calls"])
        st.metric("Execution Steps", stats["execution_steps"])
        st.metric("Re-plans", stats["replan_count"])

        st.divider()
        st.caption("Built with Streamlit + Gemini")
        st.caption("Phase 3: Planning Pattern")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "attempts" in message and message["attempts"]:
                for i, attempt in enumerate(message["attempts"], 1):
                    attempt_status = "✅" if attempt.success else "❌"
                    expander_title = f"{attempt_status} Attempt {i}" if len(message["attempts"]) > 1 else "📋 Plan & Execution"
                    with st.expander(expander_title):
                        st.markdown("**Plan:**")
                        for step in attempt.plan_steps:
                            st.markdown(f"  [{step.number}] {step.description}")
                            st.markdown(f"    Tool: `{step.tool}(\"{step.arg}\")`")
                        st.divider()
                        st.markdown("**Execution:**")
                        for result in attempt.execution_log:
                            status = "✅" if result.success else "❌"
                            st.markdown(f"  Step {result.step_number} {status}: `{result.tool}(\"{result.arg}\")` → `{result.result}`")

    # Chat input
    if prompt := st.chat_input("Ask me something..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Planning and executing..."):
                response = agent.run(prompt)

            st.markdown(response)

            # Get all attempts from agent stats
            stats = agent.get_stats()
            attempts = stats["attempts"]

            if attempts:
                for i, attempt in enumerate(attempts, 1):
                    attempt_status = "✅" if attempt.success else "❌"
                    expander_title = f"{attempt_status} Attempt {i}" if len(attempts) > 1 else "📋 Plan & Execution"
                    with st.expander(expander_title):
                        st.markdown("**Plan:**")
                        for step in attempt.plan_steps:
                            st.markdown(f"  [{step.number}] {step.description}")
                            st.markdown(f"    Tool: `{step.tool}(\"{step.arg}\")`")
                        st.divider()
                        st.markdown("**Execution:**")
                        for result in attempt.execution_log:
                            status = "✅" if result.success else "❌"
                            st.markdown(f"  Step {result.step_number} {status}: `{result.tool}(\"{result.arg}\")` → `{result.result}`")

        # Add assistant message with all attempts
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "attempts": attempts,
        })

        # Rerun to update sidebar stats
        st.rerun()


if __name__ == "__main__":
    main()
