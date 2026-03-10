"""
Streamlit UI for Phase 1 Tool-Calling Agent

A simple web interface to demo the agent to others.

RUN:
    cd apps
    uv sync
    uv run streamlit run streamlit_phase1.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from agent_core import ToolCallingAgent, AgentConfig, create_default_tools

load_dotenv()


def main():
    st.set_page_config(page_title="Tool-Calling Agent", page_icon="🤖", layout="wide")

    st.title("🤖 Tool-Calling Agent Demo")
    st.markdown("A simple agent that can use tools to answer questions.")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Create a .env file with your API key.")
        st.code("GOOGLE_API_KEY=your-key-here", language="bash")
        return

    # Initialize agent in session state (persists across reruns)
    if "agent" not in st.session_state:
        client = genai.Client(api_key=api_key)
        tools = create_default_tools()
        config = AgentConfig(verbose=False)  # Disable console logging
        st.session_state.agent = ToolCallingAgent(client, tools, config)

    agent = st.session_state.agent

    # Sidebar - show available tools and stats
    with st.sidebar:
        st.header("Available Tools")
        st.markdown("""
        - **calculator** - Math expressions
        - **get_weather** - Weather for cities
        - **get_time** - Time in timezones
        - **get_agent_stats** - Agent can query its own stats
        """)

        st.header("Example Queries")
        st.markdown("""
        - What is 15% of 250?
        - What's the weather in Tokyo?
        - What time is it in PST?
        - How many API calls have you made?
        """)

        st.divider()

        # Show live stats
        stats = agent.get_stats()
        st.metric("API Calls", stats["api_calls"])
        st.metric("Tool Calls", stats["tool_calls"])

        st.divider()
        st.caption("Built with Streamlit + Gemini")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "log" in message and message["log"]:
                with st.expander("🔧 Tool calls"):
                    for entry in message["log"]:
                        st.code(f"{entry['tool']}({entry['args']}) → {entry['result']}")

    # Chat input
    if prompt := st.chat_input("Ask me something..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Track logger history length before run
        log_start = len(agent.logger.history)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.run(prompt)

            st.markdown(response)

            # Extract tool calls from logger history (since this run)
            log = []
            for entry in agent.logger.history[log_start:]:
                if entry["event"] == "tool_request":
                    name, args = entry["data"]
                    log.append({"tool": name, "args": args, "result": None})
                elif entry["event"] == "tool_result" and log:
                    name, result = entry["data"]
                    log[-1]["result"] = result

            if log:
                with st.expander("🔧 Tool calls"):
                    for entry in log:
                        st.code(f"{entry['tool']}({entry['args']}) → {entry['result']}")

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response, "log": log})

        # Rerun to update sidebar stats
        st.rerun()


if __name__ == "__main__":
    main()
