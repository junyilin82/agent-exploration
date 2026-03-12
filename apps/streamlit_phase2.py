"""
Streamlit UI for Phase 2 ReAct Agent

A web interface to demo the ReAct agent, showing Thought/Action/Observation steps.

RUN:
    cd apps
    uv sync
    uv run streamlit run streamlit_phase2.py
"""

import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from react_agent import ReActAgent, AgentConfig, DEFAULT_TOOLS

load_dotenv()

# Extra instructions for general knowledge questions
EXTRA_INSTRUCTIONS = """
- For general knowledge questions that don't require tools, use finish() directly with the answer
- Be concise and helpful in your responses
"""


def main():
    st.set_page_config(page_title="ReAct Agent", page_icon="🧠", layout="wide")

    st.title("🧠 ReAct Agent Demo")
    st.markdown("An agent that shows its **Thought → Action → Observation** reasoning process.")

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
        st.session_state.agent = ReActAgent(
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
        - What is 25 * 4 plus the temperature in Paris?
        - What is the capital of France?
        """)

        st.divider()

        # Show live stats
        stats = agent.get_stats()
        st.metric("API Calls", stats["api_calls"])
        st.metric("Tool Calls", stats["tool_calls"])

        st.divider()
        st.caption("Built with Streamlit + Gemini")
        st.caption("Phase 2: ReAct Pattern")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "steps" in message and message["steps"]:
                with st.expander("🧠 Reasoning Steps"):
                    for step in message["steps"]:
                        st.markdown(f"**Step {step['step']}**")
                        if step["thought"]:
                            st.markdown(f"💭 *Thought:* {step['thought']}")
                        st.markdown(f"⚡ *Action:* `{step['action']}(\"{step['action_arg']}\")`")
                        if step["observation"]:
                            st.markdown(f"👁️ *Observation:* `{step['observation']}`")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask me something..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.run(prompt)

            st.markdown(response)

            # Get steps from agent stats
            steps = agent.get_stats()["steps"]

            if steps:
                with st.expander("🧠 Reasoning Steps"):
                    for step in steps:
                        st.markdown(f"**Step {step['step']}**")
                        if step["thought"]:
                            st.markdown(f"💭 *Thought:* {step['thought']}")
                        st.markdown(f"⚡ *Action:* `{step['action']}(\"{step['action_arg']}\")`")
                        if step["observation"]:
                            st.markdown(f"👁️ *Observation:* `{step['observation']}`")
                        st.divider()

        # Add assistant message with steps
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "steps": steps,
        })

        # Rerun to update sidebar stats
        st.rerun()


if __name__ == "__main__":
    main()
