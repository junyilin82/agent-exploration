"""
Streamlit UI for Phase 4 Multi-Agent Systems

A web interface to demo multi-agent patterns: Debate, Specialist Router,
Orchestrator/Worker, and the Combined System.

RUN:
    cd apps
    uv sync
    uv run streamlit run streamlit_phase4.py
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


# =============================================================================
# TRACE EVENT — captures what each agent did for UI display
# =============================================================================


@dataclass
class TraceEvent:
    """A single step in the multi-agent pipeline."""

    agent: str
    action: str
    content: str
    metadata: dict = field(default_factory=dict)


AGENT_ICONS = {
    "Proposer": "💬",
    "Critic": "🔍",
    "Router": "🔀",
    "Math Specialist": "🔢",
    "Code Specialist": "💻",
    "Writing Specialist": "✍️",
    "General Specialist": "💡",
    "Classifier": "🏷️",
    "Orchestrator": "🎯",
    "Worker": "⚙️",
    "Synthesizer": "🧩",
}


# =============================================================================
# ENUMS
# =============================================================================


class SpecialistType(Enum):
    MATH = "math"
    CODE = "code"
    WRITING = "writing"
    GENERAL = "general"


class TaskComplexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class TaskDomain(Enum):
    MATH = "math"
    CODE = "code"
    WRITING = "writing"
    RESEARCH = "research"
    GENERAL = "general"


# =============================================================================
# PROMPTS
# =============================================================================

# -- Debate prompts --

PROPOSER_INITIAL_PROMPT = """You are a helpful assistant that provides thorough, accurate answers.

Given a question, provide a clear and comprehensive answer.

Be specific and include relevant details. If you're uncertain about something, say so.

Question: {question}

Your answer:"""

PROPOSER_REVISE_PROMPT = """You are a helpful assistant revising your answer based on feedback.

Original question: {question}

Your previous answer:
{previous_answer}

Critic's feedback:
{critique}

Please revise your answer to address the valid points in the critique.
If you disagree with any criticism, explain why.

Your revised answer:"""

DEBATE_CRITIC_PROMPT = """You are a critical reviewer who checks answers for accuracy and completeness.

Your job is to:
1. Identify any factual errors or unsupported claims
2. Point out missing important information
3. Suggest improvements

Be constructive but thorough. If the answer is already excellent, say "APPROVED" and briefly explain why.

Question being answered: {question}

Answer to review:
{answer}

Your critique (or "APPROVED" if the answer is excellent):"""

# -- Router prompts --

ROUTER_PROMPT = """You are a routing assistant that classifies questions.

Analyze the user's question and decide which specialist should handle it.

Available specialists:
- MATH: For calculations, equations, statistics, mathematical reasoning
- CODE: For programming, debugging, code explanation, algorithms
- WRITING: For essays, creative writing, editing, summarization
- GENERAL: For factual questions, advice, explanations of concepts

Respond with ONLY one word: MATH, CODE, WRITING, or GENERAL

Question: {question}

Specialist:"""

SPECIALIST_PROMPTS = {
    SpecialistType.MATH: """You are a mathematics expert. Solve problems step-by-step.
Question: {question}
Your solution:""",
    SpecialistType.CODE: """You are a programming expert. Write clean, well-documented code.
Request: {question}
Your code:""",
    SpecialistType.WRITING: """You are a writing expert. Create clear, engaging content.
Request: {question}
Your writing:""",
    SpecialistType.GENERAL: """You are a knowledgeable general assistant. Provide a clear, helpful response.
Request: {question}
Your response:""",
}

# -- Orchestrator prompts --

DECOMPOSE_PROMPT = """You are a task decomposition expert.

Given a complex request, break it down into simple, independent subtasks.

Rules:
- Each subtask should be self-contained
- Subtasks should be in logical order
- Keep it to 2-4 subtasks
- Each subtask should be something a simple assistant can do

Format your response as:
SUBTASKS:
[1] First subtask description
[2] Second subtask description
END_SUBTASKS

User request: {request}

Your decomposition:"""

WORKER_PROMPT = """You are a helpful assistant completing a specific subtask.

Complete the following task thoroughly and concisely.
Provide only the result - no need to explain what you're doing.

Task: {task}

Your result:"""

SYNTHESIZE_PROMPT = """You are an expert at combining information into coherent answers.

The user asked: {original_request}

Here are the results from completing each subtask:

{subtask_results}

Synthesize these results into a single, coherent response for the user.
Make it flow naturally - don't just list the subtask results.

Your synthesized answer:"""

# -- Combined system prompts --

CLASSIFIER_PROMPT = """Analyze this request and classify it.

1. COMPLEXITY: Is this SIMPLE (can be answered directly) or COMPLEX (needs multiple steps/subtasks)?
2. DOMAIN: What domain is this? MATH, CODE, WRITING, RESEARCH, or GENERAL?

Respond in this exact format:
COMPLEXITY: <SIMPLE or COMPLEX>
DOMAIN: <MATH, CODE, WRITING, RESEARCH, or GENERAL>

Request: {request}

Classification:"""

COMBINED_SPECIALIST_PROMPTS = {
    TaskDomain.MATH: """You are a mathematics expert. Solve problems step-by-step.
Question: {request}
Your solution:""",
    TaskDomain.CODE: """You are a programming expert. Write clean, well-documented code.
Request: {request}
Your code:""",
    TaskDomain.WRITING: """You are a writing expert. Create clear, engaging content.
Request: {request}
Your writing:""",
    TaskDomain.RESEARCH: """You are a research expert. Provide thorough, factual information.
Request: {request}
Your findings:""",
    TaskDomain.GENERAL: """You are a helpful assistant. Provide a clear, helpful response.
Request: {request}
Your response:""",
}

COMBINED_CRITIC_PROMPT = """Review this answer for quality.

Request: {request}
Answer: {answer}

Check for:
1. Accuracy - Is it factually correct?
2. Completeness - Does it fully address the request?
3. Clarity - Is it easy to understand?

If acceptable, respond: APPROVED
If not, respond: NEEDS_REVISION: <brief explanation>

Your review:"""


# =============================================================================
# AGENT SYSTEMS — each returns (answer, trace_events, api_calls)
# =============================================================================


def _call_model(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float = 0,
) -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=GenerateContentConfig(temperature=temperature),
    )
    return response.text.strip() if response.text else ""


def run_debate(
    client: genai.Client,
    model: str,
    question: str,
    max_rounds: int = 3,
) -> tuple[str, list[TraceEvent], int]:
    trace: list[TraceEvent] = []
    calls = 0

    # Initial proposal
    answer = _call_model(client, model, PROPOSER_INITIAL_PROMPT.format(question=question), 0.7)
    calls += 1
    trace.append(TraceEvent("Proposer", "Initial answer", answer))

    for round_num in range(1, max_rounds + 1):
        # Critic reviews
        critique = _call_model(
            client, model, DEBATE_CRITIC_PROMPT.format(question=question, answer=answer), 0.7
        )
        calls += 1
        approved = "APPROVED" in critique.upper()
        trace.append(TraceEvent(
            "Critic",
            f"Round {round_num} review",
            critique,
            {"approved": approved},
        ))

        if approved:
            break

        # Proposer revises
        answer = _call_model(
            client,
            model,
            PROPOSER_REVISE_PROMPT.format(
                question=question, previous_answer=answer, critique=critique,
            ),
            0.7,
        )
        calls += 1
        trace.append(TraceEvent("Proposer", f"Round {round_num} revision", answer))

    return answer, trace, calls


def run_router(
    client: genai.Client,
    model: str,
    question: str,
) -> tuple[str, list[TraceEvent], int]:
    trace: list[TraceEvent] = []
    calls = 0

    # Route
    route_response = _call_model(client, model, ROUTER_PROMPT.format(question=question), 0)
    calls += 1
    route_upper = route_response.upper().strip()

    specialist = SpecialistType.GENERAL
    for s in SpecialistType:
        if s.value.upper() in route_upper:
            specialist = s
            break

    specialist_name = f"{specialist.value.capitalize()} Specialist"
    trace.append(TraceEvent("Router", "Classified request", route_response, {"routed_to": specialist_name}))

    # Specialist answers
    prompt = SPECIALIST_PROMPTS[specialist].format(question=question)
    answer = _call_model(client, model, prompt, 0.7)
    calls += 1
    trace.append(TraceEvent(specialist_name, "Generated answer", answer))

    return answer, trace, calls


def run_orchestrator(
    client: genai.Client,
    model: str,
    request: str,
) -> tuple[str, list[TraceEvent], int]:
    trace: list[TraceEvent] = []
    calls = 0

    # Decompose
    decomposition = _call_model(client, model, DECOMPOSE_PROMPT.format(request=request), 0)
    calls += 1

    subtasks = []
    pattern = r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|END_SUBTASKS|$)"
    matches = re.findall(pattern, decomposition, re.DOTALL)
    for num_str, desc in matches:
        subtasks.append((int(num_str), desc.strip()))

    subtask_descs = [f"[{n}] {d}" for n, d in subtasks]
    trace.append(TraceEvent(
        "Orchestrator", "Decomposed into subtasks", "\n".join(subtask_descs),
        {"subtask_count": len(subtasks)},
    ))

    # Workers execute
    results = []
    for num, desc in subtasks:
        result = _call_model(client, model, WORKER_PROMPT.format(task=desc), 0.7)
        calls += 1
        trace.append(TraceEvent("Worker", f"Subtask {num}: {desc[:60]}", result))
        results.append(f"[Subtask {num}] {desc}\nResult: {result}")

    # Synthesize
    synth_prompt = SYNTHESIZE_PROMPT.format(
        original_request=request,
        subtask_results="\n\n".join(results),
    )
    answer = _call_model(client, model, synth_prompt, 0.7)
    calls += 1
    trace.append(TraceEvent("Synthesizer", "Combined results", answer))

    return answer, trace, calls


def run_combined(
    client: genai.Client,
    model: str,
    request: str,
    enable_critic: bool = True,
) -> tuple[str, list[TraceEvent], int]:
    trace: list[TraceEvent] = []
    calls = 0

    # Classify
    classify_response = _call_model(client, model, CLASSIFIER_PROMPT.format(request=request), 0)
    calls += 1

    complexity = TaskComplexity.SIMPLE
    if "COMPLEX" in classify_response.upper().split("COMPLEXITY:")[-1].split("DOMAIN:")[0]:
        complexity = TaskComplexity.COMPLEX

    domain = TaskDomain.GENERAL
    for d in TaskDomain:
        if d.value.upper() in classify_response.upper().split("DOMAIN:")[-1]:
            domain = d
            break

    trace.append(TraceEvent(
        "Classifier",
        "Classified request",
        classify_response,
        {"complexity": complexity.value, "domain": domain.value},
    ))

    # Route and execute
    if complexity == TaskComplexity.SIMPLE:
        specialist_name = f"{domain.value.capitalize()} Specialist"
        prompt = COMBINED_SPECIALIST_PROMPTS[domain].format(request=request)
        answer = _call_model(client, model, prompt, 0.7)
        calls += 1
        trace.append(TraceEvent(specialist_name, "Generated answer", answer))
    else:
        # Orchestrator/Worker path
        decomposition = _call_model(
            client, model, DECOMPOSE_PROMPT.format(request=request), 0
        )
        calls += 1

        subtasks = []
        pattern = r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|END_SUBTASKS|$)"
        matches = re.findall(pattern, decomposition, re.DOTALL)
        for num_str, desc in matches:
            subtasks.append((int(num_str), desc.strip()))

        subtask_descs = [f"[{n}] {d}" for n, d in subtasks]
        trace.append(TraceEvent(
            "Orchestrator", "Decomposed into subtasks", "\n".join(subtask_descs),
            {"subtask_count": len(subtasks)},
        ))

        results = []
        for num, desc in subtasks:
            result = _call_model(
                client, model, f"Complete this task:\n{desc}\n\nYour result:", 0.7
            )
            calls += 1
            trace.append(TraceEvent("Worker", f"Subtask {num}: {desc[:60]}", result))
            results.append(f"[{num}] {desc}\nResult: {result}")

        synth_prompt = SYNTHESIZE_PROMPT.format(
            original_request=request,
            subtask_results="\n\n".join(results),
        )
        answer = _call_model(client, model, synth_prompt, 0.7)
        calls += 1
        trace.append(TraceEvent("Synthesizer", "Combined results", answer))

    # Critic
    if enable_critic:
        critique_response = _call_model(
            client, model, COMBINED_CRITIC_PROMPT.format(request=request, answer=answer), 0
        )
        calls += 1
        approved = "APPROVED" in critique_response.upper()
        trace.append(TraceEvent(
            "Critic",
            "Quality review",
            critique_response,
            {"approved": approved},
        ))

    return answer, trace, calls


# =============================================================================
# UI HELPERS
# =============================================================================


def render_trace(trace_events: list[TraceEvent]) -> None:
    """Render trace events inside an expander."""
    for event in trace_events:
        icon = AGENT_ICONS.get(event.agent, "🤖")
        st.markdown(f"**{icon} {event.agent}** — {event.action}")

        # Show metadata as small tags
        if event.metadata:
            tags = " | ".join(f"**{k}:** {v}" for k, v in event.metadata.items())
            st.caption(tags)

        with st.container():
            st.markdown(event.content)

        st.divider()


def render_flow_summary(mode: str, trace_events: list[TraceEvent]) -> None:
    """Render a one-line summary of the path taken."""
    agents = [f"{AGENT_ICONS.get(e.agent, '🤖')} {e.agent}" for e in trace_events]
    st.info(" → ".join(agents))


# =============================================================================
# EXAMPLE QUERIES PER MODE
# =============================================================================

EXAMPLES = {
    "Debate": [
        "What are the pros and cons of remote work?",
        "Is AI a threat to junior developers?",
    ],
    "Specialist Router": [
        "What is 15% of 340?",
        "Write a Python function to reverse a string.",
        "Write a haiku about programming.",
        "What is the capital of France?",
    ],
    "Orchestrator/Worker": [
        "Plan a weekend trip to San Francisco with attractions, food, and transport tips.",
        "Compare electric vs hybrid cars for a city commuter.",
    ],
    "Combined System": [
        "What is 25% of 180?",
        "Compare the pros and cons of electric vs hybrid cars and recommend for a city commuter.",
    ],
}


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    st.set_page_config(page_title="Multi-Agent System", page_icon="🤝", layout="wide")

    st.title("🤝 Multi-Agent System Demo")
    st.markdown("Multiple agents working together: **Debate, Router, Orchestrator, and Combined**.")

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Create a .env file with your API key.")
        st.code("GOOGLE_API_KEY=your-key-here", language="bash")
        return

    # Initialize client
    if "client" not in st.session_state:
        st.session_state.client = genai.Client(api_key=api_key)

    # Sidebar
    with st.sidebar:
        st.header("Pattern")
        mode = st.radio(
            "Select a multi-agent pattern:",
            ["Debate", "Specialist Router", "Orchestrator/Worker", "Combined System"],
            label_visibility="collapsed",
        )

        # Mode-specific settings
        st.divider()
        max_rounds = 3
        enable_critic = True
        if mode == "Debate":
            max_rounds = st.slider("Max Debate Rounds", 1, 5, 3)
        elif mode == "Combined System":
            enable_critic = st.checkbox("Enable Critic", value=True)

        st.divider()

        st.header("Example Queries")
        for example in EXAMPLES[mode]:
            st.markdown(f"- {example}")

        st.divider()

        # Stats
        total_calls = st.session_state.get("total_api_calls", 0)
        st.metric("Total API Calls", total_calls)

        st.divider()
        st.caption("Built with Streamlit + Gemini")
        st.caption("Phase 4: Multi-Agent Systems")

    # Clear chat when mode changes
    if st.session_state.get("prev_mode") != mode:
        st.session_state.messages = []
        st.session_state.prev_mode = mode

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "total_api_calls" not in st.session_state:
        st.session_state.total_api_calls = 0

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "trace" in message and message["trace"]:
                with st.expander("🔀 Agent Flow"):
                    render_flow_summary(mode, message["trace"])
                    render_trace(message["trace"])

    # Chat input
    if prompt := st.chat_input("Ask me something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the selected pattern
        client = st.session_state.client
        model = "gemini-2.5-flash"

        with st.chat_message("assistant"):
            with st.spinner(f"Running {mode} pattern..."):
                if mode == "Debate":
                    answer, trace, calls = run_debate(client, model, prompt, max_rounds)
                elif mode == "Specialist Router":
                    answer, trace, calls = run_router(client, model, prompt)
                elif mode == "Orchestrator/Worker":
                    answer, trace, calls = run_orchestrator(client, model, prompt)
                else:
                    answer, trace, calls = run_combined(
                        client, model, prompt, enable_critic
                    )

            st.markdown(answer)

            st.session_state.total_api_calls += calls

            if trace:
                with st.expander("🔀 Agent Flow"):
                    render_flow_summary(mode, trace)
                    render_trace(trace)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "trace": trace,
        })

        st.rerun()


if __name__ == "__main__":
    main()
