"""
01_multi_agent_concept.py - Introduction to Multi-Agent Systems

PURPOSE:
    Understand WHY and WHEN to use multiple agents instead of one.

KEY QUESTION:
    Why not just use one powerful agent?

MULTI-AGENT PATTERNS:

    1. DEBATE / CRITIQUE
       ┌─────────┐     ┌─────────┐
       │ Agent A │ ←→  │ Agent B │
       │ (Answer)│     │(Critique)│
       └─────────┘     └─────────┘

       - Agent A proposes an answer
       - Agent B critiques/verifies it
       - They iterate until consensus
       - Use case: Fact-checking, code review, decision validation

    2. SPECIALIST / ROUTER
       ┌─────────────┐
       │   Router    │
       │   Agent     │
       └──────┬──────┘
              │ Routes to specialist
       ┌──────┼──────┬──────┐
       ▼      ▼      ▼      ▼
    ┌─────┐┌─────┐┌─────┐┌─────┐
    │Math ││Code ││Write││Data │
    │Agent││Agent││Agent││Agent│
    └─────┘└─────┘└─────┘└─────┘

       - Router analyzes the task
       - Delegates to the appropriate specialist
       - Use case: Customer support, diverse skill requirements

    3. ORCHESTRATOR / WORKER
       ┌─────────────┐
       │ Orchestrator│
       │   (Plans)   │
       └──────┬──────┘
              │ Assigns tasks
       ┌──────┼──────┐
       ▼      ▼      ▼
    ┌─────┐┌─────┐┌─────┐
    │Work ││Work ││Work │
    │ er1 ││ er2 ││ er3 │
    └─────┘└─────┘└─────┘

       - Orchestrator creates a plan
       - Workers execute individual tasks
       - Orchestrator combines results
       - Use case: Complex multi-step projects

    4. HIERARCHICAL
       ┌─────────────┐
       │   Manager   │
       └──────┬──────┘
              │
       ┌──────┴──────┐
       ▼             ▼
    ┌─────┐      ┌─────┐
    │Team │      │Team │
    │Lead1│      │Lead2│
    └──┬──┘      └──┬──┘
       │            │
    ┌──┴──┐      ┌──┴──┐
    ▼     ▼      ▼     ▼
   W1    W2     W3    W4

       - Multiple levels of delegation
       - Each level has different responsibilities
       - Use case: Large-scale projects, organizational simulation

WHY MULTI-AGENT?

    1. SPECIALIZATION
       - Each agent can have focused instructions
       - Smaller, more precise prompts
       - Less confusion from conflicting requirements

    2. VERIFICATION
       - Agents can check each other's work
       - Reduces hallucination through critique
       - Multiple perspectives on a problem

    3. DIVIDE AND CONQUER
       - Break complex tasks into simpler subtasks
       - Each agent handles what it's good at
       - Parallel execution possible

    4. SEPARATION OF CONCERNS
       - Planning agent doesn't need tool access
       - Execution agent doesn't need to plan
       - Cleaner, more maintainable systems

WHEN NOT TO USE MULTI-AGENT:

    1. Simple tasks - overhead not worth it
    2. Tight latency requirements - more agents = more API calls
    3. Cost constraints - each agent call costs money
    4. When single agent performs well enough

TRADE-OFFS:

    Single Agent:
    + Simpler to build and debug
    + Faster (fewer API calls)
    + Cheaper
    - May struggle with complex/diverse tasks
    - No verification

    Multi-Agent:
    + Better at complex tasks
    + Built-in verification
    + Specialization
    - More complex to build
    - Slower (more API calls)
    - More expensive
    - Coordination overhead

RUN:
    uv run python 01_multi_agent_concept.py
"""


def main():
    print("=" * 60)
    print("MULTI-AGENT SYSTEMS - CONCEPT OVERVIEW")
    print("=" * 60)

    print("""
This script is conceptual - no code to run.

KEY PATTERNS:

1. DEBATE/CRITIQUE
   Two agents argue/verify until consensus.
   Example: One writes code, another reviews it.

2. SPECIALIST/ROUTER
   Router picks the best agent for the job.
   Example: Math questions → Math Agent, Writing → Writing Agent.

3. ORCHESTRATOR/WORKER
   One agent plans, others execute.
   Example: Manager breaks down project, workers do subtasks.

4. HIERARCHICAL
   Multiple levels of management.
   Example: CEO → Directors → Teams → Workers.

NEXT: In 02_debate_agents.py, we'll build a debate system
where two agents critique each other's answers.
""")

    print("=" * 60)
    print("Continue to 02_debate_agents.py when ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
