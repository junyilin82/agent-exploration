# Phase 5: Evaluation & Guardrails

How do you know your agent is working? How do you prevent misuse?

## Scripts

| Script | Topic | Description |
|--------|-------|-------------|
| 01_eval_concepts.py | Concepts | Why agent eval is hard, types of evaluation |
| 02_llm_as_judge.py | LLM-as-Judge | Use one LLM call to score another's output |
| 03_test_harness.py | Test Harness | Run agent against test suite, aggregate scores |
| 04_input_guardrails.py | Input Guardrails | Detect prompt injection, off-topic, unsafe input |
| 05_output_guardrails.py | Output Guardrails | Check hallucination, format, safety on output |

## Key Concepts

### Evaluation
- **Unit-level**: Test individual components (tool calls, parsing)
- **End-to-end**: Test full agent on question-answer pairs
- **LLM-as-Judge**: Use an LLM to score another LLM's output
- **Human eval**: Gold standard but expensive

### Guardrails
- **Input guardrails**: Validate before the agent runs (reject bad input)
- **Output guardrails**: Validate after the agent responds (catch bad output)
- **Defense in depth**: Combine both for a guarded pipeline

## Running

```bash
cd phase_5
uv sync
uv run python 01_eval_concepts.py
```
