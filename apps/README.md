# Agent Apps

Demo applications built on top of the tool-calling agent.

## streamlit_phase1.py

Web UI for the Phase 1 tool-calling agent.

### Run locally

```bash
cd apps
uv sync
uv run streamlit run streamlit_phase1.py
```

Opens at http://localhost:8501

### Deploy to Streamlit Cloud (free)

1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set `apps/streamlit_phase1.py` as the main file
5. Add `GOOGLE_API_KEY` as a secret

### Features

- Chat interface with history
- Shows tool calls in expandable sections
- Available tools: calculator, weather, time
