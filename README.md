# PersonalAI — Your Personal Perplexity Computer

A locally-running multi-agent AI system built with LangGraph + Ollama.
Inspired by Perplexity Computer architecture.

## Architecture
User → Orchestrator (LangGraph) → [Search Agent → Research Agent → Twitter Agent] → Output

## Tech Stack
- Orchestrator: LangGraph StateGraph
- LLM: Ollama (llama3.1:8b) — runs 100% locally
- Web Search: Tavily API
- API: FastAPI (coming)
- UI: Chainlit (coming)

## Setup
```bash
# 1. Clone the repo
git clone https://github.com/Premchand916/personalai.git
cd personalai

# 2. Create virtual environment
pip install uv
uv venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Copy env file and fill in keys
cp .env.example .env

# 5. Pull Ollama model
ollama pull llama3.1:8b

# 6. Run
python main.py
```

## Sessions Completed
- [x] Session 1: State + Graph Architecture
- [x] Session 2: Search Agent (live web search)
- [ ] Session 3: Research Agent
- [ ] Session 4: Twitter Agent
- [ ] Session 5: FastAPI + Chainlit UI
