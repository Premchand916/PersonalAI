# PersonalAI - Your Personal Perplexity Computer

A locally running multi-agent AI system built with LangGraph and Ollama.

## Architecture
User -> Orchestrator (LangGraph) -> Search Agent -> Research Agent -> Publisher Agent -> Output

## Tech Stack
- Orchestrator: LangGraph StateGraph
- LLM: Ollama
- Web Search: Tavily API
- API: FastAPI
- Publisher: Telegram bot

## Setup
```bash
# 1. Clone the repo
git clone https://github.com/Premchand916/personalai.git
cd personalai

# 2. Create a virtual environment
pip install uv
uv venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Create a local environment file
cp .env.example .env           # Mac/Linux
copy .env.example .env         # Windows

# 5. Pull an Ollama model
ollama pull llama3.2:1b

# 6. Run the API
python -m uvicorn api.server:app --reload

# 7. Run tests
python -m unittest discover -s tests -v
```

## Environment Variables
Fill `.env` locally with:

- `TAVILY_API_KEY`
- `OLLAMA_MODEL`
- `OLLAMA_FALLBACK_MODEL`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

`.env` is ignored by Git. Commit `.env.example` instead of real secrets.
If `.env` was ever committed before, remove it from Git tracking with
`git rm --cached .env` and rotate the exposed secrets.
