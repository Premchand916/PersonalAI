import asyncio
from functools import lru_cache
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, field_validator

from orchestrator.state import PersonalAIState

load_dotenv()

app = FastAPI(
    title="PersonalAI",
    description="Your personal research assistant for collecting and publishing findings.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_graph():
    from orchestrator.graph import build_graph

    return build_graph()


class ResearchRequest(BaseModel):
    task: str
    post_to_telegram: bool = True

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Task cannot be empty.")
        if len(value) > 500:
            raise ValueError("Task must be 500 characters or fewer.")
        return value


class ResearchResponse(BaseModel):
    status: str
    task: str
    research_summary: str
    thread: list[str]
    sources_found: int
    sources_used: int
    error: str | None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "PersonalAI"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    initial_state: PersonalAIState = {
        "task": request.task,
        "post_to_telegram": request.post_to_telegram,
        "search_queries": [],
        "search_results": [],
        "scraped_content": [],
        "research_summary": "",
        "thread": [],
        "final_status": "in_progress",
        "messages": [],
        "error": None,
        "current_agent": "search_agent",
    }

    try:
        graph = get_graph()
        result = await asyncio.to_thread(graph.invoke, initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    return ResearchResponse(
        status=result.get("final_status", "unknown"),
        task=request.task,
        research_summary=result.get("research_summary", ""),
        thread=result.get("thread") or result.get("twitter_thread", []),
        sources_found=len(result.get("search_results", [])),
        sources_used=len(result.get("scraped_content", [])),
        error=result.get("error"),
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root)],
    )
