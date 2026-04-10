# api/server.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — FastAPI Server
# ─────────────────────────────────────────────────────────────────────────────
#
# v1: bare FastAPI app, no lifespan, no scheduler
#
# v2 changes:
#   - lifespan() context manager manages APScheduler startup/shutdown
#   - APScheduler registers the 9am morning_brief job on startup
#   - /intelligence/run endpoint for manual pipeline trigger
#   - /intelligence/watchlist endpoints for topic management
#   - validate_env() called at module load (not just main.py)
#
# LIFESPAN PATTERN (FastAPI best practice):
#   @asynccontextmanager
#   async def lifespan(app):
#       scheduler.start()    ← runs on startup
#       yield
#       scheduler.shutdown() ← runs on shutdown (Ctrl+C or server kill)
#
#   This replaces the deprecated @app.on_event("startup") pattern.
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, field_validator

load_dotenv()

from tools.secrets_guard import validate_env, safe_log
validate_env("api/server.py")

from orchestrator.state import PersonalAIState
from tools.scheduler import get_scheduler, start_scheduler, stop_scheduler, schedule_daily_job


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — starts and stops APScheduler with the FastAPI server
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything BEFORE yield runs on startup.
    Everything AFTER yield runs on shutdown.

    On startup:
        1. Get the singleton scheduler instance
        2. Register the 9am morning brief job
        3. Start the scheduler

    On shutdown:
        4. Stop the scheduler gracefully
    """
    safe_log("[Server] Starting up — initialising scheduler...")

    # Import here (not at top) so validate_env() runs before graph imports
    from schedules.morning_brief import run_morning_brief
    from memory.store import init_db

    init_db()

    scheduler = get_scheduler()

    if scheduler is not None:
        # Read schedule from .env (defaults: 9am)
        hour   = int(os.getenv("BRIEF_SCHEDULE_HOUR",   "9"))
        minute = int(os.getenv("BRIEF_SCHEDULE_MINUTE", "0"))

        schedule_daily_job(
            scheduler=  scheduler,
            job_fn=     run_morning_brief,
            hour=       hour,
            minute=     minute,
            job_id=     "morning_brief",
            replace=    True,    # safe to restart server — won't duplicate job
        )

        start_scheduler(scheduler)
        safe_log(
            f"[Server] Morning brief scheduled: "
            f"daily at {hour:02d}:{minute:02d} local time"
        )

        # Store on app state so routes can access if needed
        app.state.scheduler = scheduler
    else:
        safe_log(
            "[Server] APScheduler unavailable — "
            "install apscheduler to enable scheduled briefs",
            level="WARN"
        )
        app.state.scheduler = None

    safe_log("[Server] Ready")
    yield  # ← server is running here

    # Shutdown
    safe_log("[Server] Shutting down...")
    stop_scheduler(app.state.scheduler)
    safe_log("[Server] Goodbye")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=       "IntelligenceOS",
    description= "PersonalAI v2 — Proactive knowledge engine with scheduled morning briefs.",
    version=     "2.0.0",
    lifespan=    lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED GRAPH BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_v1_graph():
    """v1 pipeline — search → research → publish (user-triggered)."""
    from orchestrator.graph import build_graph
    return build_graph()


@lru_cache(maxsize=1)
def get_v2_graph():
    """v2 pipeline — watchlist → graph → brief → alert (scheduled)."""
    from orchestrator.intelligence_graph import build_intelligence_graph
    return build_intelligence_graph()


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    task:             str
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
    status:           str
    task:             str
    research_summary: str
    thread:           list[str]
    sources_found:    int
    sources_used:     int
    error:            str | None


class WatchlistAddRequest(BaseModel):
    topic:           str
    check_frequency: str = "daily"

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Topic cannot be empty.")
        if len(value) > 200:
            raise ValueError("Topic must be 200 characters or fewer.")
        return value

    @field_validator("check_frequency")
    @classmethod
    def validate_frequency(cls, value: str) -> str:
        allowed = {"daily", "hourly", "manual"}
        if value not in allowed:
            raise ValueError(f"check_frequency must be one of: {allowed}")
        return value


class IntelligenceRunRequest(BaseModel):
    watchlist: list[str] | None = None  # None → reads from DB


class BriefResponse(BaseModel):
    status:          str
    delta_count:     int
    connection_count:int
    alert_sent:      bool
    brief_preview:   str    # first 500 chars of the brief
    error:           str | None


# ─────────────────────────────────────────────────────────────────────────────
# v1 ENDPOINTS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "IntelligenceOS", "version": "2.0.0"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """
    v1 pipeline: on-demand research for a single topic.
    search_agent → research_agent → publisher_agent
    """
    initial_state: PersonalAIState = {
        "task":             request.task,
        "post_to_telegram": request.post_to_telegram,
        "search_queries":   [],
        "search_results":   [],
        "scraped_content":  [],
        "research_summary": "",
        "thread":           [],
        "final_status":     "in_progress",
        "messages":         [],
        "error":            None,
        "current_agent":    "search_agent",
    }

    try:
        graph  = get_v1_graph()
        result = await asyncio.to_thread(graph.invoke, initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    return ResearchResponse(
        status=           result.get("final_status", "unknown"),
        task=             request.task,
        research_summary= result.get("research_summary", ""),
        thread=           result.get("thread") or result.get("twitter_thread", []),
        sources_found=    len(result.get("search_results", [])),
        sources_used=     len(result.get("scraped_content", [])),
        error=            result.get("error"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# v2 ENDPOINTS (IntelligenceOS)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/intelligence/run", response_model=BriefResponse)
async def run_intelligence(request: IntelligenceRunRequest) -> BriefResponse:
    """
    v2 pipeline: manually trigger the full IntelligenceOS pipeline.
    watchlist_agent → graph_agent → briefing_agent → alert_agent

    Optionally pass a watchlist to override the topics in the DB.
    If watchlist is null/omitted, reads from watchlist_topics table.

    This is the same function APScheduler calls at 9am —
    just triggered on-demand via HTTP instead of by the clock.
    """
    from orchestrator.intelligence_graph import build_initial_state

    state = build_initial_state(watchlist=request.watchlist)

    try:
        graph  = get_v2_graph()
        result = await asyncio.to_thread(graph.invoke, state)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Intelligence pipeline failed: {exc}"
        ) from exc

    brief    = result.get("formatted_brief", "")
    err      = result.get("error")

    return BriefResponse(
        status=           result.get("final_status", "unknown"),
        delta_count=      len(result.get("delta_events",    [])),
        connection_count= len(result.get("new_connections", [])),
        alert_sent=       result.get("alert_sent", False),
        brief_preview=    brief[:500],
        error=            err,
    )


@app.get("/intelligence/watchlist")
async def get_watchlist_topics() -> dict:
    """List all active watchlist topics."""
    from memory.store import get_watchlist
    topics = get_watchlist(active_only=False)
    return {
        "topics": [
            {
                "id":              t.id,
                "topic":           t.topic,
                "active":          t.active,
                "check_frequency": t.check_frequency,
                "last_checked":    t.last_checked,
                "created_at":      t.created_at,
            }
            for t in topics
        ],
        "count": len(topics),
    }


@app.post("/intelligence/watchlist", status_code=201)
async def add_watchlist_topic(request: WatchlistAddRequest) -> dict:
    """Add a new topic to the watchlist."""
    from memory.store import add_to_watchlist
    row_id = add_to_watchlist(request.topic, request.check_frequency)
    return {
        "added":           bool(row_id),
        "topic":           request.topic,
        "check_frequency": request.check_frequency,
        "message": (
            f"Topic '{request.topic}' added to watchlist"
            if row_id else
            f"Topic '{request.topic}' already exists"
        ),
    }


@app.delete("/intelligence/watchlist/{topic}")
async def remove_watchlist_topic(topic: str) -> dict:
    """Deactivate a watchlist topic (soft delete)."""
    from memory.store import remove_from_watchlist
    remove_from_watchlist(topic)
    return {"removed": True, "topic": topic}


@app.get("/intelligence/scheduler")
async def scheduler_status() -> dict:
    """Return current scheduler status and next job run time."""
    scheduler = getattr(app.state, "scheduler", None)

    if scheduler is None:
        return {"running": False, "reason": "APScheduler not available"}

    jobs = []
    for job in scheduler.get_jobs():
        next_run = job.next_run_time
        jobs.append({
            "id":       job.id,
            "name":     job.name,
            "next_run": next_run.isoformat() if next_run else None,
        })

    return {
        "running":  scheduler.running,
        "jobs":     jobs,
        "job_count":len(jobs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SERVER ENTRYPOINT  (python -m uvicorn api.server:app --reload)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    uvicorn.run(
        "api.server:app",
        host=        "0.0.0.0",
        port=        8000,
        reload=      True,
        reload_dirs= [str(project_root)],
    )
