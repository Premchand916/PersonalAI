# schedules/morning_brief.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — 9am Morning Brief Job
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT THIS FILE IS
# ──────────────────
# This is the "alarm clock" of IntelligenceOS.
# It contains ONE async function — run_morning_brief() — that:
#   1. Validates all secrets are present
#   2. Initialises the database
#   3. Builds the IntelligenceOS v2 pipeline
#   4. Invokes it (watchlist → graph → brief → alert)
#   5. Logs the result
#   6. Never crashes (all exceptions caught — an 8am failure must not
#      kill the server until the next run)
#
# HOW IT GETS CALLED
# ───────────────────
# APScheduler calls run_morning_brief() at 9am every day.
# The scheduler is registered in api/server.py via FastAPI lifespan.
#
# You can also call it manually any time (for testing):
#   python schedules/morning_brief.py
#
# EXECUTION CONTEXT
# ──────────────────
# run_morning_brief() is an ASYNC function.
# APScheduler's AsyncIOExecutor runs it inside FastAPI's event loop.
# graph.invoke() is synchronous (LangGraph doesn't have native async yet),
# so we wrap it with asyncio.to_thread() to avoid blocking the event loop.
#
# WHY NOT graph.astream()?
# ─────────────────────────
# For the scheduled brief, we don't need streaming — there's no UI watching.
# graph.invoke() gives us the final result in one call.
# astream() is for the Chainlit UI where you want real-time updates.
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import sys
import io
from datetime import datetime

# UTF-8 fix must be FIRST — before any other imports that might print
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from tools.secrets_guard import validate_env, safe_log
from memory.store import init_db
from orchestrator.intelligence_graph import build_intelligence_graph, build_initial_state


# ─────────────────────────────────────────────────────────────────────────────
# THE JOB FUNCTION — APScheduler calls this at 9am
# ─────────────────────────────────────────────────────────────────────────────

async def run_morning_brief() -> dict:
    """
    The core IntelligenceOS job — runs once per day at 9am.

    Designed to be:
        - SAFE: all exceptions caught, never kills the server
        - LOGGED: every step logged with safe_log() (no print, no secrets)
        - FAST: graph.invoke() offloaded to thread (non-blocking)
        - IDEMPOTENT: safe to run manually or re-trigger after failure

    Returns the final state dict from the pipeline.
    Returns {} on any unrecoverable error.
    """
    run_start = datetime.now()
    safe_log(
        f"\n{'='*55}\n"
        f"[MorningBrief] Starting at {run_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*55}"
    )

    try:
        # ── Step 1: Validate all secrets are present ──────────────────────────
        # If a key is missing, validate_env() calls sys.exit(1).
        # In a scheduled job, we catch this separately so the scheduler
        # doesn't crash — it just skips this run and logs the failure.
        validate_env("morning_brief.py")

    except SystemExit as e:
        safe_log(
            f"[MorningBrief] ABORTED — secrets validation failed (exit code {e.code}). "
            f"Check your .env file.",
            level="ERROR"
        )
        return {}

    try:
        # ── Step 2: Ensure database tables exist ──────────────────────────────
        init_db()

        # ── Step 3: Build the pipeline ────────────────────────────────────────
        # build_intelligence_graph() compiles the LangGraph StateGraph.
        # We rebuild it each run (not cached) so any code changes to agents
        # take effect without restarting the server.
        safe_log("[MorningBrief] Building IntelligenceOS pipeline...")
        graph = build_intelligence_graph()

        # ── Step 4: Build starting state ──────────────────────────────────────
        # No watchlist arg → WatchlistAgent reads from watchlist_topics table.
        # This is the standard path: the DB controls what gets monitored.
        state = build_initial_state()

        safe_log("[MorningBrief] Running pipeline (watchlist → graph → brief → alert)...")

        # ── Step 5: Run the pipeline ───────────────────────────────────────────
        # asyncio.to_thread() moves the synchronous graph.invoke() call into
        # a thread pool worker, freeing the event loop during the run.
        # Typical runtime: 30–120 seconds (depends on watchlist size + LLM speed)
        result = await asyncio.to_thread(graph.invoke, state)

        # ── Step 6: Log the outcome ───────────────────────────────────────────
        run_end     = datetime.now()
        elapsed     = (run_end - run_start).total_seconds()
        delta_count = len(result.get("delta_events",      []))
        conn_count  = len(result.get("new_connections",   []))
        alert_sent  = result.get("alert_sent",  False)
        final_status= result.get("final_status", "unknown")
        brief_len   = len(result.get("formatted_brief",   ""))

        safe_log(
            f"\n{'='*55}\n"
            f"[MorningBrief] COMPLETE in {elapsed:.1f}s\n"
            f"  Status      : {final_status}\n"
            f"  Deltas found: {delta_count}\n"
            f"  Connections : {conn_count}\n"
            f"  Brief length: {brief_len} chars\n"
            f"  Alert sent  : {alert_sent}\n"
            f"{'='*55}"
        )

        return result

    except Exception as e:
        run_end = datetime.now()
        elapsed = (run_end - run_start).total_seconds()
        safe_log(
            f"[MorningBrief] FAILED after {elapsed:.1f}s — {type(e).__name__}: {e}",
            level="ERROR"
        )
        # Return empty dict — scheduler continues running, next 9am will retry
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL RUN ENTRY POINT  (python schedules/morning_brief.py)
# ─────────────────────────────────────────────────────────────────────────────
# Useful for:
#   - Testing the full pipeline without waiting for 9am
#   - Running ad-hoc on demand from PowerShell
#   - Debugging a specific agent or topic

if __name__ == "__main__":
    print("\nIntelligenceOS — Manual Morning Brief Run")
    print("=" * 45)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("(This is the same job that APScheduler runs at 9am)")
    print()

    # asyncio.run() is the correct way to call an async function
    # from a synchronous __main__ on Windows (works without ProactorEventLoop hack)
    result = asyncio.run(run_morning_brief())

    if result:
        print("\n── Pipeline result summary ─────────────────────────")
        print(f"  final_status : {result.get('final_status')}")
        print(f"  delta_events : {len(result.get('delta_events', []))} found")
        print(f"  alert_sent   : {result.get('alert_sent', False)}")
        brief = result.get("formatted_brief", "")
        if brief:
            print(f"\n── Brief preview (first 800 chars) ─────────────────")
            print(brief[:800])
            if len(brief) > 800:
                print(f"  ... [{len(brief)-800} more chars]")
    else:
        print("\n[!] Pipeline returned no result — check logs above for errors.")
