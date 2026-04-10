# tools/scheduler.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — APScheduler Wrapper
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY A WRAPPER AROUND APSCHEDULER?
# ──────────────────────────────────
# APScheduler v3 has several gotchas that cause silent failures in production:
#   - Timezone not set → jobs fire at wrong time after DST change
#   - max_instances not set → overlapping 9am runs pile up
#   - Memory store → jobs lost on server restart
#   - BackgroundScheduler in async FastAPI → blocks the event loop
#
# This wrapper encapsulates all the correct configuration in ONE place.
# Everywhere else in the codebase just calls:
#
#   from tools.scheduler import get_scheduler, start_scheduler, stop_scheduler
#
# And gets a correctly-configured AsyncIOScheduler with:
#   - System local timezone
#   - SQLite job store (jobs survive restarts)
#   - max_instances=1 (no overlapping runs)
#   - coalesce=True (if missed, run once not many times)
#
# VERSION NOTE:
#   APScheduler v3.x (3.11.2+) — current stable release.
#   APScheduler v4 is pre-release — NOT used here.
#   v3 uses: AsyncIOScheduler, scheduled_job decorator, SQLAlchemyJobStore
# ─────────────────────────────────────────────────────────────────────────────

import os
from functools import lru_cache

from tools.secrets_guard import safe_log

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.executors.asyncio import AsyncIOExecutor
    _APScheduler_AVAILABLE = True
except ImportError:
    _APScheduler_AVAILABLE = False
    safe_log(
        "[Scheduler] apscheduler not installed — scheduled jobs disabled. "
        "Run: pip install apscheduler",
        level="WARN"
    )

# Job store DB lives in project root alongside personalai_memory.db
_SCHEDULER_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scheduler_jobs.db"
)


@lru_cache(maxsize=1)
def get_scheduler() -> "AsyncIOScheduler | None":
    """
    Build and return the singleton APScheduler instance.

    Configuration choices explained:
        AsyncIOScheduler  → runs in FastAPI's async event loop (not a thread)
        SQLAlchemyJobStore → persists jobs to SQLite; survives restarts
        coalesce=True     → if server was down at 9am, run ONCE on restart
                            (not once per missed interval)
        misfire_grace_time=3600 → allow up to 1hr late (handles slow startup)
        max_instances=1   → never allow two 9am runs to overlap

    Returns None if apscheduler is not installed (graceful degradation).
    """
    if not _APScheduler_AVAILABLE:
        return None

    jobstores = {
        "default": SQLAlchemyJobStore(url=f"sqlite:///{_SCHEDULER_DB}")
    }

    executors = {
        "default": AsyncIOExecutor()
    }

    job_defaults = {
        "coalesce":          True,    # missed run → execute once, not many
        "max_instances":     1,       # no overlapping runs
        "misfire_grace_time": 3600,   # up to 1hr late is still acceptable
    }

    scheduler = AsyncIOScheduler(
        jobstores=    jobstores,
        executors=    executors,
        job_defaults= job_defaults,
        # No timezone arg → uses system local time (correct for cron "9am")
    )

    safe_log("[Scheduler] AsyncIOScheduler configured with SQLite job store")
    return scheduler


def start_scheduler(scheduler: "AsyncIOScheduler") -> None:
    """
    Start the scheduler. Call this from FastAPI lifespan on startup.
    Safe to call even if scheduler is already running (checks first).
    """
    if scheduler is None:
        safe_log("[Scheduler] Scheduler not available — skipping start", level="WARN")
        return

    if scheduler.running:
        safe_log("[Scheduler] Scheduler already running — skipped")
        return

    scheduler.start()
    safe_log("[Scheduler] Started — jobs will fire on schedule")


def stop_scheduler(scheduler: "AsyncIOScheduler") -> None:
    """
    Gracefully shut down the scheduler.
    Call this from FastAPI lifespan on shutdown.
    wait=False means it won't block for running jobs to complete.
    """
    if scheduler is None:
        return

    if not scheduler.running:
        return

    scheduler.shutdown(wait=False)
    safe_log("[Scheduler] Stopped")


def schedule_daily_job(
    scheduler:    "AsyncIOScheduler",
    job_fn:       callable,
    hour:         int = 9,
    minute:       int = 0,
    job_id:       str = "morning_brief",
    replace:      bool = True,
) -> None:
    """
    Register a daily cron job on the scheduler.

    Args:
        scheduler : The AsyncIOScheduler instance from get_scheduler()
        job_fn    : Async function to call (must be async def)
        hour      : Hour to fire in LOCAL time (default: 9 → 9am)
        minute    : Minute to fire (default: 0 → :00)
        job_id    : Unique ID for the job (prevents duplicate registration)
        replace   : If True, replace an existing job with same ID

    How to read the cron config:
        trigger="cron", hour=9, minute=0
        → "run once per day, at 09:00 in local system time"
        → equivalent to Linux cron: 0 9 * * *

    WHY job_id matters:
        Without it, every server restart adds a SECOND copy of the job.
        After 7 restarts → 7 identical 9am runs. Very bad.
        job_id + replace=True means: "there can be only one."
    """
    if scheduler is None:
        safe_log("[Scheduler] Cannot register job — scheduler unavailable", level="WARN")
        return

    kwargs = dict(
        func=        job_fn,
        trigger=     "cron",
        hour=        hour,
        minute=      minute,
        id=          job_id,
        replace_existing= replace,
    )

    scheduler.add_job(**kwargs)

    safe_log(
        f"[Scheduler] Job '{job_id}' registered — "
        f"fires daily at {hour:02d}:{minute:02d} local time"
    )
