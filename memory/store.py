# memory/store.py

import sqlite3
import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# Database lives in project root
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "personalai_memory.db"
)


@dataclass
class ResearchRun:
    """
    One complete research session.
    What we save after every pipeline run.
    """
    id:         int
    task:       str
    summary:    str
    thread:     list[str]
    status:     str
    sources:    int
    created_at: str


def init_db() -> None:
    """
    Creates the database and table if they don't exist.
    Safe to call multiple times — won't overwrite existing data.
    
    Like creating a notebook if you don't have one yet.
    If you already have one, does nothing.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_runs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task       TEXT    NOT NULL,
            summary    TEXT    NOT NULL,
            thread     TEXT    NOT NULL,
            status     TEXT    NOT NULL,
            sources    INTEGER NOT NULL DEFAULT 0,
            created_at TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"[Memory] Database ready at: {DB_PATH}")


def save_run(
    task:    str,
    summary: str,
    thread:  list[str],
    status:  str,
    sources: int = 0,
) -> int:
    """
    Saves one research run to database.
    Returns the ID of the saved run.

    Why return the ID?
    So the caller knows which row was created.
    Useful for linking follow-up actions to the run.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # thread is a list — JSON serialize it for storage
    # database stores text, not Python lists
    thread_json = json.dumps(thread)
    created_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO research_runs
            (task, summary, thread, status, sources, created_at)
        VALUES
            (?, ?, ?, ?, ?, ?)
    """, (task, summary, thread_json, status, sources, created_at))

    run_id = cursor.lastrowid     # ID of the row just inserted
    conn.commit()
    conn.close()

    print(f"[Memory] Saved run #{run_id}: '{task[:50]}...'")
    return run_id


def get_recent_runs(limit: int = 5) -> list[ResearchRun]:
    """
    Returns the most recent N research runs.
    Default: last 5 runs.

    Like flipping to the last 5 pages of your notebook.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, task, summary, thread, status, sources, created_at
        FROM research_runs
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [
        ResearchRun(
            id=         row[0],
            task=       row[1],
            summary=    row[2],
            thread=     json.loads(row[3]),   # deserialize JSON back to list
            status=     row[4],
            sources=    row[5],
            created_at= row[6],
        )
        for row in rows
    ]


def search_runs(query: str) -> list[ResearchRun]:
    """
    Searches past runs by task keyword.
    
    Example: search_runs("LangGraph")
    Returns all runs where task contained "LangGraph"

    SQL LIKE with % = contains search:
    WHERE task LIKE '%LangGraph%'
    means: task contains 'LangGraph' anywhere
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, task, summary, thread, status, sources, created_at
        FROM research_runs
        WHERE task    LIKE ?
           OR summary LIKE ?
        ORDER BY created_at DESC
        LIMIT 10
    """, (f"%{query}%", f"%{query}%"))

    rows = cursor.fetchall()
    conn.close()

    return [
        ResearchRun(
            id=         row[0],
            task=       row[1],
            summary=    row[2],
            thread=     json.loads(row[3]),
            status=     row[4],
            sources=    row[5],
            created_at= row[6],
        )
        for row in rows
    ]


def get_run_by_id(run_id: int) -> Optional[ResearchRun]:
    """
    Fetches one specific run by its ID.
    Returns None if not found.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, task, summary, thread, status, sources, created_at
        FROM research_runs
        WHERE id = ?
    """, (run_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return ResearchRun(
        id=         row[0],
        task=       row[1],
        summary=    row[2],
        thread=     json.loads(row[3]),
        status=     row[4],
        sources=    row[5],
        created_at= row[6],
    )


def format_run_for_display(run: ResearchRun) -> str:
    """
    Formats a ResearchRun for display in UI or terminal.
    Returns a clean markdown string.
    """
    thread_preview = ""
    if run.thread:
        thread_preview = f"\n\n**Thread preview:**\n{run.thread[0][:100]}..."

    return (
        f"**Run #{run.id}** — {run.created_at}\n"
        f"**Task:** {run.task}\n"
        f"**Status:** {run.status} | "
        f"**Sources:** {run.sources}"
        f"{thread_preview}"
    )


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize database
    init_db()

    # Save a test run
    run_id = save_run(
        task=    "test memory system",
        summary= "KEY FINDINGS: Memory works correctly",
        thread=  ["1/ Memory test post", "2/ Second post"],
        status=  "completed",
        sources= 5,
    )
    print(f"Saved test run with ID: {run_id}")

    # Retrieve recent runs
    recent = get_recent_runs(limit=3)
    print(f"\nRecent runs ({len(recent)}):")
    for run in recent:
        print(f"  #{run.id}: {run.task} — {run.created_at}")

    # Search
    results = search_runs("memory")
    print(f"\nSearch 'memory' found {len(results)} runs")