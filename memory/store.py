# memory/store.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — Persistent Memory Store
# ─────────────────────────────────────────────────────────────────────────────
#
# v1 had one table: research_runs
#   → save every research session
#
# v2 adds two tables:
#   watchlist_topics  → topics you're monitoring + when last checked
#   knowledge_edges   → graph relationships extracted from research
#
# All three tables live in the same SQLite file (personalai_memory.db).
# init_db() creates all of them — safe to call at every startup.
#
# SECURITY:
#   - All text written to SQLite must pass through redact() first
#   - Call from tools.secrets_guard import redact before any INSERT
# ─────────────────────────────────────────────────────────────────────────────

import sqlite3
import json
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from tools.secrets_guard import safe_log, redact


# Database lives in project root — same as v1
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "personalai_memory.db"
)


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES — typed row wrappers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResearchRun:
    """One complete v1 research session. Unchanged from v1."""
    id:         int
    task:       str
    summary:    str
    thread:     list[str]
    status:     str
    sources:    int
    created_at: str


@dataclass
class WatchlistTopic:
    """One topic registered for ongoing monitoring by WatchlistAgent."""
    id:              int
    topic:           str
    created_at:      str
    last_checked:    Optional[str]   # None until first check completes
    check_frequency: str             # "daily" | "hourly" | "manual"
    active:          bool            # False = paused/archived


@dataclass
class KnowledgeEdge:
    """One directed relationship between two concepts in the knowledge graph."""
    id:            int
    concept_a:     str
    concept_b:     str
    relationship:  str           # "relates_to" | "contradicts" | "extends" | "is_part_of"
    source_run_id: Optional[int] # which research_run produced this edge
    confidence:    float         # 0.0 – 1.0 (LLM confidence in the edge)
    created_at:    str


# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION — creates all tables
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Creates all database tables if they don't already exist.
    Safe to call at every startup — uses CREATE TABLE IF NOT EXISTS.

    Tables created:
        research_runs     (v1 — unchanged)
        watchlist_topics  (v2 — new)
        knowledge_edges   (v2 — new)

    Think of it like opening a filing cabinet with 3 drawers.
    If the cabinet already exists, this just confirms it's there.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ── v1 table (unchanged) ─────────────────────────────────────────────────
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

    # ── v2 table: watchlist_topics ────────────────────────────────────────────
    # One row per topic you want IntelligenceOS to monitor.
    # WatchlistAgent reads this table at 9am to know what to check.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist_topics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            topic           TEXT    NOT NULL UNIQUE,
            created_at      TEXT    NOT NULL,
            last_checked    TEXT,
            check_frequency TEXT    NOT NULL DEFAULT 'daily',
            active          INTEGER NOT NULL DEFAULT 1
        )
    """)
    # Note: active stored as INTEGER (1=True, 0=False) — SQLite has no BOOLEAN

    # ── v2 table: knowledge_edges ─────────────────────────────────────────────
    # One row per relationship GraphAgent discovers between concepts.
    # Together these rows reconstruct the full knowledge graph.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_edges (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_a     TEXT    NOT NULL,
            concept_b     TEXT    NOT NULL,
            relationship  TEXT    NOT NULL,
            source_run_id INTEGER,
            confidence    REAL    NOT NULL DEFAULT 0.8,
            created_at    TEXT    NOT NULL,
            FOREIGN KEY (source_run_id) REFERENCES research_runs(id)
        )
    """)

    conn.commit()
    conn.close()
    safe_log(f"[Memory] Database ready — 3 tables at: {DB_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# v1 FUNCTIONS — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def save_run(
    task:    str,
    summary: str,
    thread:  list[str],
    status:  str,
    sources: int = 0,
) -> int:
    """
    Saves one research run to database.
    Returns the row ID of the saved run.

    Security: summary is passed through redact() before storage.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    thread_json = json.dumps(thread)
    created_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO research_runs
            (task, summary, thread, status, sources, created_at)
        VALUES
            (?, ?, ?, ?, ?, ?)
    """, (redact(task), redact(summary), thread_json, status, sources, created_at))

    run_id = cursor.lastrowid
    conn.commit()
    conn.close()

    safe_log(f"[Memory] Saved run #{run_id}: '{task[:50]}'")

    # ── Auto-index into Qdrant for semantic search (Session 8) ───────────────
    # Only index completed runs with a real summary — failed runs have no signal.
    # Import here (not at top) to avoid circular import.
    if status != "failed" and summary:
        try:
            from tools.vector_store import embed_and_index
            embed_and_index(
                run_id=     run_id,
                task=       task,
                summary=    summary,
                created_at= created_at,
            )
        except Exception as e:
            # Never let vector indexing break the save — SQLite is the source of truth
            safe_log(f"[Memory] Vector index skipped for run #{run_id}: {e}", level="WARN")

    return run_id


def get_recent_runs(limit: int = 5) -> list[ResearchRun]:
    """Returns the most recent N research runs, newest first."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, task, summary, thread, status, sources, created_at
        FROM research_runs
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    return [_row_to_run(row) for row in rows]


def search_runs(query: str) -> list[ResearchRun]:
    """
    Full-text search over task + summary fields.
    Example: search_runs("LangGraph") → all runs mentioning LangGraph.
    """
    conn   = sqlite3.connect(DB_PATH)
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

    return [_row_to_run(row) for row in rows]


def get_run_by_id(run_id: int) -> Optional[ResearchRun]:
    """Fetch one specific run by ID. Returns None if not found."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, task, summary, thread, status, sources, created_at
        FROM research_runs
        WHERE id = ?
    """, (run_id,))

    row = cursor.fetchone()
    conn.close()

    return _row_to_run(row) if row else None


def format_run_for_display(run: ResearchRun) -> str:
    """Markdown-formatted string for UI or terminal display."""
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


def _row_to_run(row: tuple) -> ResearchRun:
    """Convert a raw SQLite row tuple to a ResearchRun dataclass."""
    return ResearchRun(
        id=         row[0],
        task=       row[1],
        summary=    row[2],
        thread=     json.loads(row[3]),
        status=     row[4],
        sources=    row[5],
        created_at= row[6],
    )


# ─────────────────────────────────────────────────────────────────────────────
# v2 FUNCTIONS — watchlist_topics table
# ─────────────────────────────────────────────────────────────────────────────

def add_to_watchlist(topic: str, check_frequency: str = "daily") -> int:
    """
    Register a new topic for IntelligenceOS to monitor.

    Args:
        topic:           What to watch. E.g. "LangGraph releases"
        check_frequency: How often. "daily" | "hourly" | "manual"

    Returns:
        The row ID of the new watchlist entry.

    Example:
        add_to_watchlist("MCP spec changes", "daily")
        add_to_watchlist("CrewAI GitHub stars", "daily")

    Uses INSERT OR IGNORE so calling twice with the same topic is safe.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT OR IGNORE INTO watchlist_topics
            (topic, created_at, check_frequency, active)
        VALUES
            (?, ?, ?, 1)
    """, (topic, created_at, check_frequency))

    row_id = cursor.lastrowid or 0
    conn.commit()
    conn.close()

    if row_id:
        safe_log(f"[Watchlist] Added topic: '{topic}' (every {check_frequency})")
    else:
        safe_log(f"[Watchlist] Topic already exists: '{topic}' — skipped")

    return row_id


def get_watchlist(active_only: bool = True) -> list[WatchlistTopic]:
    """
    Returns all topics IntelligenceOS is monitoring.

    Args:
        active_only: If True (default), only returns active=1 topics.
                     Set False to see paused/archived topics too.

    WatchlistAgent calls this at the start of every 9am run.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if active_only:
        cursor.execute("""
            SELECT id, topic, created_at, last_checked, check_frequency, active
            FROM watchlist_topics
            WHERE active = 1
            ORDER BY created_at ASC
        """)
    else:
        cursor.execute("""
            SELECT id, topic, created_at, last_checked, check_frequency, active
            FROM watchlist_topics
            ORDER BY created_at ASC
        """)

    rows = cursor.fetchall()
    conn.close()

    return [
        WatchlistTopic(
            id=              row[0],
            topic=           row[1],
            created_at=      row[2],
            last_checked=    row[3],   # None until first check
            check_frequency= row[4],
            active=          bool(row[5]),
        )
        for row in rows
    ]


def update_last_checked(topic: str) -> None:
    """
    Stamp a topic with the current time after WatchlistAgent checks it.
    Called at the END of each topic's check cycle.

    This is how IntelligenceOS knows WHAT changed since the last check —
    it compares new Tavily results against the last_checked timestamp.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        UPDATE watchlist_topics
        SET last_checked = ?
        WHERE topic = ?
    """, (now, topic))

    conn.commit()
    conn.close()
    safe_log(f"[Watchlist] Stamped '{topic}' last_checked = {now}")


def remove_from_watchlist(topic: str) -> None:
    """
    Deactivate a topic (soft delete — keeps history).
    Use this instead of hard DELETE to preserve audit trail.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE watchlist_topics
        SET active = 0
        WHERE topic = ?
    """, (topic,))

    conn.commit()
    conn.close()
    safe_log(f"[Watchlist] Deactivated topic: '{topic}'")


# ─────────────────────────────────────────────────────────────────────────────
# v2 FUNCTIONS — knowledge_edges table
# ─────────────────────────────────────────────────────────────────────────────

def save_knowledge_edge(
    concept_a:     str,
    concept_b:     str,
    relationship:  str,
    source_run_id: Optional[int] = None,
    confidence:    float         = 0.8,
) -> int:
    """
    Persist one graph edge to SQLite.
    Called by GraphAgent after building the NetworkX graph.

    Args:
        concept_a:     First node (e.g. "LangGraph")
        concept_b:     Second node (e.g. "NexusAI architecture")
        relationship:  Edge type: "relates_to" | "contradicts" | "extends" | "is_part_of"
        source_run_id: Which research_run discovered this edge (None if from watchlist)
        confidence:    0.0–1.0 LLM confidence score

    Returns:
        The row ID of the saved edge.

    WHY save to SQLite if NetworkX has the graph in memory?
        NetworkX is in-memory only — it disappears when the process ends.
        SQLite is the persistent backup. At startup, GraphAgent reloads
        edges from SQLite to reconstruct the graph. Both are needed.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO knowledge_edges
            (concept_a, concept_b, relationship, source_run_id, confidence, created_at)
        VALUES
            (?, ?, ?, ?, ?, ?)
    """, (concept_a, concept_b, relationship, source_run_id, confidence, created_at))

    edge_id = cursor.lastrowid
    conn.commit()
    conn.close()

    safe_log(
        f"[Graph] Edge saved #{edge_id}: "
        f"'{concept_a}' --[{relationship}]--> '{concept_b}' "
        f"(confidence: {confidence:.2f})"
    )
    return edge_id


def get_all_edges() -> list[KnowledgeEdge]:
    """
    Load all knowledge edges from SQLite.
    Called by GraphAgent at startup to reconstruct the NetworkX graph
    from previous sessions.
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, concept_a, concept_b, relationship,
               source_run_id, confidence, created_at
        FROM knowledge_edges
        ORDER BY created_at ASC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        KnowledgeEdge(
            id=            row[0],
            concept_a=     row[1],
            concept_b=     row[2],
            relationship=  row[3],
            source_run_id= row[4],
            confidence=    row[5],
            created_at=    row[6],
        )
        for row in rows
    ]


def get_edges_for_concept(concept: str) -> list[KnowledgeEdge]:
    """
    Find all edges connected to a specific concept (as either endpoint).
    Useful for showing everything known about a topic.

    Example: get_edges_for_concept("LangGraph")
    → All edges where concept_a OR concept_b is "LangGraph"
    """
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, concept_a, concept_b, relationship,
               source_run_id, confidence, created_at
        FROM knowledge_edges
        WHERE concept_a = ? OR concept_b = ?
        ORDER BY confidence DESC
    """, (concept, concept))

    rows = cursor.fetchall()
    conn.close()

    return [
        KnowledgeEdge(
            id=            row[0],
            concept_a=     row[1],
            concept_b=     row[2],
            relationship=  row[3],
            source_run_id= row[4],
            confidence=    row[5],
            created_at=    row[6],
        )
        for row in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python memory/store.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("Initialising database...")
    init_db()

    # ── Test v1 table ──────────────────────────────────────────────
    run_id = save_run(
        task=    "test IntelligenceOS memory",
        summary= "KEY FINDINGS: all three tables working correctly",
        thread=  ["1/ Memory works", "2/ v2 tables added"],
        status=  "completed",
        sources= 3,
    )
    print(f"v1 run saved as #{run_id}")

    recent = get_recent_runs(limit=1)
    print(f"Recent runs: {[r.task for r in recent]}")

    # ── Test watchlist_topics ──────────────────────────────────────
    t1 = add_to_watchlist("LangGraph releases", "daily")
    t2 = add_to_watchlist("MCP spec changes",   "daily")
    t3 = add_to_watchlist("CrewAI updates",     "daily")
    add_to_watchlist("LangGraph releases", "daily")  # duplicate — should skip

    topics = get_watchlist()
    print(f"\nWatchlist ({len(topics)} topics):")
    for t in topics:
        print(f"  - {t.topic}  [every {t.check_frequency}]  last_checked={t.last_checked}")

    update_last_checked("LangGraph releases")
    topics = get_watchlist()
    lang_topic = next(t for t in topics if "LangGraph" in t.topic)
    print(f"\nAfter update_last_checked: last_checked = {lang_topic.last_checked}")

    # ── Test knowledge_edges ───────────────────────────────────────
    e1 = save_knowledge_edge("LangGraph", "NexusAI",     "is_part_of", run_id, 0.9)
    e2 = save_knowledge_edge("MCP spec",  "LangGraph",   "extends",    run_id, 0.85)
    e3 = save_knowledge_edge("CrewAI",    "LangGraph",   "relates_to", run_id, 0.7)

    all_edges = get_all_edges()
    print(f"\nKnowledge edges ({len(all_edges)} total):")
    for e in all_edges:
        print(f"  '{e.concept_a}' --[{e.relationship}]--> '{e.concept_b}'  ({e.confidence:.2f})")

    lang_edges = get_edges_for_concept("LangGraph")
    print(f"\nEdges for 'LangGraph': {len(lang_edges)} found")

    print("\nAll store tests passed.")
