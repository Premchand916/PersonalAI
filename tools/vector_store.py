# tools/vector_store.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — Qdrant Semantic Search
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT THIS FILE DOES
# ────────────────────
# Every time you research something, IntelligenceOS saves the summary to
# SQLite. That's keyword-searchable ("find runs with 'LangGraph' in them").
#
# But keyword search has a hard limit:
#   "LangGraph state machine" will NOT match "stateful agent orchestration"
#   even though they mean the same thing.
#
# Semantic search solves this. It works on MEANING, not words.
# You ask: "What have I researched that's similar to MCP tool calling?"
# Qdrant finds: runs about "LangGraph tool nodes", "agent action APIs",
#               "function calling patterns" — even if "MCP" never appeared.
#
# HOW IT WORKS (3-step pipeline):
#
#   STEP 1 — EMBED
#     sentence-transformers converts text → a 384-dim float vector.
#     Similar texts → similar vectors (close in vector space).
#     "LangGraph orchestration" and "stateful agent pipeline" get
#     vectors that are mathematically close together.
#
#   STEP 2 — STORE
#     Qdrant stores each vector alongside metadata (run_id, task, date).
#     Local mode (QdrantClient(path="./qdrant_db")) — no Docker, no server.
#     The entire vector DB is just a folder on your disk.
#
#   STEP 3 — SEARCH
#     Given a query, embed it → search Qdrant for closest vectors.
#     Returns top-K research runs ranked by semantic similarity.
#
# LOCAL MODE vs SERVER MODE:
#   This file uses local mode: QdrantClient(path=QDRANT_PATH)
#   Data persists in qdrant_db/ folder in project root.
#   No Docker. No external service. Works offline.
#   Handles ~20K vectors easily. More than enough for personal research history.
#
# MODEL USED: all-MiniLM-L6-v2
#   - 384 dimensions (small and fast)
#   - 80MB download, loads in ~2s
#   - Excellent for semantic similarity on short English texts
#   - Runs on CPU — no GPU needed
# ─────────────────────────────────────────────────────────────────────────────

import os
from dataclasses import dataclass
from typing import Optional

from tools.secrets_guard import safe_log

# ── Optional imports — graceful degradation if not installed ──────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    safe_log(
        "[VectorStore] qdrant-client not installed — semantic search disabled. "
        "Run: pip install qdrant-client",
        level="WARN"
    )

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    safe_log(
        "[VectorStore] sentence-transformers not installed — embedding disabled. "
        "Run: pip install sentence-transformers",
        level="WARN"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Vector DB lives in project root, alongside personalai_memory.db
QDRANT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "qdrant_db"
)

# Qdrant collection name — one collection holds all research run embeddings
COLLECTION_NAME = "research_runs"

# Embedding model — 384-dim, fast on CPU, great for short English text
EMBED_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# How many results to return by default from semantic_search()
DEFAULT_TOP_K = 5


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON CLIENTS — loaded once, reused across calls
# ─────────────────────────────────────────────────────────────────────────────

_qdrant_client:  Optional["QdrantClient"]      = None
_embed_model:    Optional["SentenceTransformer"] = None


def _get_qdrant() -> Optional["QdrantClient"]:
    """
    Get (or create) the singleton Qdrant local client.
    Creates the qdrant_db/ folder if it doesn't exist.
    Creates the research_runs collection if it doesn't exist.
    Returns None if qdrant-client is not installed.
    """
    global _qdrant_client
    if not _QDRANT_AVAILABLE:
        return None

    if _qdrant_client is None:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        _ensure_collection(_qdrant_client)
        safe_log(f"[VectorStore] Qdrant local client ready at: {QDRANT_PATH}")

    return _qdrant_client


def _get_model() -> Optional["SentenceTransformer"]:
    """
    Get (or load) the singleton sentence-transformers model.
    First call downloads ~80MB from HuggingFace (cached after that).
    Returns None if sentence-transformers is not installed.
    """
    global _embed_model
    if not _ST_AVAILABLE:
        return None

    if _embed_model is None:
        safe_log(f"[VectorStore] Loading embedding model '{EMBED_MODEL}'...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
        safe_log(f"[VectorStore] Model ready — {VECTOR_SIZE}-dim vectors")

    return _embed_model


def _ensure_collection(client: "QdrantClient") -> None:
    """
    Create the Qdrant collection if it doesn't already exist.
    Safe to call on every startup — no-ops if collection exists.

    Collection config:
        vectors: 384-dim floats
        distance: Cosine similarity
            → score of 1.0 = identical meaning
            → score of 0.0 = completely unrelated
    """
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=     VECTOR_SIZE,
                distance= Distance.COSINE,
            ),
        )
        safe_log(f"[VectorStore] Created collection '{COLLECTION_NAME}'")


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS — search result returned to callers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticSearchResult:
    """One result from a semantic search query."""
    run_id:     int
    task:       str
    summary:    str
    score:      float    # cosine similarity 0.0–1.0 (higher = more similar)
    created_at: str


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def embed_and_index(
    run_id:     int,
    task:       str,
    summary:    str,
    created_at: str,
) -> bool:
    """
    Embed a research run summary and store it in Qdrant.

    Called by memory/store.save_run() after every successful research run,
    so the vector index always stays in sync with SQLite.

    Args:
        run_id:     SQLite research_runs.id (used as Qdrant point ID)
        task:       The research task/query
        summary:    LLM-synthesised research summary
        created_at: ISO timestamp string

    Returns:
        True if successfully indexed, False on any failure.

    WHY embed task + summary together?
        The task gives the intent ("latest LangGraph features").
        The summary gives the content ("StateGraph, tool nodes, streaming").
        Combining them makes the vector represent BOTH what you wanted
        AND what you found. Better retrieval.
    """
    client = _get_qdrant()
    model  = _get_model()

    if client is None or model is None:
        safe_log(
            "[VectorStore] Skipping index — Qdrant or model unavailable",
            level="WARN"
        )
        return False

    try:
        # Combine task + summary for richer embedding
        text_to_embed = f"Task: {task}\n\nSummary: {summary}"

        # Embed → 384-dim float list
        vector = model.encode(text_to_embed, convert_to_numpy=True).tolist()

        # Build the Qdrant point
        point = PointStruct(
            id=      run_id,           # use SQLite run_id as point ID (unique int)
            vector=  vector,
            payload= {                 # metadata stored alongside the vector
                "run_id":     run_id,
                "task":       task[:500],
                "summary":    summary[:1000],
                "created_at": created_at,
            }
        )

        # Upsert — safe to call multiple times for same run_id
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point],
        )

        safe_log(f"[VectorStore] Indexed run #{run_id}: '{task[:60]}'")
        return True

    except Exception as e:
        safe_log(f"[VectorStore] Index failed for run #{run_id}: {e}", level="WARN")
        return False


def semantic_search(
    query:   str,
    top_k:   int = DEFAULT_TOP_K,
    min_score: float = 0.3,
) -> list[SemanticSearchResult]:
    """
    Find past research runs semantically similar to a query.

    Args:
        query:     Natural language query (not keyword — full sentence works best)
        top_k:     Maximum results to return (default: 5)
        min_score: Minimum cosine similarity to include (0.0–1.0, default: 0.3)
                   0.3 = loosely related, 0.7 = very similar, 1.0 = identical

    Returns:
        List of SemanticSearchResult, sorted by similarity score descending.
        Returns [] if Qdrant unavailable or no results above min_score.

    Example:
        results = semantic_search("stateful agent orchestration patterns")
        # Returns runs about: LangGraph, StateGraph, agent memory,
        #                     multi-agent pipelines — even without those exact words

    HOW SIMILARITY WORKS:
        Cosine similarity measures the ANGLE between two vectors.
        If your query vector and a stored run vector point in the same
        direction in 384-dimensional space, their angle is small → high score.
        "stateful agents" and "LangGraph StateGraph" learned to point
        in similar directions from training on millions of text pairs.
    """
    client = _get_qdrant()
    model  = _get_model()

    if client is None or model is None:
        safe_log(
            "[VectorStore] Semantic search unavailable — returning empty results",
            level="WARN"
        )
        return []

    if not query.strip():
        return []

    try:
        # Embed the query the same way we embedded the documents
        query_vector = model.encode(query, convert_to_numpy=True).tolist()

        # Search Qdrant for nearest neighbours
        hits = client.search(
            collection_name= COLLECTION_NAME,
            query_vector=    query_vector,
            limit=           top_k,
            score_threshold= min_score,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(SemanticSearchResult(
                run_id=     payload.get("run_id",     hit.id),
                task=       payload.get("task",       ""),
                summary=    payload.get("summary",    ""),
                score=      round(hit.score, 4),
                created_at= payload.get("created_at", ""),
            ))

        safe_log(
            f"[VectorStore] Query: '{query[:60]}' → "
            f"{len(results)} result(s) above score {min_score}"
        )
        return results

    except Exception as e:
        safe_log(f"[VectorStore] Search failed: {e}", level="WARN")
        return []


def index_all_runs() -> int:
    """
    Bulk-index all existing research_runs from SQLite into Qdrant.

    Call this ONCE after first installing Qdrant, to backfill the
    vector index with your existing research history.
    After that, new runs are indexed automatically via embed_and_index()
    inside save_run().

    Returns the number of runs successfully indexed.
    """
    client = _get_qdrant()
    model  = _get_model()

    if client is None or model is None:
        safe_log("[VectorStore] Cannot bulk-index — Qdrant or model unavailable", level="WARN")
        return 0

    # Import here to avoid circular import (store.py imports vector_store.py)
    from memory.store import get_recent_runs

    # get_recent_runs with a very high limit to get all runs
    runs = get_recent_runs(limit=10_000)

    safe_log(f"[VectorStore] Bulk indexing {len(runs)} research runs...")
    indexed = 0

    for run in runs:
        if run.status == "failed":
            continue  # Don't index failed runs — no useful summary
        success = embed_and_index(
            run_id=     run.id,
            task=       run.task,
            summary=    run.summary,
            created_at= run.created_at,
        )
        if success:
            indexed += 1

    safe_log(f"[VectorStore] Bulk index complete — {indexed}/{len(runs)} runs indexed")
    return indexed


def collection_info() -> dict:
    """
    Return stats about the Qdrant collection.
    Useful for the /intelligence/search endpoint's health check.
    """
    client = _get_qdrant()
    if client is None:
        return {"available": False, "reason": "qdrant-client not installed"}

    try:
        info = client.get_collection(COLLECTION_NAME)
        return {
            "available":   True,
            "collection":  COLLECTION_NAME,
            "vector_count": info.points_count,
            "vector_size": VECTOR_SIZE,
            "model":       EMBED_MODEL,
            "path":        QDRANT_PATH,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python tools/vector_store.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("Testing vector_store.py ...\n")
    print(f"Qdrant available      : {_QDRANT_AVAILABLE}")
    print(f"sentence-transformers : {_ST_AVAILABLE}")

    if not (_QDRANT_AVAILABLE and _ST_AVAILABLE):
        print("\nInstall missing dependencies:")
        if not _QDRANT_AVAILABLE:
            print("  pip install qdrant-client")
        if not _ST_AVAILABLE:
            print("  pip install sentence-transformers")
        sys.exit(0)

    # ── Index 3 synthetic research runs ───────────────────────────────────────
    print("\n── Indexing test runs ────────────────────────────────")
    runs = [
        (1, "LangGraph v0.3 new features",
         "KEY FINDINGS: LangGraph released StateGraph improvements. Tool nodes added. Streaming enhanced. Anthropic contributed MCP integration. Breaking changes to add_conditional_edges()."),
        (2, "CrewAI vs LangGraph agent orchestration",
         "KEY FINDINGS: CrewAI uses role-based agent design. LangGraph offers finer state control. Both support multi-agent workflows. LangGraph preferred for complex stateful pipelines."),
        (3, "Qdrant vector database local deployment",
         "KEY FINDINGS: Qdrant supports local mode without Docker. QdrantClient(path=...) creates file-based DB. Cosine similarity works well for text embeddings. sentence-transformers integrates easily."),
    ]

    for run_id, task, summary in runs:
        ok = embed_and_index(run_id, task, summary, "2026-04-11 09:00:00")
        print(f"  Run #{run_id}: {'OK' if ok else 'FAILED'} — {task[:50]}")

    # ── Semantic search ────────────────────────────────────────────────────────
    print("\n── Semantic search tests ─────────────────────────────")

    queries = [
        ("stateful agent orchestration pipeline",     "Should match runs 1 and 2"),
        ("vector database embedding similarity",       "Should match run 3"),
        ("tool calling protocol API integration",      "Should match run 1 (MCP/tools)"),
    ]

    for query, expectation in queries:
        print(f"\nQuery: '{query}'")
        print(f"Expect: {expectation}")
        results = semantic_search(query, top_k=3, min_score=0.2)
        if results:
            for r in results:
                print(f"  [{r.score:.3f}] Run #{r.run_id}: {r.task[:55]}")
        else:
            print("  No results found")

    # ── Collection info ────────────────────────────────────────────────────────
    print("\n── Collection info ───────────────────────────────────")
    info = collection_info()
    for k, v in info.items():
        print(f"  {k:15}: {v}")

    print("\nAll vector_store tests complete.")
