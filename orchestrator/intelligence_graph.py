# orchestrator/intelligence_graph.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — v2 LangGraph Pipeline
# ─────────────────────────────────────────────────────────────────────────────
#
# HOW THIS RELATES TO orchestrator/graph.py (v1)
# ─────────────────────────────────────────────────
# v1 graph.py:
#     search_agent → research_agent → publisher_agent → END
#     Triggered by: user request
#     Purpose: research one topic on demand
#
# v2 intelligence_graph.py (this file):
#     watchlist_agent → graph_agent → briefing_agent → alert_agent → END
#     Triggered by: APScheduler at 9am (or manual test run)
#     Purpose: check all topics, build knowledge graph, deliver morning brief
#
# Both graphs use the SAME PersonalAIState TypedDict.
# v1 agents read/write v1 fields (task, thread, research_summary, etc.)
# v2 agents read/write v2 fields (watchlist, delta_events, knowledge_graph, etc.)
# They are separate compiled graphs — not merged into one.
#
# PIPELINE FLOW (Sessions 2–6, built incrementally):
#
#   Session 2 ✅:  watchlist_agent (REAL)
#   Session 3 ✅:  graph_agent     (REAL)
#   Session 5 ✅:  briefing_agent  (REAL — Jinja2 template)
#   Session 6 ✅:  alert_agent     (REAL — Telegram delivery)
#   Session 7:     hooked into APScheduler for 9am auto-run
#
# ROUTING LOGIC
# ──────────────
# After WatchlistAgent: always proceed to GraphAgent
# After GraphAgent:     always proceed to BriefingAgent
# After BriefingAgent:  always proceed to AlertAgent
# After AlertAgent:     END
#
# No conditional routing yet — that comes in Session 6 when AlertAgent
# decides between "send alert now" vs "just log and continue".
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import END, StateGraph

from agents.alert_agent import alert_agent_node
from agents.briefing_agent import briefing_agent_node
from agents.graph_agent import graph_agent_node
from agents.watchlist_agent import watchlist_agent_node
from orchestrator.state import PersonalAIState
from tools.secrets_guard import safe_log


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_intelligence_graph():
    """
    Compile and return the IntelligenceOS v2 LangGraph pipeline.

    Node order:
        watchlist_agent → graph_agent → briefing_agent → alert_agent → END

    Sessions 3–6 will swap stub nodes for real implementations.
    The graph shape stays the same — only the node functions change.

    Usage:
        from orchestrator.intelligence_graph import build_intelligence_graph

        graph  = build_intelligence_graph()
        result = graph.invoke(initial_state)

    For async streaming (Chainlit UI):
        async for chunk in graph.astream(initial_state):
            ...
    """
    graph = StateGraph(PersonalAIState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("watchlist_agent",  watchlist_agent_node)  # Session 2 — REAL
    graph.add_node("graph_agent",      graph_agent_node)       # Session 3 — REAL
    graph.add_node("briefing_agent",   briefing_agent_node)    # Session 5 — REAL
    graph.add_node("alert_agent",      alert_agent_node)       # Session 6 — REAL

    # ── Set entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("watchlist_agent")

    # ── Wire edges (sequential, no branching yet) ─────────────────────────────
    graph.add_edge("watchlist_agent", "graph_agent")
    graph.add_edge("graph_agent",     "briefing_agent")
    graph.add_edge("briefing_agent",  "alert_agent")
    graph.add_edge("alert_agent",     END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL STATE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_initial_state(watchlist: list[str] | None = None) -> PersonalAIState:
    """
    Build the starting state for an IntelligenceOS pipeline run.

    Args:
        watchlist: Optional list of topics to override the DB watchlist.
                   If None, WatchlistAgent reads directly from SQLite.

    All v1 fields get safe defaults so the shared State TypedDict is valid.
    All v2 fields that aren't set here will be populated by the agents.
    """
    state: PersonalAIState = {
        # ── v1 fields (required by TypedDict) ────────────────────────────────
        "task":             "intelligenceos morning watchlist check",
        "post_to_telegram": False,   # AlertAgent controls Telegram delivery
        "search_queries":   [],
        "search_results":   [],
        "scraped_content":  [],
        "research_summary": "",
        "thread":           [],
        "final_status":     "in_progress",
        "messages":         [],
        "error":            None,
        "current_agent":    "watchlist_agent",

        # ── v2 fields (set here if known up front) ────────────────────────────
        "delta_events":      [],
        "importance_scores": {},
        "last_checked":      {},
        "knowledge_graph":   {},
        "new_connections":   [],
        "formatted_brief":   "",
        "alert_sent":        False,
    }

    # Only add watchlist if explicitly provided
    # (If absent, WatchlistAgent reads from DB — the standard scheduled path)
    if watchlist is not None:
        state["watchlist"] = watchlist

    return state


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python orchestrator/intelligence_graph.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    from tools.secrets_guard import validate_env
    from memory.store import init_db, add_to_watchlist

    validate_env("intelligence_graph self-test")
    init_db()

    # Seed topics for the test
    add_to_watchlist("LangGraph releases",  "daily")
    add_to_watchlist("Anthropic MCP spec",  "daily")

    safe_log("\n[Test] Building intelligence graph...")
    graph = build_intelligence_graph()

    safe_log("[Test] Building initial state (watchlist from DB)...")
    state = build_initial_state()   # no watchlist arg → reads from DB

    safe_log("[Test] Running pipeline...\n")
    result = graph.invoke(state)

    print("\n" + "=" * 60)
    print("  INTELLIGENCEOS PIPELINE RESULT")
    print("=" * 60)
    print(f"  final_status    : {result.get('final_status')}")
    print(f"  delta_events    : {len(result.get('delta_events', []))} events")
    print(f"  importance      : {result.get('importance_scores', {})}")
    print(f"  alert_sent      : {result.get('alert_sent', False)}")
    print(f"  brief length    : {len(result.get('formatted_brief', ''))} chars")

    brief = result.get("formatted_brief", "")
    if brief:
        print("\n── Morning Brief Preview ─────────────────────────────")
        print(brief[:600])
        if len(brief) > 600:
            print("  ... [truncated]")

    print("=" * 60)
