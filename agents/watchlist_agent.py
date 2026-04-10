# agents/watchlist_agent.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — WatchlistAgent
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT THIS AGENT DOES
# ─────────────────────
# WatchlistAgent is the "overnight security guard" of IntelligenceOS.
# It runs once at 9am (triggered by APScheduler) and checks every topic
# you've registered in the watchlist_topics table.
#
# For each topic it:
#   1. Searches Tavily for recent content on that topic
#   2. Compares results against the last_checked timestamp
#   3. Uses the LLM to decide: "Did anything actually change?"
#   4. If yes → creates a delta_event describing what changed
#   5. Scores the delta 0–10 for importance (AlertAgent reads this)
#   6. Stamps last_checked in the database
#
# WHY IT'S DIFFERENT FROM SearchAgent
# ─────────────────────────────────────
# SearchAgent runs ONCE per user request, on a single task.
# WatchlistAgent runs AUTOMATICALLY, on MULTIPLE topics, and
# specifically asks "what is NEW since I last checked?" —
# not "what exists about this topic in general?"
#
# The key concept: DELTA DETECTION.
# A delta is a change, not just information. The LLM's job here
# is to act as a filter: "Is this genuinely new, or just the same
# content I would have seen yesterday?"
#
# FLOW:
#   state["watchlist"] → [topic1, topic2, ...]
#     ↓ (for each topic)
#   Tavily search (time-filtered to recent)
#     ↓
#   LLM: "Is this a real change since {last_checked}? Score it."
#     ↓
#   delta_events.append({topic, summary, source_url, detected_at})
#   importance_scores[topic] = score
#     ↓
#   update_last_checked(topic) in SQLite
#     ↓
#   state["delta_events"], state["importance_scores"], state["last_checked"]
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
from datetime import datetime
from typing import Optional

from memory.store import get_watchlist, update_last_checked
from orchestrator.state import PersonalAIState
from tools.ollama_client import LLMInvocationError, invoke_prompt
from tools.secrets_guard import safe_log
from tools.web_search import WebSearchError, web_search


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# How many Tavily results to fetch per topic
RESULTS_PER_TOPIC = 5

# Default importance score when LLM fails to parse a number
DEFAULT_IMPORTANCE = 5.0

# Minimum importance to be included in delta_events at all
# (filters out purely trivial updates)
MIN_IMPORTANCE_THRESHOLD = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NODE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def watchlist_agent_node(state: PersonalAIState) -> dict:
    """
    LangGraph node: checks all watched topics for new developments.

    Reads from state:
        watchlist       List[str]  — topics to check (populated by caller)

    If watchlist is empty, reads directly from watchlist_topics SQLite table.
    This means it works both ways:
        - Scheduled run (9am): no watchlist in state → reads from DB
        - Manual run: caller can inject watchlist=['topic1','topic2'] into state

    Writes to state:
        delta_events       List[dict]     — what changed per topic
        importance_scores  Dict[str,float]— 0–10 score per topic
        last_checked       Dict[str,str]  — ISO timestamps per topic
    """
    safe_log("[WatchlistAgent] Starting watchlist check...")

    # ── Get the list of topics to check ──────────────────────────────────────
    # Priority: state["watchlist"] → SQLite watchlist_topics table
    watchlist: list[str] = list(state.get("watchlist", []))

    if not watchlist:
        safe_log("[WatchlistAgent] No watchlist in state — loading from database")
        db_topics = get_watchlist(active_only=True)
        watchlist = [t.topic for t in db_topics]

    if not watchlist:
        safe_log("[WatchlistAgent] Watchlist is empty — nothing to check", level="WARN")
        return {
            "delta_events":      [],
            "importance_scores": {},
            "last_checked":      {},
            "current_agent":     "graph_agent",
            "error":             "Watchlist is empty. Add topics with memory.store.add_to_watchlist()",
        }

    safe_log(f"[WatchlistAgent] Checking {len(watchlist)} topics: {watchlist}")

    # ── Check each topic ──────────────────────────────────────────────────────
    all_delta_events:   list[dict]       = []
    all_importance:     dict[str, float] = {}
    all_last_checked:   dict[str, str]   = {}

    for topic in watchlist:
        safe_log(f"[WatchlistAgent] ── Checking: '{topic}'")

        delta_event, importance = _check_topic(topic)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_last_checked[topic] = now

        # Stamp the database regardless of whether we found a delta
        try:
            update_last_checked(topic)
        except Exception as e:
            safe_log(f"[WatchlistAgent] Could not update last_checked for '{topic}': {e}", level="WARN")

        all_importance[topic] = importance

        if delta_event and importance >= MIN_IMPORTANCE_THRESHOLD:
            all_delta_events.append(delta_event)
            safe_log(
                f"[WatchlistAgent] Delta found for '{topic}' "
                f"(importance: {importance:.1f}/10)"
            )
        else:
            safe_log(
                f"[WatchlistAgent] No significant delta for '{topic}' "
                f"(importance: {importance:.1f}/10 — below threshold {MIN_IMPORTANCE_THRESHOLD})"
            )

    safe_log(
        f"[WatchlistAgent] Done — "
        f"{len(all_delta_events)} deltas found across {len(watchlist)} topics"
    )

    return {
        "delta_events":      all_delta_events,
        "importance_scores": all_importance,
        "last_checked":      all_last_checked,
        "watchlist":         watchlist,
        "current_agent":     "graph_agent",
        "error":             None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL: CHECK ONE TOPIC
# ─────────────────────────────────────────────────────────────────────────────

def _check_topic(topic: str) -> tuple[Optional[dict], float]:
    """
    Run the full check cycle for one topic:
        1. Search Tavily for recent content
        2. Ask LLM: is this new? what changed? how important?
        3. Return (delta_event or None, importance_score)

    Returns:
        (delta_event_dict, importance_score)
        delta_event is None if nothing notable was found.
    """
    # Step 1: search for this topic
    results = _search_topic(topic)

    if not results:
        safe_log(f"[WatchlistAgent]   No search results for '{topic}'", level="WARN")
        return None, 0.0

    # Step 2: ask LLM to evaluate the results
    delta_event, importance = _evaluate_with_llm(topic, results)

    return delta_event, importance


def _search_topic(topic: str) -> list[dict]:
    """
    Run Tavily search for a topic.
    Uses 'recent' biasing by appending the current year to the query
    to prefer fresh content over evergreen articles.

    Returns list of {title, url, snippet} dicts.
    Returns [] on any failure (graceful degradation).
    """
    current_year = datetime.now().year
    # Bias the query toward recent content
    query = f"{topic} {current_year} latest update news release"

    try:
        results = web_search(query, max_results=RESULTS_PER_TOPIC)
        safe_log(f"[WatchlistAgent]   Tavily returned {len(results)} results")
        return results
    except WebSearchError as e:
        safe_log(f"[WatchlistAgent]   Tavily search failed for '{topic}': {e}", level="WARN")
        return []


def _evaluate_with_llm(
    topic:   str,
    results: list[dict],
) -> tuple[Optional[dict], float]:
    """
    Ask Ollama to read the search results and decide:
      - Did anything actually change / is there genuinely new information?
      - How important is it (0–10)?
      - Summarise the delta in 2–3 sentences.

    Returns (delta_event_dict, importance_score).
    Returns (None, 0.0) if LLM is unavailable — fails gracefully.
    """
    # Format the search results for the LLM
    results_text = _format_results_for_llm(results)
    today        = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""You are an AI intelligence analyst monitoring technology topics.

TODAY'S DATE: {today}
TOPIC BEING MONITORED: {topic}

RECENT WEB SEARCH RESULTS:
{results_text}

Your task:
1. Analyse these results for GENUINELY NEW information about "{topic}"
2. Decide if there is a real update, release, change, or development worth noting
3. Score the importance from 0 to 10:
   - 0-2: Nothing new, same old information
   - 3-5: Minor update, interesting but not urgent
   - 6-7: Notable development, worth including in morning brief
   - 8-9: Significant change that may affect decisions
   - 10:  Critical breaking news requiring immediate attention

Respond in this EXACT JSON format (no extra text, no markdown):
{{
  "has_delta": true or false,
  "importance": 7.5,
  "summary": "2-3 sentence description of what changed and why it matters",
  "source_url": "the most relevant URL from the results above"
}}"""

    try:
        raw = invoke_prompt(prompt, temperature=0)
        return _parse_llm_response(topic, raw)

    except LLMInvocationError as e:
        safe_log(
            f"[WatchlistAgent]   LLM unavailable for '{topic}' — "
            f"using fallback heuristic: {e}",
            level="WARN"
        )
        # Fallback: if Ollama is down, create a basic delta from the top result
        return _fallback_delta(topic, results)


def _parse_llm_response(
    topic: str,
    raw:   str,
) -> tuple[Optional[dict], float]:
    """
    Parse the LLM's JSON response into a delta_event dict and importance score.

    Handles messy LLM output:
    - Extracts JSON even if surrounded by extra text
    - Falls back to default importance if number can't be parsed

    Returns (delta_event or None, importance_score).
    """
    # Strip markdown code fences if present (```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Try to extract JSON object from the response
    json_match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not json_match:
        safe_log(
            f"[WatchlistAgent]   Could not find JSON in LLM response for '{topic}'",
            level="WARN"
        )
        return None, DEFAULT_IMPORTANCE

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError:
        safe_log(
            f"[WatchlistAgent]   JSON parse failed for '{topic}' response",
            level="WARN"
        )
        return None, DEFAULT_IMPORTANCE

    has_delta  = bool(parsed.get("has_delta", False))
    importance = float(parsed.get("importance", DEFAULT_IMPORTANCE))
    summary    = str(parsed.get("summary", "No summary provided"))
    source_url = str(parsed.get("source_url", ""))

    importance = max(0.0, min(10.0, importance))   # clamp to 0–10

    if not has_delta:
        return None, importance

    delta_event = {
        "topic":       topic,
        "summary":     summary,
        "source_url":  source_url,
        "importance":  importance,
        "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return delta_event, importance


def _fallback_delta(
    topic:   str,
    results: list[dict],
) -> tuple[Optional[dict], float]:
    """
    When Ollama is unavailable, create a basic delta from the top search result.

    This ensures the morning brief still runs even if the local LLM is down.
    Importance is set to DEFAULT_IMPORTANCE (5.0) — conservative middle value.
    BriefingAgent will still describe it; AlertAgent won't trigger (below 7.0).
    """
    if not results:
        return None, 0.0

    top = results[0]
    summary = (
        f"[Fallback — LLM unavailable] "
        f"Recent content found for '{topic}': {top.get('title', 'Unknown title')}. "
        f"{top.get('snippet', '')[:200]}"
    )

    delta_event = {
        "topic":       topic,
        "summary":     summary,
        "source_url":  top.get("url", ""),
        "importance":  DEFAULT_IMPORTANCE,
        "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return delta_event, DEFAULT_IMPORTANCE


def _format_results_for_llm(results: list[dict]) -> str:
    """
    Format search results into a clean block of text for the LLM prompt.
    Keeps it concise — the LLM doesn't need full page content, just snippets.
    """
    lines = []
    for i, r in enumerate(results, 1):
        title   = r.get("title",   "No title")
        url     = r.get("url",     "No URL")
        snippet = r.get("snippet", "No content")[:300]
        lines.append(
            f"Result {i}:\n"
            f"  Title:   {title}\n"
            f"  URL:     {url}\n"
            f"  Snippet: {snippet}\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python agents/watchlist_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    from tools.secrets_guard import validate_env
    validate_env("watchlist_agent self-test")

    from memory.store import init_db, add_to_watchlist
    init_db()

    # Seed 2 test topics
    add_to_watchlist("LangGraph releases",  "daily")
    add_to_watchlist("Anthropic MCP spec",  "daily")

    # Build a minimal state with empty watchlist — agent reads from DB
    test_state: PersonalAIState = {
        "task":             "intelligence os watchlist check",
        "post_to_telegram": False,
        "search_queries":   [],
        "search_results":   [],
        "scraped_content":  [],
        "research_summary": "",
        "thread":           [],
        "final_status":     "in_progress",
        "messages":         [],
        "error":            None,
        "current_agent":    "watchlist_agent",
        # watchlist intentionally omitted — agent loads from DB
    }

    result = watchlist_agent_node(test_state)

    print("\n── WatchlistAgent Result ──────────────────────────────")
    print(f"  delta_events      : {len(result['delta_events'])} events")
    print(f"  importance_scores : {result['importance_scores']}")
    print(f"  last_checked      : {result['last_checked']}")
    print(f"  next agent        : {result['current_agent']}")

    if result["delta_events"]:
        print("\n── Delta Events ───────────────────────────────────────")
        for i, event in enumerate(result["delta_events"], 1):
            print(f"\n  Event {i}:")
            print(f"    Topic      : {event['topic']}")
            print(f"    Importance : {event['importance']}/10")
            print(f"    Summary    : {event['summary'][:150]}...")
            print(f"    Source     : {event['source_url']}")
    else:
        print("\n  No delta events found (all topics below threshold or no changes).")
