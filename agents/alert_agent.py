# agents/alert_agent.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — AlertAgent
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT THIS AGENT DOES
# ─────────────────────
# AlertAgent is the "last mile" of IntelligenceOS — it decides what
# gets sent to your Telegram, and when.
#
# It has TWO delivery modes:
#
#   MODE 1 — IMMEDIATE ALERT (if any topic scores >= threshold)
#     Sent first, before the morning brief.
#     Short, urgent. Example:
#       "🚨 ALERT — LangGraph v0.3 released (9.0/10)
#        Breaking changes to StateGraph API — affects NexusAI
#        → https://github.com/langchain-ai/langgraph/releases"
#
#   MODE 2 — MORNING BRIEF (always sent, after any alerts)
#     The full formatted_brief from BriefingAgent.
#     Telegram has a 4096-char limit per message — if the brief is longer,
#     AlertAgent splits it into multiple messages automatically.
#
# IMPORTANCE THRESHOLD:
#   Reads from env var ALERT_IMPORTANCE_THRESHOLD (default: 7.0)
#   Topics above this score get an immediate alert BEFORE the full brief.
#   Topics below just appear in the brief.
#
# TELEGRAM DELIVERY:
#   Reuses the existing tools/telegram_tool.py from v1.
#   post_thread() sends a list of messages sequentially.
#   AlertAgent builds the message list and calls post_thread().
#
# FAILURE HANDLING:
#   If Telegram fails → logs the failure, sets alert_sent=False.
#   The pipeline does NOT crash — brief is still saved in state.
#   Next run will retry automatically (APScheduler doesn't know about failure).
#
# FLOW:
#   importance_scores{} → filter topics >= ALERT_IMPORTANCE_THRESHOLD
#     ↓
#   For each high-importance topic:
#     build_alert_message() → short urgent Telegram message
#     post_thread([alert_message]) → send immediately
#     ↓
#   Split formatted_brief into ≤4096-char chunks
#   post_thread(chunks) → send full brief
#     ↓
#   state["alert_sent"] = True/False
#   state["final_status"] = "completed"
# ─────────────────────────────────────────────────────────────────────────────

import os
from datetime import datetime

from orchestrator.state import PersonalAIState
from tools.secrets_guard import safe_log
from tools.telegram_tool import PostFailure, PostSuccess, post_thread


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Telegram hard limit per message
TELEGRAM_MAX_CHARS = 4096

# Default threshold — topics above this score trigger an IMMEDIATE alert
# before the morning brief is sent. Override via .env ALERT_IMPORTANCE_THRESHOLD
DEFAULT_ALERT_THRESHOLD = 7.0

# Prefix for all alert messages
ALERT_PREFIX = "🚨 INTELLIGENCEOS ALERT"
BRIEF_PREFIX = "📋 INTELLIGENCEOS BRIEF"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NODE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def alert_agent_node(state: PersonalAIState) -> dict:
    """
    LangGraph node: delivers alerts and morning brief to Telegram.

    Reads from state:
        importance_scores  Dict[str,float]  — topic scores from WatchlistAgent
        delta_events       List[dict]       — event details for alert messages
        formatted_brief    str              — full brief from BriefingAgent

    Writes to state:
        alert_sent    bool   — True if at least one Telegram message was sent
        final_status  str    — "completed" always (failures are logged not fatal)
    """
    safe_log("[AlertAgent] Starting delivery check...")

    importance_scores = state.get("importance_scores", {})
    delta_events      = state.get("delta_events",      [])
    formatted_brief   = state.get("formatted_brief",   "")
    threshold         = _get_threshold()

    safe_log(f"[AlertAgent] Alert threshold: {threshold}/10")

    # ── Identify high-importance topics ───────────────────────────────────────
    high_importance_events = [
        event for event in delta_events
        if importance_scores.get(event.get("topic", ""), 0) >= threshold
    ]

    alert_sent = False

    # ── MODE 1: Send immediate alerts for urgent topics ───────────────────────
    if high_importance_events:
        safe_log(
            f"[AlertAgent] {len(high_importance_events)} topic(s) above threshold "
            f"({threshold}/10) — sending immediate alert(s)"
        )

        for event in high_importance_events:
            topic      = event.get("topic",      "Unknown")
            summary    = event.get("summary",    "")
            source_url = event.get("source_url", "")
            score      = importance_scores.get(topic, 0)

            alert_message = _build_alert_message(topic, summary, source_url, score)
            safe_log(f"[AlertAgent] Sending alert for '{topic}' ({score:.1f}/10)...")

            result = post_thread([alert_message])

            if isinstance(result, PostSuccess):
                safe_log(f"[AlertAgent] Alert sent for '{topic}'")
                alert_sent = True
            elif isinstance(result, PostFailure):
                safe_log(
                    f"[AlertAgent] Alert failed for '{topic}': "
                    f"{result.reason} — {result.detail}",
                    level="WARN"
                )
    else:
        safe_log(
            f"[AlertAgent] No topics above threshold ({threshold}/10) "
            f"— skipping immediate alerts"
        )

    # ── MODE 2: Send the full morning brief ───────────────────────────────────
    if formatted_brief:
        safe_log(
            f"[AlertAgent] Sending morning brief "
            f"({len(formatted_brief)} chars) to Telegram..."
        )

        # Split into Telegram-safe chunks if needed
        chunks = _split_for_telegram(formatted_brief, header=BRIEF_PREFIX)

        result = post_thread(chunks)

        if isinstance(result, PostSuccess):
            safe_log(
                f"[AlertAgent] Morning brief delivered "
                f"({result.count} message(s))"
            )
            alert_sent = True
        elif isinstance(result, PostFailure):
            safe_log(
                f"[AlertAgent] Brief delivery failed: "
                f"{result.reason} — {result.detail}",
                level="WARN"
            )
    else:
        safe_log("[AlertAgent] No formatted brief in state — skipping brief delivery", level="WARN")

    safe_log(
        f"[AlertAgent] Done — "
        f"alert_sent={alert_sent}, "
        f"high_importance={len(high_importance_events)} topic(s)"
    )

    return {
        "alert_sent":   alert_sent,
        "final_status": "completed",
        "current_agent": "end",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_alert_message(
    topic:      str,
    summary:    str,
    source_url: str,
    score:      float,
) -> str:
    """
    Build a short, urgent alert message for one high-importance topic.

    Format:
        🚨 INTELLIGENCEOS ALERT
        [9.0/10] LANGGRAPH RELEASES

        Breaking changes to StateGraph API — affects NexusAI...

        🔗 https://github.com/...
        ⏰ 09:04

    Kept under TELEGRAM_MAX_CHARS — never needs splitting.
    """
    now       = datetime.now().strftime("%H:%M")
    score_bar = _importance_bar(score)

    # Truncate summary to keep message concise
    short_summary = summary[:300].rstrip()
    if len(summary) > 300:
        short_summary += "..."

    lines = [
        f"{ALERT_PREFIX}",
        f"{score_bar} [{score:.1f}/10]  {topic.upper()}",
        "",
        short_summary,
    ]

    if source_url:
        lines += ["", f"🔗 {source_url}"]

    lines += [f"⏰ {now}"]

    return "\n".join(lines)


def _importance_bar(score: float) -> str:
    """Return a visual indicator based on importance score."""
    if score >= 9.0:
        return "🔴🔴🔴"
    elif score >= 8.0:
        return "🔴🔴"
    elif score >= 7.0:
        return "🔴"
    elif score >= 5.0:
        return "🟡"
    else:
        return "🟢"


def _split_for_telegram(
    text:   str,
    header: str = "",
) -> list[str]:
    """
    Split a long brief into Telegram-safe chunks (≤ TELEGRAM_MAX_CHARS each).

    Strategy:
        1. Try to split on the "━━━" section dividers (natural break points)
        2. If a section is still too long, split on double-newlines (paragraphs)
        3. If a paragraph is still too long, hard-split at TELEGRAM_MAX_CHARS

    Returns a list of strings, each safe to send as one Telegram message.
    The first chunk gets the BRIEF_PREFIX header prepended.
    """
    if not text:
        return [f"{header}\n\n(no brief content)"] if header else []

    # If it fits in one message (most common case), just send it
    full = f"{header}\n\n{text}" if header else text
    if len(full) <= TELEGRAM_MAX_CHARS:
        return [full]

    # Split on section dividers
    divider = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    sections = text.split(divider)

    chunks      : list[str] = []
    current     : str       = f"{header}\n\n" if header else ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        candidate = current + section + "\n\n" + divider + "\n"

        if len(candidate) <= TELEGRAM_MAX_CHARS:
            current = candidate
        else:
            # Save current chunk and start a new one
            if current.strip():
                chunks.append(current.strip())
            # If the section itself is too big, split on paragraphs
            if len(section) > TELEGRAM_MAX_CHARS:
                for para_chunk in _split_on_paragraphs(section):
                    chunks.append(para_chunk)
                current = ""
            else:
                current = section + "\n\n" + divider + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [full[:TELEGRAM_MAX_CHARS]]


def _split_on_paragraphs(text: str) -> list[str]:
    """Split text on double-newlines. Hard-cuts any paragraph still over limit."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = current + "\n\n" + para if current else para
        if len(candidate) <= TELEGRAM_MAX_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # Hard cut if single paragraph is still too long
            while len(para) > TELEGRAM_MAX_CHARS:
                chunks.append(para[:TELEGRAM_MAX_CHARS])
                para = para[TELEGRAM_MAX_CHARS:]
            current = para

    if current:
        chunks.append(current.strip())

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_threshold() -> float:
    """
    Read ALERT_IMPORTANCE_THRESHOLD from environment.
    Returns DEFAULT_ALERT_THRESHOLD if not set or invalid.
    """
    raw = os.getenv("ALERT_IMPORTANCE_THRESHOLD", "")
    if not raw:
        return DEFAULT_ALERT_THRESHOLD
    try:
        value = float(raw)
        return max(0.0, min(10.0, value))
    except ValueError:
        safe_log(
            f"[AlertAgent] Invalid ALERT_IMPORTANCE_THRESHOLD='{raw}' "
            f"— using default {DEFAULT_ALERT_THRESHOLD}",
            level="WARN"
        )
        return DEFAULT_ALERT_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python agents/alert_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    from tools.secrets_guard import validate_env
    validate_env("alert_agent self-test")

    # ── Test message builders (no Telegram call) ──────────────────────────────
    alert_msg = _build_alert_message(
        topic=      "LangGraph releases",
        summary=    "LangGraph v0.3 released with breaking changes to StateGraph API. All agents using add_conditional_edges() must migrate.",
        source_url= "https://github.com/langchain-ai/langgraph/releases",
        score=      9.0,
    )
    print("── Alert message ─────────────────────────────────────")
    print(alert_msg)
    print(f"\nLength: {len(alert_msg)} chars (limit: {TELEGRAM_MAX_CHARS})")
    assert len(alert_msg) <= TELEGRAM_MAX_CHARS, "Alert message exceeds Telegram limit!"
    print("Alert message length check ... OK")

    # ── Test telegram split for a long brief ──────────────────────────────────
    long_text = ("This is section content.\n" * 100)
    chunks = _split_for_telegram(long_text, header=BRIEF_PREFIX)
    print(f"\n── Telegram split test ───────────────────────────────")
    print(f"Input length  : {len(long_text)} chars")
    print(f"Chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk)} chars")
        assert len(chunk) <= TELEGRAM_MAX_CHARS, f"Chunk {i} exceeds Telegram limit!"
    print("Split test ... OK")

    # ── Test threshold reading ─────────────────────────────────────────────────
    threshold = _get_threshold()
    print(f"\n── Threshold ─────────────────────────────────────────")
    print(f"ALERT_IMPORTANCE_THRESHOLD = {threshold}/10")

    # ── Test full agent node (Telegram optional) ──────────────────────────────
    print("\n── AlertAgent node (brief only, no Telegram call) ────")
    print("  (Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID to test real delivery)")
    test_state: PersonalAIState = {
        "task":             "alert agent self-test",
        "post_to_telegram": False,
        "search_queries":   [],
        "search_results":   [],
        "scraped_content":  [],
        "research_summary": "",
        "thread":           [],
        "final_status":     "in_progress",
        "messages":         [],
        "error":            None,
        "current_agent":    "alert_agent",
        "delta_events": [
            {
                "topic":       "LangGraph releases",
                "summary":     "LangGraph v0.3 released with breaking changes.",
                "source_url":  "https://github.com/langchain-ai/langgraph",
                "importance":  9.0,
                "detected_at": "2026-04-11 09:00:00",
            }
        ],
        "importance_scores": {"LangGraph releases": 9.0},
        "last_checked":      {},
        "knowledge_graph":   {},
        "new_connections":   [],
        "formatted_brief":   "Test brief content — " + "x" * 50,
        "alert_sent":        False,
        "watchlist":         ["LangGraph releases"],
    }

    result = alert_agent_node(test_state)
    print(f"\n  alert_sent   : {result['alert_sent']}")
    print(f"  final_status : {result['final_status']}")
    print("\nAlertAgent self-test complete.")
