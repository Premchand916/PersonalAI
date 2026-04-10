# agents/briefing_agent.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — BriefingAgent
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT THIS AGENT DOES
# ─────────────────────
# BriefingAgent is the "editor-in-chief" of IntelligenceOS.
# It receives everything the previous two agents produced and synthesises
# it into a single, readable morning brief that tells you:
#
#   1. Executive summary  — "Here's the one-paragraph version of your morning"
#   2. Key deltas         — colour-coded by importance (🔴🟡🟢)
#   3. New connections    — surprising bridges the GraphAgent found
#   4. Action items       — what you should actually DO today
#   5. Stats              — graph size, topics watched, deltas found
#
# WHY JINJA2?
# ────────────
# Jinja2 is the gold standard for templated text generation.
# You define the STRUCTURE once in morning_brief.j2 (the template).
# BriefingAgent just fills in the variables.
#
# This separation matters:
#   - Non-engineers can edit the brief format (it's just a text file)
#   - A/B testing different formats = editing one file, not Python code
#   - Telegram has character limits → template enforces conciseness
#   - The LLM is only used for the executive summary + action items
#     (creative synthesis), NOT for the layout (deterministic template)
#
# DATA FLOW:
#   state["delta_events"]      → template variable: delta_events
#   state["importance_scores"] → template variable: importance_scores
#   state["new_connections"]   → template variable: new_connections
#   state["knowledge_graph"]   → extracts node/edge counts for stats
#     ↓
#   LLM → executive_summary (2-3 sentences)
#   LLM → action_items      (3-5 bullet points)
#     ↓
#   Jinja2 Environment.render() → formatted_brief (str)
#     ↓
#   state["formatted_brief"]
# ─────────────────────────────────────────────────────────────────────────────

import os
from datetime import datetime

from orchestrator.state import PersonalAIState
from tools.ollama_client import LLMInvocationError, invoke_prompt
from tools.secrets_guard import safe_log

try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False
    safe_log(
        "[BriefingAgent] jinja2 not installed — will use fallback plain-text format. "
        "Run: pip install Jinja2",
        level="WARN"
    )

# Template lives in templates/ relative to project root
TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates"
)
TEMPLATE_FILE = "morning_brief.j2"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NODE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def briefing_agent_node(state: PersonalAIState) -> dict:
    """
    LangGraph node: renders the complete morning brief.

    Reads from state:
        delta_events       List[dict]      — what changed (WatchlistAgent)
        importance_scores  Dict[str,float] — topic importance 0–10
        new_connections    List[dict]      — surprising graph bridges (GraphAgent)
        knowledge_graph    dict            — node-link graph dict (for stats)
        watchlist          List[str]       — watched topics list

    Writes to state:
        formatted_brief    str             — complete morning brief text
    """
    safe_log("[BriefingAgent] Generating morning brief...")

    delta_events     = state.get("delta_events",      [])
    importance_scores= state.get("importance_scores", {})
    new_connections  = state.get("new_connections",   [])
    knowledge_graph  = state.get("knowledge_graph",   {})
    watchlist        = state.get("watchlist",         [])

    # ── Extract graph stats ───────────────────────────────────────────────────
    graph_nodes = len(knowledge_graph.get("nodes", []))
    graph_edges = len(knowledge_graph.get("links", []))

    # ── Generate executive summary via LLM ────────────────────────────────────
    executive_summary = _generate_executive_summary(
        delta_events, new_connections, importance_scores
    )

    # ── Generate action items via LLM ─────────────────────────────────────────
    action_items = _generate_action_items(
        delta_events, new_connections, importance_scores
    )

    # ── Render the Jinja2 template ────────────────────────────────────────────
    now = datetime.now()
    template_vars = {
        "date":              now.strftime("%A, %d %B %Y"),
        "time":              now.strftime("%H:%M"),
        "executive_summary": executive_summary,
        "delta_events":      delta_events,
        "importance_scores": importance_scores,
        "new_connections":   new_connections,
        "action_items":      action_items,
        "topics_watched":    len(watchlist) or len(importance_scores),
        "graph_nodes":       graph_nodes,
        "graph_edges":       graph_edges,
    }

    formatted_brief = _render_template(template_vars)

    safe_log(f"[BriefingAgent] Brief ready — {len(formatted_brief)} chars")

    return {
        "formatted_brief": formatted_brief,
        "current_agent":   "alert_agent",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM SYNTHESIS — executive summary + action items
# ─────────────────────────────────────────────────────────────────────────────

def _generate_executive_summary(
    delta_events:      list[dict],
    new_connections:   list[dict],
    importance_scores: dict[str, float],
) -> str:
    """
    Ask the LLM for a 2–3 sentence executive summary of everything that
    happened overnight. This is the "opening paragraph" of the brief.

    Falls back to a plain-text description if Ollama is unavailable.
    """
    if not delta_events and not new_connections:
        return "No significant changes were detected overnight. Your watched topics are stable."

    # Build a compact context for the LLM
    delta_lines = []
    for e in delta_events[:5]:
        score = importance_scores.get(e["topic"], 0)
        delta_lines.append(f"- [{score:.1f}/10] {e['topic']}: {e['summary'][:120]}")

    conn_lines = [
        f"- {c.get('label_a','?')} ↔ {c.get('label_b','?')}: {c.get('why_notable','')[:80]}"
        for c in new_connections[:3]
    ]

    context = ""
    if delta_lines:
        context += "OVERNIGHT CHANGES:\n" + "\n".join(delta_lines)
    if conn_lines:
        context += "\n\nNEW KNOWLEDGE CONNECTIONS:\n" + "\n".join(conn_lines)

    prompt = f"""You are writing the executive summary for a personal AI intelligence brief.

{context}

Write a concise 2-3 sentence executive summary that:
1. States the most important thing that happened overnight
2. Mentions any surprising connections if present
3. Sets the tone for the day

Be direct and specific. No filler. Write for an AI engineer who wants
the signal, not the noise. Max 60 words total."""

    try:
        summary = invoke_prompt(prompt, temperature=0.4)
        return summary.strip().strip('"')
    except LLMInvocationError as e:
        safe_log(f"[BriefingAgent] LLM unavailable for summary: {e}", level="WARN")
        # Fallback: build summary from highest-importance delta
        if delta_events:
            top = max(delta_events, key=lambda x: importance_scores.get(x["topic"], 0))
            score = importance_scores.get(top["topic"], 0)
            return (
                f"{len(delta_events)} change(s) detected overnight. "
                f"Highest importance: '{top['topic']}' scored {score:.1f}/10. "
                f"{top['summary'][:100]}"
            )
        return f"{len(new_connections)} new knowledge connection(s) discovered in your research graph."


def _generate_action_items(
    delta_events:      list[dict],
    new_connections:   list[dict],
    importance_scores: dict[str, float],
) -> list[str]:
    """
    Ask the LLM for 3–5 specific action items based on what changed.
    These appear at the bottom of the brief as "what should I do today?"

    Falls back to templated items if Ollama is unavailable.
    """
    if not delta_events and not new_connections:
        return ["Continue monitoring your topics — no action needed today."]

    # Prioritise high-importance events
    sorted_events = sorted(
        delta_events,
        key=lambda e: importance_scores.get(e["topic"], 0),
        reverse=True
    )[:4]

    event_lines = [
        f"- {e['topic']} (importance {importance_scores.get(e['topic'], 0):.1f}/10): "
        f"{e['summary'][:100]}"
        for e in sorted_events
    ]

    conn_lines = [
        f"- {c.get('label_a','?')} connects to {c.get('label_b','?')}: "
        f"{c.get('why_notable','')[:80]}"
        for c in new_connections[:2]
    ]

    context = "\n".join(event_lines + conn_lines)

    prompt = f"""Based on these overnight intelligence updates for an AI engineer:

{context}

Generate 3-5 specific, actionable items for today. Each item should:
- Start with an action verb (Review, Investigate, Update, Test, Read, etc.)
- Be concrete and completable in one day
- Directly address one of the changes or connections above

Return ONLY a Python list of strings. No explanation. Example:
["Review LangGraph v0.3 changelog for breaking changes", "Update NexusAI dependency"]"""

    try:
        raw = invoke_prompt(prompt, temperature=0.3)
        # Try to parse as Python list
        import ast, re
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            parsed = ast.literal_eval(match.group())
            if isinstance(parsed, list) and parsed:
                return [str(item).strip() for item in parsed[:5] if str(item).strip()]
    except (LLMInvocationError, ValueError, SyntaxError) as e:
        safe_log(f"[BriefingAgent] LLM unavailable for action items: {e}", level="WARN")

    # Fallback: generate items directly from deltas
    items = []
    for e in sorted_events[:3]:
        score = importance_scores.get(e["topic"], 0)
        if score >= 7.0:
            items.append(f"Investigate: {e['topic']} — high importance ({score:.1f}/10)")
        else:
            items.append(f"Review: {e['topic']} update")
    for c in new_connections[:2]:
        items.append(
            f"Explore connection: {c.get('label_a','?')} ↔ {c.get('label_b','?')}"
        )
    return items or ["Review watched topics in IntelligenceOS dashboard."]


# ─────────────────────────────────────────────────────────────────────────────
# JINJA2 RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def _render_template(variables: dict) -> str:
    """
    Render the Jinja2 morning brief template with the given variables.

    Template location: templates/morning_brief.j2
    Falls back to _plain_text_brief() if Jinja2 is not installed or
    the template file is missing.
    """
    if not _JINJA2_AVAILABLE:
        return _plain_text_brief(variables)

    try:
        env = Environment(
            loader=        FileSystemLoader(TEMPLATES_DIR),
            trim_blocks=   True,   # remove newline after block tags
            lstrip_blocks= True,   # strip leading whitespace from block tags
        )
        template = env.get_template(TEMPLATE_FILE)
        return template.render(**variables)

    except TemplateNotFound:
        safe_log(
            f"[BriefingAgent] Template '{TEMPLATE_FILE}' not found in {TEMPLATES_DIR} "
            f"— using plain-text fallback",
            level="WARN"
        )
        return _plain_text_brief(variables)

    except Exception as e:
        safe_log(
            f"[BriefingAgent] Template render error: {e} — using fallback",
            level="WARN"
        )
        return _plain_text_brief(variables)


def _plain_text_brief(v: dict) -> str:
    """
    Pure-Python fallback brief when Jinja2 is unavailable or template is missing.
    Produces the same sections as the Jinja2 template, in plain text.
    """
    lines = [
        f"IntelligenceOS Morning Brief",
        f"{v.get('date','')}  {v.get('time','')}",
        "=" * 42,
        "",
    ]

    if v.get("executive_summary"):
        lines += ["EXECUTIVE SUMMARY", v["executive_summary"], ""]

    lines += ["=" * 42]

    delta_events      = v.get("delta_events", [])
    importance_scores = v.get("importance_scores", {})

    if delta_events:
        lines += [f"WHAT CHANGED OVERNIGHT ({len(delta_events)} delta(s))", ""]
        for event in delta_events:
            score = importance_scores.get(event["topic"], 0)
            flag  = "HIGH" if score >= 8 else "MED" if score >= 5 else "LOW"
            lines += [
                f"[{flag} {score:.1f}/10] {event['topic'].upper()}",
                event["summary"],
                f"Source: {event['source_url']}",
                "",
            ]
    else:
        lines += ["No significant changes detected overnight.", ""]

    lines += ["=" * 42]

    new_connections = v.get("new_connections", [])
    if new_connections:
        lines += [f"NEW CONNECTIONS FOUND ({len(new_connections)})", ""]
        for c in new_connections:
            lines += [
                f"* {c.get('label_a','?')} <-> {c.get('label_b','?')}",
                f"  {c.get('why_notable','')}",
                "",
            ]
    else:
        lines += ["Knowledge graph updated — no surprising new bridges today.", ""]

    lines += ["=" * 42]

    action_items = v.get("action_items", [])
    if action_items:
        lines += ["ACTION ITEMS FOR TODAY", ""]
        for i, item in enumerate(action_items, 1):
            lines.append(f"  {i}. {item}")
        lines.append("")

    lines += [
        "=" * 42,
        f"Topics watched : {v.get('topics_watched', 0)}",
        f"Deltas found   : {len(delta_events)}",
        f"Graph nodes    : {v.get('graph_nodes', 0)}",
        f"Graph edges    : {v.get('graph_edges', 0)}",
        f"New connections: {len(new_connections)}",
        "=" * 42,
        "Powered by IntelligenceOS v2",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python agents/briefing_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    from tools.secrets_guard import validate_env
    from memory.store import init_db

    validate_env("briefing_agent self-test")
    init_db()

    # Build a realistic test state
    test_state: PersonalAIState = {
        "task":             "intelligenceos morning check",
        "post_to_telegram": False,
        "search_queries":   [],
        "search_results":   [],
        "scraped_content":  [],
        "research_summary": "",
        "thread":           [],
        "final_status":     "in_progress",
        "messages":         [],
        "error":            None,
        "current_agent":    "briefing_agent",
        "watchlist":        ["LangGraph releases", "Anthropic MCP spec", "CrewAI updates"],
        "delta_events": [
            {
                "topic":       "LangGraph releases",
                "summary":     "LangGraph v0.3 released with breaking changes to StateGraph API. Affects all agents using add_conditional_edges(). Migration guide published.",
                "source_url":  "https://github.com/langchain-ai/langgraph/releases",
                "importance":  9.0,
                "detected_at": "2026-04-11 09:00:00",
            },
            {
                "topic":       "Anthropic MCP spec",
                "summary":     "Anthropic published MCP spec v2 with new tool_use protocol. Breaks backward compatibility with v1 clients.",
                "source_url":  "https://anthropic.com/mcp",
                "importance":  8.5,
                "detected_at": "2026-04-11 09:00:01",
            },
        ],
        "importance_scores": {
            "LangGraph releases":  9.0,
            "Anthropic MCP spec":  8.5,
            "CrewAI updates":      3.0,
        },
        "last_checked": {},
        "knowledge_graph": {
            "nodes": [{"id": "langgraph"}, {"id": "mcp_spec"}, {"id": "crewai"}, {"id": "stategraph"}],
            "links": [{"source": "langgraph", "target": "stategraph"}, {"source": "mcp_spec", "target": "langgraph"}],
        },
        "new_connections": [
            {
                "node_a":       "mcp_spec",
                "node_b":       "langgraph",
                "label_a":      "MCP Spec",
                "label_b":      "LangGraph",
                "relationship": "bridges",
                "why_notable":  "MCP Spec v2 directly affects LangGraph's tool-use integration pattern in your NexusAI project.",
            }
        ],
        "formatted_brief": "",
        "alert_sent":      False,
    }

    result = briefing_agent_node(test_state)
    brief  = result.get("formatted_brief", "")

    print("\n" + "=" * 62)
    print("  BRIEFING AGENT OUTPUT")
    print("=" * 62)
    print(brief)
    print("=" * 62)
    print(f"\nBrief length: {len(brief)} chars")
    print(f"Next agent  : {result.get('current_agent')}")
