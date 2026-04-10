# agents/graph_agent.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — GraphAgent
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT THIS AGENT DOES
# ─────────────────────
# GraphAgent is the "librarian who cross-references everything."
# After WatchlistAgent finds what changed, GraphAgent asks:
#   "How does this connect to everything else you've ever researched?"
#
# It builds and maintains a NetworkX DiGraph where:
#   NODES = real-world things: LangGraph, MCP Spec, Anthropic, StateGraph
#   EDGES = relationships:     LangGraph --[extends]--> StateGraph
#                              MCP Spec  --[relates_to]--> LangGraph
#
# SOURCES IT READS:
#   1. delta_events  (from WatchlistAgent — what changed TODAY)
#   2. research_runs (from SQLite — everything ever researched)
#   3. knowledge_graph.json (from last run — accumulated graph)
#
# THE KEY INSIGHT — ENTITY EXTRACTION:
#   The LLM reads each research summary and asks:
#     "What are the distinct concepts here? How do they relate?"
#   Output: structured JSON → concept_a, concept_b, relationship, confidence
#   GraphAgent feeds this into NetworkX.
#
# NODE IDENTITY (de-duplication):
#   All node IDs are normalised: lowercase, spaces→underscores.
#   "LangGraph" and "langgraph" and "LANGGRAPH" all map to node "langgraph".
#   If researched 5 times → ONE node, run_count=5.
#   This is the fundamental difference between a knowledge graph
#   and a research log: it models the WORLD, not your history.
#
# BRIDGE DETECTION:
#   After building the graph, GraphAgent uses betweenness centrality
#   to find "bridge nodes" — concepts that unexpectedly connect
#   separate research threads. These become new_connections[].
#   The LLM then explains WHY each bridge is notable in plain English.
#
# FLOW:
#   load_graph() from JSON
#     ↓
#   for each delta_event: extract_entities() → add nodes + edges
#   for each recent research_run: extract_entities() → add nodes + edges
#     ↓
#   find_new_connections() → bridge nodes via betweenness centrality
#   explain_connections() → LLM explains why each bridge is interesting
#     ↓
#   save_graph() to JSON
#   save_knowledge_edge() × N to SQLite
#     ↓
#   state["knowledge_graph"] = graph_to_dict()
#   state["new_connections"] = explained connections
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import unicodedata
from datetime import datetime

from memory.graph_store import (
    add_edge,
    add_node,
    dict_to_graph,
    find_new_connections,
    graph_to_dict,
    load_graph,
    new_graph,
    save_graph,
)
from memory.store import (
    get_recent_runs,
    save_knowledge_edge,
)
from orchestrator.state import PersonalAIState
from tools.ollama_client import LLMInvocationError, invoke_prompt
from tools.secrets_guard import safe_log


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# How many past research runs to pull into the graph each session
# (keeps graph building fast; older runs are already in the JSON)
RECENT_RUNS_LIMIT = 10

# Maximum entities to extract per text chunk (keeps LLM prompt bounded)
MAX_ENTITIES_PER_CHUNK = 8

# Valid edge types (LLM must choose from these)
VALID_EDGE_TYPES = {"relates_to", "extends", "contradicts", "is_part_of"}

# Default edge type if LLM returns something unexpected
DEFAULT_EDGE_TYPE = "relates_to"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NODE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def graph_agent_node(state: PersonalAIState) -> dict:
    """
    LangGraph node: builds/updates the knowledge graph from all research.

    Reads from state:
        delta_events   List[dict]  — today's changes (from WatchlistAgent)

    Also reads directly from:
        knowledge_graph.json  — accumulated graph from previous sessions
        SQLite research_runs  — full research history

    Writes to state:
        knowledge_graph   dict         — NetworkX graph as JSON-serialisable dict
        new_connections   List[dict]   — surprising cross-topic bridges found
    """
    safe_log("[GraphAgent] Starting knowledge graph build...")

    # ── Step 1: Load the existing accumulated graph ───────────────────────────
    # Every run ADDS to the graph — we never rebuild from scratch.
    # Previous sessions' nodes/edges persist via knowledge_graph.json.
    graph = load_graph()
    safe_log(
        f"[GraphAgent] Loaded graph: "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    # ── Step 2: Process today's delta events ──────────────────────────────────
    # These are the freshest signals — what just changed in watched topics.
    delta_events = state.get("delta_events", [])
    safe_log(f"[GraphAgent] Processing {len(delta_events)} delta event(s)...")

    new_edges_count = 0
    for event in delta_events:
        topic   = event.get("topic",   "unknown")
        summary = event.get("summary", "")
        if not summary:
            continue

        edges = _extract_entities(summary, context=f"delta event for topic: {topic}")
        for edge in edges:
            _apply_edge_to_graph(graph, edge, run_id=None)
            new_edges_count += 1

        # Always add the topic itself as a node
        node_id = _normalise_id(topic)
        add_node(graph, node_id, label=topic, category="topic")

    safe_log(f"[GraphAgent] Added {new_edges_count} edges from delta events")

    # ── Step 3: Process recent research history ───────────────────────────────
    # Pull the last N research runs from SQLite and integrate them.
    # Runs already in the graph will just increment run_count on their nodes —
    # no duplicate edges are created (add_edge() averages weight on repeats).
    safe_log(f"[GraphAgent] Loading last {RECENT_RUNS_LIMIT} research runs from SQLite...")
    recent_runs = get_recent_runs(limit=RECENT_RUNS_LIMIT)
    history_edges_count = 0

    for run in recent_runs:
        if not run.summary or run.status == "failed":
            continue

        edges = _extract_entities(
            run.summary,
            context=f"research run #{run.id}: {run.task}"
        )
        for edge in edges:
            _apply_edge_to_graph(graph, edge, run_id=run.id)
            _persist_edge_to_sqlite(edge, run_id=run.id)
            history_edges_count += 1

        # Add the research task itself as a "topic" node
        task_node_id = _normalise_id(run.task[:50])
        add_node(graph, task_node_id, label=run.task[:50], category="topic")

    safe_log(
        f"[GraphAgent] Added {history_edges_count} edges from "
        f"{len(recent_runs)} historical runs"
    )

    # ── Step 4: Find new cross-topic connections ───────────────────────────────
    # Bridge detection: which nodes unexpectedly connect separate clusters?
    raw_connections = find_new_connections(graph)
    safe_log(f"[GraphAgent] Bridge detection found {len(raw_connections)} candidate(s)")

    # Ask the LLM to explain each bridge in plain English
    explained_connections = _explain_connections(raw_connections, graph)
    safe_log(f"[GraphAgent] Explained {len(explained_connections)} new connection(s)")

    # ── Step 5: Save updated graph ────────────────────────────────────────────
    save_graph(graph)
    safe_log(
        f"[GraphAgent] Final graph: "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    return {
        "knowledge_graph": graph_to_dict(graph),
        "new_connections": explained_connections,
        "current_agent":   "briefing_agent",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION — LLM reads text → structured edges
# ─────────────────────────────────────────────────────────────────────────────

def _extract_entities(text: str, context: str = "") -> list[dict]:
    """
    Ask Ollama to read a research summary and extract concept relationships.

    Returns a list of edge dicts:
        [{
            "concept_a":    str,   # source node label
            "concept_b":    str,   # target node label
            "relationship": str,   # "relates_to" | "extends" | "contradicts" | "is_part_of"
            "confidence":   float, # 0.0–1.0
            "category_a":   str,   # "framework" | "concept" | "person" | "company" | "topic"
            "category_b":   str,
        }, ...]

    Falls back to _fallback_extract() if LLM is unavailable.
    Returns [] on complete failure.
    """
    # Keep text bounded so the prompt doesn't balloon
    truncated = text[:1500]

    prompt = f"""You are a knowledge graph builder. Extract concept relationships from this text.

CONTEXT: {context}

TEXT:
{truncated}

Identify up to {MAX_ENTITIES_PER_CHUNK} relationships between distinct concepts, frameworks, tools, people, or companies mentioned.

For each relationship, choose a type:
  - "relates_to"  : general connection between two concepts
  - "extends"     : concept A builds upon or is based on concept B
  - "contradicts" : concept A conflicts with or challenges concept B
  - "is_part_of"  : concept A is a component or subset of concept B

For each node, choose a category:
  - "framework"   : software framework or library (e.g. LangGraph, CrewAI)
  - "concept"     : abstract idea or pattern (e.g. StateGraph, RAG, agents)
  - "person"      : individual person (e.g. Yann LeCun)
  - "company"     : organisation (e.g. Anthropic, OpenAI)
  - "topic"       : broad research area (e.g. AI safety, vector search)

Respond ONLY with a JSON array. No explanation. No markdown. Example:
[
  {{"concept_a": "LangGraph", "category_a": "framework", "concept_b": "StateGraph", "category_b": "concept", "relationship": "extends", "confidence": 0.9}},
  {{"concept_a": "CrewAI", "category_a": "framework", "concept_b": "LangGraph", "category_b": "framework", "relationship": "relates_to", "confidence": 0.75}}
]

If you cannot find any clear relationships, return an empty array: []"""

    try:
        raw = invoke_prompt(prompt, temperature=0)
        return _parse_entity_response(raw)
    except LLMInvocationError as e:
        safe_log(
            f"[GraphAgent] LLM unavailable for entity extraction — "
            f"using fallback: {e}",
            level="WARN"
        )
        return _fallback_extract(text)


def _parse_entity_response(raw: str) -> list[dict]:
    """
    Parse the LLM's JSON array response into edge dicts.

    Handles messy output:
    - Strips markdown code fences
    - Extracts JSON array even if surrounded by text
    - Validates required fields
    - Normalises relationship type to known values
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Extract JSON array
    array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not array_match:
        return []

    try:
        parsed = json.loads(array_match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    edges = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        concept_a = str(item.get("concept_a", "")).strip()
        concept_b = str(item.get("concept_b", "")).strip()
        if not concept_a or not concept_b:
            continue
        if concept_a.lower() == concept_b.lower():
            continue   # skip self-loops

        relationship = str(item.get("relationship", DEFAULT_EDGE_TYPE)).strip()
        if relationship not in VALID_EDGE_TYPES:
            relationship = DEFAULT_EDGE_TYPE

        confidence = float(item.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))

        edges.append({
            "concept_a":    concept_a,
            "concept_b":    concept_b,
            "relationship": relationship,
            "confidence":   confidence,
            "category_a":   str(item.get("category_a", "concept")).strip(),
            "category_b":   str(item.get("category_b", "concept")).strip(),
        })

    return edges[:MAX_ENTITIES_PER_CHUNK]


def _fallback_extract(text: str) -> list[dict]:
    """
    When Ollama is down, extract concepts using regex heuristics.

    Strategy: find capitalised multi-word phrases (likely proper nouns
    / framework names) and link adjacent ones with "relates_to".
    This produces low-confidence edges that still add signal to the graph.
    """
    # Match capitalised words / acronyms of 2+ chars (CamelCase or ALLCAPS)
    pattern = r"\b([A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,})*|[A-Z]{2,})\b"
    matches = re.findall(pattern, text)

    # Deduplicate while preserving order
    seen  = set()
    terms = []
    for m in matches:
        m_clean = m.strip()
        if m_clean and m_clean.lower() not in seen and len(m_clean) > 2:
            seen.add(m_clean.lower())
            terms.append(m_clean)

    edges = []
    for i in range(len(terms) - 1):
        if len(edges) >= MAX_ENTITIES_PER_CHUNK:
            break
        edges.append({
            "concept_a":    terms[i],
            "concept_b":    terms[i + 1],
            "relationship": "relates_to",
            "confidence":   0.5,           # low confidence — heuristic extraction
            "category_a":   "concept",
            "category_b":   "concept",
        })

    return edges


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_id(text: str) -> str:
    """
    Convert any concept label into a consistent, stable node ID.

    Rules:
        - Unicode → ASCII (é → e, ü → u)
        - Lowercase
        - Strip punctuation except hyphens
        - Spaces and hyphens → underscore
        - Collapse multiple underscores

    Examples:
        "LangGraph"   → "langgraph"
        "MCP Spec"    → "mcp_spec"
        "GPT-4"       → "gpt_4"
        "Anthropic"   → "anthropic"

    This ensures "LangGraph" mentioned in 5 different summaries
    always maps to the same graph node.
    """
    # Normalise unicode (é → e)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = text.lower().strip()

    # Replace spaces, hyphens, dots with underscore
    text = re.sub(r"[\s\-\.]+", "_", text)

    # Remove any remaining non-alphanumeric, non-underscore chars
    text = re.sub(r"[^a-z0-9_]", "", text)

    # Collapse multiple underscores
    text = re.sub(r"_+", "_", text).strip("_")

    return text or "unknown"


def _apply_edge_to_graph(
    graph:  object,
    edge:   dict,
    run_id: int | None,
) -> None:
    """
    Add one extracted edge (and its two nodes) to the NetworkX graph.
    Node IDs are normalised so repeated mentions update the same node.
    """
    concept_a = edge["concept_a"]
    concept_b = edge["concept_b"]
    node_id_a = _normalise_id(concept_a)
    node_id_b = _normalise_id(concept_b)

    # Add / update both nodes
    add_node(
        graph, node_id_a,
        label=    concept_a,
        category= edge.get("category_a", "concept"),
        run_id=   run_id,
    )
    add_node(
        graph, node_id_b,
        label=    concept_b,
        category= edge.get("category_b", "concept"),
        run_id=   run_id,
    )

    # Add the directed edge
    add_edge(
        graph,
        source=    node_id_a,
        target=    node_id_b,
        edge_type= edge["relationship"],
        weight=    edge["confidence"],
        run_id=    run_id,
    )


def _persist_edge_to_sqlite(edge: dict, run_id: int | None) -> None:
    """
    Save an edge to the knowledge_edges SQLite table.
    This is the permanent audit trail — the JSON file is the fast cache.
    Swallows errors so one bad write doesn't break the whole pipeline.
    """
    try:
        save_knowledge_edge(
            concept_a=    edge["concept_a"],
            concept_b=    edge["concept_b"],
            relationship= edge["relationship"],
            source_run_id=run_id,
            confidence=   edge["confidence"],
        )
    except Exception as e:
        safe_log(
            f"[GraphAgent] SQLite edge write failed "
            f"({edge['concept_a']} → {edge['concept_b']}): {e}",
            level="WARN"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION EXPLANATION — LLM explains WHY bridges are notable
# ─────────────────────────────────────────────────────────────────────────────

def _explain_connections(
    raw_connections: list[dict],
    graph:           object,
) -> list[dict]:
    """
    Take the raw bridge connections from find_new_connections() and ask
    the LLM to explain each one in plain English.

    WHY: The centality algorithm finds structural bridges but can't explain
    them. "node_a bridges 3 research threads" is a graph fact.
    "LangGraph connects your NexusAI architecture to the new MCP spec
    because both rely on stateful agent orchestration" is an insight.

    Returns the same list with "why_notable" field enriched by the LLM.
    Falls back to the algorithmic description if LLM is unavailable.
    """
    if not raw_connections:
        return []

    explained = []
    for conn in raw_connections:
        node_a = conn.get("node_a", "")
        node_b = conn.get("node_b", "")
        if not node_a or not node_b:
            continue

        # Get display labels from the graph for the prompt
        label_a = _get_node_label(graph, node_a)
        label_b = _get_node_label(graph, node_b)

        # Build a short neighbourhood description for context
        neighbours_a = _describe_neighbours(graph, node_a)
        neighbours_b = _describe_neighbours(graph, node_b)

        prompt = f"""You are an AI research analyst for a personal intelligence system.

In the knowledge graph, "{label_b}" appears to be a bridge connecting "{label_a}" to other research threads.

"{label_a}" is connected to: {neighbours_a}
"{label_b}" is connected to: {neighbours_b}

In ONE concise sentence (max 25 words), explain why the connection between "{label_a}" and "{label_b}" is notable or surprising for an AI engineer to know about.

Respond with just the one sentence. No preamble."""

        try:
            explanation = invoke_prompt(prompt, temperature=0.3)
            explanation = explanation.strip().strip('"')
        except LLMInvocationError:
            # Keep the algorithmic description as fallback
            explanation = conn.get("why_notable", f"'{label_a}' and '{label_b}' share unexpected structural connections")

        explained.append({
            "node_a":       node_a,
            "node_b":       node_b,
            "label_a":      label_a,
            "label_b":      label_b,
            "relationship": conn.get("relationship", "bridges"),
            "why_notable":  explanation,
        })

    return explained


def _get_node_label(graph: object, node_id: str) -> str:
    """Return the human-readable label for a node, falling back to the ID."""
    try:
        return graph.nodes[node_id].get("label", node_id)
    except Exception:
        return node_id


def _describe_neighbours(graph: object, node_id: str, limit: int = 5) -> str:
    """
    Build a short comma-separated string of a node's neighbours' labels.
    Used to give the LLM context about what a node is connected to.
    """
    try:
        import networkx as nx
        successors   = [_get_node_label(graph, n) for n in graph.successors(node_id)]
        predecessors = [_get_node_label(graph, n) for n in graph.predecessors(node_id)]
        all_nb = list(dict.fromkeys(successors + predecessors))[:limit]
        return ", ".join(all_nb) if all_nb else "nothing yet"
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python agents/graph_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    from tools.secrets_guard import validate_env
    from memory.store import init_db, save_run, add_to_watchlist

    validate_env("graph_agent self-test")
    init_db()

    # Seed the database with 2 research runs to give the graph some history
    run1 = save_run(
        task=    "LangGraph v0.3 new features",
        summary= (
            "KEY FINDINGS: LangGraph v0.3 released with improved StateGraph. "
            "Anthropic contributed MCP spec integration. "
            "NexusAI architecture uses LangGraph as its orchestration core. "
            "StateGraph extends the concept of finite state machines for AI agents."
        ),
        thread=  ["1/ LangGraph updated"],
        status=  "completed",
        sources= 3,
    )
    run2 = save_run(
        task=    "CrewAI vs LangGraph comparison",
        summary= (
            "KEY FINDINGS: CrewAI and LangGraph both implement multi-agent orchestration. "
            "CrewAI is built on top of LangChain. "
            "LangGraph offers finer state control. "
            "Anthropic endorses LangGraph for production agent systems."
        ),
        thread=  ["1/ CrewAI vs LangGraph"],
        status=  "completed",
        sources= 4,
    )

    # Build state with some delta events
    test_state: PersonalAIState = {
        "task":             "graph agent self-test",
        "post_to_telegram": False,
        "search_queries":   [],
        "search_results":   [],
        "scraped_content":  [],
        "research_summary": "",
        "thread":           [],
        "final_status":     "in_progress",
        "messages":         [],
        "error":            None,
        "current_agent":    "graph_agent",
        "delta_events": [
            {
                "topic":       "Anthropic MCP spec",
                "summary":     "Anthropic released MCP spec v2 with new tool calling protocol. Affects LangGraph integration patterns.",
                "source_url":  "https://anthropic.com/mcp",
                "importance":  8.5,
                "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ],
        "importance_scores": {"Anthropic MCP spec": 8.5},
        "last_checked":      {},
        "knowledge_graph":   {},
        "new_connections":   [],
        "formatted_brief":   "",
        "alert_sent":        False,
    }

    safe_log("\n[Test] Running GraphAgent node...")
    result = graph_agent_node(test_state)

    print("\n" + "=" * 60)
    print("  GRAPH AGENT RESULT")
    print("=" * 60)

    kg = result.get("knowledge_graph", {})
    nodes = kg.get("nodes", [])
    links = kg.get("links", [])
    print(f"  Nodes in graph : {len(nodes)}")
    print(f"  Edges in graph : {len(links)}")

    if nodes:
        print("\n── Nodes ─────────────────────────────────────────────")
        for n in nodes[:10]:
            print(f"  [{n.get('category','?'):10}]  {n.get('label', n.get('id','?'))}")
        if len(nodes) > 10:
            print(f"  ... and {len(nodes)-10} more")

    if links:
        print("\n── Edges ─────────────────────────────────────────────")
        for l in links[:10]:
            print(f"  {l.get('source'):20} --[{l.get('type','?'):12}]--> {l.get('target')}")
        if len(links) > 10:
            print(f"  ... and {len(links)-10} more")

    connections = result.get("new_connections", [])
    if connections:
        print("\n── New Connections (Surprising Bridges) ──────────────")
        for c in connections:
            print(f"  '{c.get('label_a','?')}' ↔ '{c.get('label_b','?')}'")
            print(f"  Why notable: {c.get('why_notable','')}")
            print()

    print("=" * 60)
    print("GraphAgent self-test complete.")
