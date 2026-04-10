# memory/graph_store.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — Knowledge Graph Persistence
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY A SEPARATE FILE FOR GRAPH PERSISTENCE?
# ──────────────────────────────────────────
# NetworkX lives entirely in memory — when the process ends, the graph is gone.
# We need two complementary storage strategies:
#
#   1. SQLite (memory/store.py)
#      → knowledge_edges table stores individual edges as rows
#      → queryable: "give me all edges for LangGraph"
#      → permanent audit trail
#
#   2. JSON file (this file)
#      → entire NetworkX graph serialised as node-link JSON
#      → fast full-graph reload at startup
#      → GraphAgent reads this at the start of every run
#
# Think of it like: SQLite = individual LEGO bricks in a box
#                   JSON   = a photo of the assembled model
# Both exist. The photo is faster to read back. The bricks are queryable.
#
# NETWORKX FORMAT USED: node_link_data / node_link_graph
#   - Most portable JSON format in NetworkX
#   - Preserves node attributes, edge attributes, graph direction
#   - Human-readable: you can open the JSON and see your knowledge graph
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
from datetime import datetime
from typing import Optional

from tools.secrets_guard import safe_log

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    safe_log(
        "[GraphStore] networkx not installed — graph features disabled. "
        "Run: pip install networkx",
        level="WARN"
    )


# Graph JSON file lives next to the SQLite database in project root
GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge_graph.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def new_graph() -> "nx.DiGraph":
    """
    Create a fresh, empty directed knowledge graph.

    WHY DiGraph (directed graph)?
        Relationships have direction:
        "LangGraph EXTENDS StateGraph" is NOT the same as
        "StateGraph EXTENDS LangGraph"
        Direction matters. Use DiGraph.

    Node attributes we'll use:
        label      (str)  : human-readable name
        category   (str)  : "topic" | "concept" | "person" | "company" | "framework"
        first_seen (str)  : ISO timestamp when this node was first added
        run_count  (int)  : how many research runs mentioned this node

    Edge attributes we'll use:
        type       (str)  : "relates_to" | "contradicts" | "extends" | "is_part_of"
        weight     (float): 0.0–1.0 confidence score
        source_run (int)  : research_run.id that produced this edge
        created_at (str)  : ISO timestamp
    """
    if not _NX_AVAILABLE:
        return {}   # fallback empty dict if networkx not installed

    G = nx.DiGraph()
    G.graph["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    G.graph["version"]    = "intelligenceos-v2"
    return G


def save_graph(graph: "nx.DiGraph") -> None:
    """
    Serialise the NetworkX graph to JSON and write to GRAPH_PATH.

    Uses NetworkX node-link format:
        {
          "directed": true,
          "multigraph": false,
          "graph": {"created_at": "...", "version": "..."},
          "nodes": [{"id": "LangGraph", "label": "LangGraph", ...}, ...],
          "links": [{"source": "LangGraph", "target": "NexusAI",
                     "type": "is_part_of", "weight": 0.9, ...}, ...]
        }

    Called by GraphAgent after building/updating the graph.
    Overwrites the previous version (only latest graph is kept on disk).
    SQLite keeps the full history via knowledge_edges table.
    """
    if not _NX_AVAILABLE:
        safe_log("[GraphStore] networkx unavailable — graph not saved", level="WARN")
        return

    if not isinstance(graph, nx.DiGraph):
        safe_log("[GraphStore] save_graph received non-DiGraph — skipped", level="WARN")
        return

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    # Serialise to node-link JSON format
    data = nx.node_link_data(graph)

    # Add metadata to the JSON envelope
    data["_meta"] = {
        "saved_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "node_count": node_count,
        "edge_count": edge_count,
    }

    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    safe_log(
        f"[GraphStore] Saved graph: {node_count} nodes, "
        f"{edge_count} edges → {GRAPH_PATH}"
    )


def load_graph() -> "nx.DiGraph":
    """
    Load the knowledge graph from JSON back into a NetworkX DiGraph.
    Returns an empty DiGraph if the file doesn't exist yet.

    Called by GraphAgent at the START of every run to load prior knowledge
    before adding new nodes/edges from the current research session.

    This is how IntelligenceOS accumulates knowledge over time:
        Run 1: LangGraph → NexusAI
        Run 2: loads graph, adds MCP → LangGraph
        Run 3: loads graph, adds CrewAI → MCP
        Graph now has all 3 runs connected.
    """
    if not _NX_AVAILABLE:
        safe_log("[GraphStore] networkx unavailable — returning empty dict", level="WARN")
        return {}

    if not os.path.exists(GRAPH_PATH):
        safe_log("[GraphStore] No graph file found — starting with empty graph")
        return new_graph()

    try:
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Remove our metadata key before passing to networkx
        data.pop("_meta", None)

        graph = nx.node_link_graph(data, directed=True, multigraph=False)

        safe_log(
            f"[GraphStore] Loaded graph: "
            f"{graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        return graph

    except (json.JSONDecodeError, KeyError, Exception) as e:
        safe_log(
            f"[GraphStore] Failed to load graph ({e}) — "
            f"starting fresh",
            level="WARN"
        )
        return new_graph()


def add_node(
    graph:    "nx.DiGraph",
    node_id:  str,
    label:    str         = "",
    category: str         = "concept",
    run_id:   Optional[int] = None,
) -> "nx.DiGraph":
    """
    Add or update a node in the graph.

    WHY ADD OR UPDATE?
        If "LangGraph" was researched 5 times, it should be ONE node
        with run_count=5 — not 5 separate nodes.
        This is the core identity principle of the knowledge graph:
        nodes represent THINGS IN THE WORLD, not research sessions.

    Args:
        node_id:  Unique identifier (use the concept name, lowercased)
        label:    Human-readable display name
        category: "topic" | "concept" | "person" | "company" | "framework"
        run_id:   Source research run (for tracing where this came from)

    Returns:
        The updated graph (modified in place, but also returned for chaining)
    """
    if not _NX_AVAILABLE:
        return graph

    if graph.has_node(node_id):
        # Node exists — increment run_count, update label if provided
        current = graph.nodes[node_id]
        graph.nodes[node_id]["run_count"] = current.get("run_count", 1) + 1
        if label:
            graph.nodes[node_id]["label"] = label
    else:
        # New node
        graph.add_node(
            node_id,
            label=      label or node_id,
            category=   category,
            first_seen= datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_count=  1,
            source_run= run_id,
        )

    return graph


def add_edge(
    graph:        "nx.DiGraph",
    source:       str,
    target:       str,
    edge_type:    str   = "relates_to",
    weight:       float = 0.8,
    run_id:       Optional[int] = None,
) -> "nx.DiGraph":
    """
    Add a directed edge between two nodes.
    If either node doesn't exist yet, it's created automatically.

    Args:
        source:    Source node ID (e.g. "langgraph")
        target:    Target node ID (e.g. "nexusai")
        edge_type: Relationship type — one of:
                   "relates_to"  : general connection
                   "extends"     : source builds on target
                   "contradicts" : source conflicts with target
                   "is_part_of"  : source is a component of target
        weight:    Confidence 0.0–1.0
        run_id:    Research run that produced this edge

    Returns:
        The updated graph
    """
    if not _NX_AVAILABLE:
        return graph

    # Auto-create nodes if they don't exist
    if not graph.has_node(source):
        add_node(graph, source, run_id=run_id)
    if not graph.has_node(target):
        add_node(graph, target, run_id=run_id)

    # If edge already exists, update weight (average of old and new)
    if graph.has_edge(source, target):
        old_weight = graph[source][target].get("weight", weight)
        graph[source][target]["weight"] = (old_weight + weight) / 2
    else:
        graph.add_edge(
            source, target,
            type=       edge_type,
            weight=     weight,
            source_run= run_id,
            created_at= datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    return graph


def graph_to_dict(graph: "nx.DiGraph") -> dict:
    """
    Convert graph to a plain dict for storing in LangGraph State.
    (State can't hold a NetworkX object — it must be JSON-serialisable.)

    Returns the node-link format dict.
    GraphAgent stores this in state["knowledge_graph"].
    """
    if not _NX_AVAILABLE or not isinstance(graph, nx.DiGraph):
        return {}
    return nx.node_link_data(graph)


def dict_to_graph(data: dict) -> "nx.DiGraph":
    """
    Reconstruct a NetworkX DiGraph from a node-link format dict.
    Used when GraphAgent reads state["knowledge_graph"].
    """
    if not _NX_AVAILABLE or not data:
        return new_graph()
    try:
        return nx.node_link_graph(data, directed=True, multigraph=False)
    except Exception as e:
        safe_log(f"[GraphStore] dict_to_graph failed: {e}", level="WARN")
        return new_graph()


def find_new_connections(graph: "nx.DiGraph") -> list[dict]:
    """
    Scan the graph for nodes that appear in multiple separate topic clusters
    but weren't originally connected — these are the "surprising insights."

    Strategy used: find nodes with HIGH betweenness centrality that
    connect nodes from DIFFERENT original research sessions.

    Returns a list of connection dicts:
        [{
            "node_a":       str,
            "node_b":       str,
            "relationship": str,
            "why_notable":  str,  (placeholder — BriefingAgent will fill this)
        }]

    BriefingAgent takes this list and asks Ollama to explain WHY each
    connection is notable in plain English.
    """
    if not _NX_AVAILABLE or not isinstance(graph, nx.DiGraph):
        return []

    if graph.number_of_nodes() < 3:
        return []   # Need at least 3 nodes to find non-trivial connections

    connections = []

    try:
        # Find nodes that bridge different parts of the graph
        centrality = nx.betweenness_centrality(graph, normalized=True)

        # Threshold: top 20% most central nodes are "bridge" nodes
        if centrality:
            threshold = sorted(centrality.values())[-max(1, len(centrality) // 5)]

            bridge_nodes = [
                node for node, score in centrality.items()
                if score >= threshold
            ]

            for node in bridge_nodes:
                neighbors = list(graph.neighbors(node))
                predecessors = list(graph.predecessors(node))
                all_connected = set(neighbors + predecessors)

                if len(all_connected) >= 2:
                    # Build connection entries for pairs
                    connected_list = list(all_connected)
                    for i in range(min(2, len(connected_list))):
                        connections.append({
                            "node_a":       connected_list[i],
                            "node_b":       node,
                            "relationship": "bridges",
                            "why_notable":  (
                                f"'{node}' connects {len(all_connected)} "
                                f"separate research threads"
                            ),
                        })

    except Exception as e:
        safe_log(f"[GraphStore] find_new_connections error: {e}", level="WARN")

    return connections[:5]   # Return top 5 most notable connections


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python memory/graph_store.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("Testing graph_store.py ...\n")

    # Build a small knowledge graph
    G = new_graph()
    print(f"Empty graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Add nodes
    add_node(G, "langgraph",    "LangGraph",      "framework")
    add_node(G, "nexusai",      "NexusAI",        "topic")
    add_node(G, "mcp_spec",     "MCP Spec",       "concept")
    add_node(G, "crewai",       "CrewAI",         "framework")
    add_node(G, "stategraph",   "StateGraph",     "concept")

    # Simulate second mention of LangGraph — should increment run_count
    add_node(G, "langgraph",    "LangGraph",      "framework")
    assert G.nodes["langgraph"]["run_count"] == 2, "run_count should be 2"
    print("run_count de-duplication ... OK")

    # Add edges
    add_edge(G, "langgraph",  "nexusai",    "is_part_of", 0.9)
    add_edge(G, "mcp_spec",   "langgraph",  "extends",    0.85)
    add_edge(G, "crewai",     "langgraph",  "relates_to", 0.7)
    add_edge(G, "stategraph", "langgraph",  "is_part_of", 0.95)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Test serialisation round-trip
    d = graph_to_dict(G)
    G2 = dict_to_graph(d)
    assert G2.number_of_nodes() == G.number_of_nodes(), "Round-trip node count mismatch"
    assert G2.number_of_edges() == G.number_of_edges(), "Round-trip edge count mismatch"
    print("graph_to_dict → dict_to_graph round-trip ... OK")

    # Test save/load
    save_graph(G)
    G3 = load_graph()
    assert G3.number_of_nodes() == G.number_of_nodes(), "Save/load node count mismatch"
    print(f"save_graph + load_graph ... OK")

    # Test bridge detection
    connections = find_new_connections(G)
    print(f"New connections found: {len(connections)}")
    for c in connections:
        print(f"  '{c['node_a']}' ↔ '{c['node_b']}': {c['why_notable']}")

    print("\nAll graph_store tests passed.")
