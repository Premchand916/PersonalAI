# orchestrator/graph.py  (updated)

from langgraph.graph import StateGraph, END
from orchestrator.state import PersonalAIState
from agents.search_agent import search_agent_node

def build_graph():
    graph = StateGraph(PersonalAIState)
    
    # Only search agent for now — others coming next session
    graph.add_node("search_agent", search_agent_node)
    
    graph.set_entry_point("search_agent")
    graph.add_edge("search_agent", END)     # temporary: end after search
    
    return graph.compile()