# orchestrator/graph.py

from langgraph.graph import StateGraph, END
from orchestrator.state import PersonalAIState
from agents.search_agent import search_agent_node
from agents.research_agent import research_agent_node

def build_graph():
    graph = StateGraph(PersonalAIState)
    
    graph.add_node("search_agent",   search_agent_node)
    graph.add_node("research_agent", research_agent_node)
    
    graph.set_entry_point("search_agent")
    graph.add_edge("search_agent",   "research_agent")
    graph.add_edge("research_agent", END)           # twitter coming next session
    
    return graph.compile()