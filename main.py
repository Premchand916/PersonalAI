# main.py

from orchestrator.graph import build_graph

def run(task: str):
    graph = build_graph()
    
    # Initial state — only task is filled, everything else empty
    initial_state = {
        "task": task,
        "search_queries":   [],
        "search_results":   [],
        "scraped_content":  [],
        "research_summary": "",
        "twitter_thread":   [],
        "final_status":     "in_progress",
        "messages":         [],
        "error":            None,
        "current_agent":    "search_agent",
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Print results
    print("\n" + "="*50)
    print("SEARCH RESULTS:")
    print("="*50)
    for i, r in enumerate(result["search_results"], 1):
        print(f"\n{i}. {r['title']}")
        print(f"   URL: {r['url']}")
        print(f"   {r['snippet'][:150]}...")
    
    return result

if __name__ == "__main__":
    run("latest AI agent frameworks and tools 2026")