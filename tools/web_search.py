# tools/web_search.py

import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Initialize once, reuse everywhere
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily.
    Returns a list of results, each with: title, url, content snippet.
    
    Why Tavily over Google?
    - Built specifically for AI agents
    - Returns clean text (no HTML garbage)
    - Free tier is generous
    - One line to call, structured output back
    """
    try:
        response = tavily.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",    # deeper crawl than basic
            include_answer=False,       # we want raw results, not a summary
        )
        
        # Extract only what we need — clean it up
        results = []
        for item in response.get("results", []):
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("url", ""),
                "snippet": item.get("content", ""),  # short text preview
            })
        
        return results

    except Exception as e:
        # Never let a tool crash silently
        print(f"[web_search] ERROR: {e}")
        return []   # return empty list, not None — agents expect a list


# ── Quick test (run this file directly to verify) ──────────────
if __name__ == "__main__":
    results = web_search("latest AI agent frameworks 2026")
    for r in results:
        print(f"\nTitle:   {r['title']}")
        print(f"URL:     {r['url']}")
        print(f"Snippet: {r['snippet'][:100]}...")