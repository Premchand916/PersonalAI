import os
from functools import lru_cache

from dotenv import load_dotenv

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

load_dotenv()


class WebSearchError(RuntimeError):
    """Raised when Tavily search cannot be completed."""


@lru_cache(maxsize=1)
def get_tavily_client():
    if TavilyClient is None:
        raise WebSearchError(
            "tavily-python is not installed. Install the project dependencies first."
        )

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise WebSearchError("TAVILY_API_KEY is not set.")

    return TavilyClient(api_key=api_key)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    if not query.strip():
        return []

    try:
        client = get_tavily_client()
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })

        return results
    except Exception as exc:
        message = str(exc).lower()
        if (
            "failed to establish a new connection" in message
            or "forbidden by its access permissions" in message
        ):
            raise WebSearchError(
                "Could not reach Tavily. Check your internet access, firewall, or proxy settings."
            ) from exc
        raise WebSearchError(f"Tavily search failed: {exc}") from exc


if __name__ == "__main__":
    results = web_search("latest AI agent frameworks 2026")
    for result in results:
        print(f"\nTitle:   {result['title']}")
        print(f"URL:     {result['url']}")
        print(f"Snippet: {result['snippet'][:100]}...")
