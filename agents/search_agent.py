import ast
import re

from orchestrator.state import PersonalAIState
from tools.ollama_client import LLMInvocationError, invoke_prompt
from tools.web_search import WebSearchError, web_search


def _parse_queries(raw: str, fallback: str) -> list[str]:
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            return [fallback]
        try:
            parsed = ast.literal_eval(match.group())
        except (ValueError, SyntaxError):
            return [fallback]

    if not isinstance(parsed, list):
        return [fallback]

    queries = [str(item).strip() for item in parsed if str(item).strip()]
    return queries[:3] or [fallback]


def search_agent_node(state: PersonalAIState) -> dict:
    print(f"\n[Search Agent] Task: {state['task']}")

    prompt = f"""You are a search query generator for an AI research agent.

User's task: {state['task']}

Generate 3 specific, targeted search queries to find the best information.

Return ONLY a Python list of strings. No explanation. Example format:
["query one", "query two", "query three"]"""

    try:
        raw = invoke_prompt(prompt, temperature=0)
        queries = _parse_queries(raw, state["task"])
    except LLMInvocationError as exc:
        print(f"[Search Agent] LLM unavailable, using task as query: {exc}")
        queries = [state["task"]]

    print(f"[Search Agent] Queries: {queries}")

    all_results = []
    search_errors = []
    for query in queries:
        try:
            results = web_search(query, max_results=3)
            all_results.extend(results)
        except WebSearchError as exc:
            print(f"[Search Agent] Search failed for '{query}': {exc}")
            search_errors.append(str(exc))

    seen = set()
    unique_results = []
    for result in all_results:
        url = result.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        unique_results.append(result)

    print(f"[Search Agent] Found {len(unique_results)} unique results")

    return {
        "search_queries": queries,
        "search_results": unique_results,
        "current_agent": "research_agent",
        "error": search_errors[0] if search_errors and not unique_results else None,
    }
