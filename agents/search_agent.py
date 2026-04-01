# agents/search_agent.py

import os, re, ast
from langchain_ollama import ChatOllama
from orchestrator.state import PersonalAIState
from tools.web_search import web_search
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    temperature=0,
)

def search_agent_node(state: PersonalAIState) -> dict:
    print(f"\n[Search Agent] Task: {state['task']}")

    # Step 1: Generate smart search queries via LLM
    prompt = f"""You are a search query generator for an AI research agent.

User's task: {state['task']}

Generate 3 specific, targeted search queries to find the best information.

Return ONLY a Python list of strings. No explanation. Example format:
["query one", "query two", "query three"]"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Step 2: Parse LLM output safely
    try:
        queries = ast.literal_eval(raw)
    except:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            try:
                queries = ast.literal_eval(match.group())
            except:
                queries = [state['task']]
        else:
            queries = [state['task']]

    print(f"[Search Agent] Queries: {queries}")

    # Step 3: Search the web for each query
    all_results = []
    for query in queries:
        results = web_search(query, max_results=3)
        all_results.extend(results)

    # Step 4: Deduplicate by URL
    seen, unique_results = set(), []
    for r in all_results:
        if r['url'] not in seen:
            seen.add(r['url'])
            unique_results.append(r)

    print(f"[Search Agent] Found {len(unique_results)} unique results")

    return {
        "search_queries":  queries,
        "search_results":  unique_results,
        "current_agent":   "research_agent",
    }