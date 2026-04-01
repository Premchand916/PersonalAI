# agents/research_agent.py

import os
from langchain_ollama import ChatOllama
from orchestrator.state import PersonalAIState
from tools.scraper import scrape_with_retry, ScrapeSuccess, ScrapeFailure
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    temperature=0,
)

def research_agent_node(state: PersonalAIState) -> dict:
    print(f"\n[Research Agent] Processing {len(state['search_results'])} URLs...")

    successful = []     # ScrapeSuccess objects
    failed     = []     # ScrapeFailure objects

    # ── Step 1: Scrape all URLs, categorize results ────────────────
    for i, result in enumerate(state["search_results"]):
        url = result["url"]
        print(f"[Research Agent] Scraping {i+1}/{len(state['search_results'])}: {url}")

        scrape_result = scrape_with_retry(url)

        if isinstance(scrape_result, ScrapeSuccess):
            print(f"  ✅ Success — {scrape_result.chars} chars extracted")
            successful.append({
                "url":     url,
                "title":   result.get("title", ""),
                "content": scrape_result.content,
            })

        elif isinstance(scrape_result, ScrapeFailure):
            retry_note = " (after retry)" if scrape_result.retried else ""
            print(f"  ❌ {scrape_result.reason}{retry_note}: {scrape_result.detail}")
            failed.append({
                "url":    url,
                "reason": scrape_result.reason,
                "detail": scrape_result.detail,
            })

    # ── Step 2: Report failures clearly ───────────────────────────
    if failed:
        print(f"\n[Research Agent] ⚠️  {len(failed)} URLs failed:")
        for f in failed:
            print(f"  • [{f['reason']}] {f['url']}")
            print(f"    └─ {f['detail']}")

    print(f"\n[Research Agent] Usable sources: {len(successful)}/{len(state['search_results'])}")

    # ── Step 3: Abort if nothing scraped ──────────────────────────
    if not successful:
        return {
            "research_summary": "RESEARCH FAILED: No URLs could be scraped.",
            "scraped_content":  [],
            "error": f"All {len(failed)} URLs failed. Failures: {failed}",
            "current_agent":    "twitter_agent",
        }

    # ── Step 4: Build LLM context from successful scrapes ─────────
    combined = ""
    for i, page in enumerate(successful, 1):
        combined += f"\nSOURCE {i}: {page['url']}\n"
        combined += f"{page['content']}\n"
        combined += "-" * 40 + "\n"

    # ── Step 5: Synthesize with LLM ───────────────────────────────
    prompt = f"""You are a research analyst AI.

ORIGINAL TASK: {state['task']}

WEB CONTENT FROM {len(successful)} SOURCES:
{combined}

Write a structured research summary with:
1. KEY FINDINGS (5-7 bullet points of the most important facts)
2. MAIN THEMES (2-3 recurring themes across sources)
3. NOTABLE DATA POINTS (specific numbers, stats, quotes)
4. SOURCES USED (list URLs you found most useful)

Be specific. Use facts from the content. Do not make anything up."""

    print("[Research Agent] Synthesizing with LLM...")
    response = llm.invoke(prompt)
    research_summary = response.content.strip()
    print(f"[Research Agent] Summary generated ({len(research_summary)} chars)")

    return {
        "scraped_content":  [p["content"] for p in successful],
        "research_summary": research_summary,
        "current_agent":    "twitter_agent",
        "error":            None,   # explicitly clear any previous error
    }