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

# ── NOW ASYNC ─────────────────────────────────────────────────────
async def research_agent_node(state: PersonalAIState) -> dict:
    print(f"\n[Research Agent] Processing {len(state['search_results'])} URLs...")

    successful = []
    failed     = []

    # ── Scrape all URLs ───────────────────────────────────────────
    for i, result in enumerate(state["search_results"]):
        url = result["url"]
        print(f"[Research Agent] Scraping {i+1}/{len(state['search_results'])}: {url}")

        scrape_result = scrape_with_retry(url, max_chars=1000)

        if isinstance(scrape_result, ScrapeSuccess):
            print(f"  ✅ {scrape_result.chars} chars")
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

    # ── Report failures ───────────────────────────────────────────
    if failed:
        print(f"\n[Research Agent] ⚠️  {len(failed)} URLs failed:")
        for f in failed:
            print(f"  • [{f['reason']}] {f['url']}")
            print(f"    └─ {f['detail']}")

    print(f"\n[Research Agent] Usable: {len(successful)}/{len(state['search_results'])}")

    # ── Abort if nothing scraped ──────────────────────────────────
    if not successful:
        return {
            "research_summary": "RESEARCH FAILED: No URLs could be scraped.",
            "scraped_content":  [],
            "error": f"All {len(failed)} URLs failed.",
            "current_agent": "publisher_agent",
        }

    # ── Build context ─────────────────────────────────────────────
    combined = ""
    for i, page in enumerate(successful, 1):
        combined += f"\nSOURCE {i}: {page['url']}\n"
        combined += f"{page['content']}\n"
        combined += "-" * 40 + "\n"

    prompt = f"""You are a research analyst AI.

ORIGINAL TASK: {state['task']}

WEB CONTENT FROM {len(successful)} SOURCES:
{combined}

Write a structured research summary with:
1. KEY FINDINGS (5-7 bullet points)
2. MAIN THEMES (2-3 themes)
3. NOTABLE DATA POINTS (specific numbers, stats)
4. SOURCES USED (list URLs)

Be specific. Facts only. No fabrication."""

    # ── TOKEN STREAMING ───────────────────────────────────────────
    print("[Research Agent] Streaming summary: ", end="", flush=True)

    research_summary = ""                           # empty bucket

    async for token_chunk in llm.astream(prompt):  # each drip
        token = token_chunk.content
        research_summary += token                   # fill bucket
        print(token, end="", flush=True)            # show immediately

    print()  # newline when done
    print(f"[Research Agent] ✅ Complete ({len(research_summary)} chars)")

    return {
        "scraped_content":  [p["content"] for p in successful],
        "research_summary": research_summary.strip(),
        "current_agent":    "publisher_agent",
        "error":            None,
    }