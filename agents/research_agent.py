import re

from orchestrator.state import PersonalAIState
from tools.ollama_client import LLMInvocationError, invoke_prompt
from tools.scraper import ScrapeFailure, ScrapeSuccess, scrape_with_retry


def _extract_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _fallback_summary(task: str, pages: list[dict]) -> str:
    findings = []
    data_points = []
    themes = []
    sources = []

    for page in pages[:5]:
        sources.append(f"- {page['url']}")
        title = page.get("title") or page["url"]
        themes.append(f"- {title}")

        for sentence in _extract_sentences(page["content"]):
            if len(findings) < 5:
                findings.append(f"- {sentence}")
            if any(char.isdigit() for char in sentence) and len(data_points) < 3:
                data_points.append(f"- {sentence}")
            if len(findings) >= 5 and len(data_points) >= 3:
                break

    if not findings:
        findings.append(
            "- Fallback summary mode was used, but the scraped pages did not yield"
            " enough clean text to summarize."
        )

    if not data_points:
        data_points.append("- No clear numeric data points were extracted in fallback mode.")

    return "\n".join(
        [
            f"KEY FINDINGS FOR: {task}",
            *findings[:5],
            "",
            "MAIN THEMES",
            *themes[:3],
            "",
            "NOTABLE DATA POINTS",
            *data_points[:3],
            "",
            "SOURCES USED",
            *sources[:5],
            "",
            "NOTE",
            "- This summary was generated without Ollama because the local model was unavailable.",
        ]
    )


def research_agent_node(state: PersonalAIState) -> dict:
    print(f"\n[Research Agent] Processing {len(state['search_results'])} URLs...")

    if not state["search_results"]:
        return {
            "research_summary": "",
            "scraped_content": [],
            "final_status": "failed",
            "error": state.get("error")
            or (
                "No search results were returned. Check Tavily configuration, "
                "network access, or try a broader task."
            ),
            "current_agent": "publisher_agent",
        }

    successful = []
    failed = []

    for i, result in enumerate(state["search_results"]):
        url = result["url"]
        print(f"[Research Agent] Scraping {i + 1}/{len(state['search_results'])}: {url}")

        scrape_result = scrape_with_retry(url, max_chars=1000)

        if isinstance(scrape_result, ScrapeSuccess):
            print(f"  [OK] Success - {scrape_result.chars} chars extracted")
            successful.append({
                "url": url,
                "title": result.get("title", ""),
                "content": scrape_result.content,
            })
        elif isinstance(scrape_result, ScrapeFailure):
            retry_note = " (after retry)" if scrape_result.retried else ""
            print(f"  [X] {scrape_result.reason}{retry_note}: {scrape_result.detail}")
            failed.append({
                "url": url,
                "reason": scrape_result.reason,
                "detail": scrape_result.detail,
            })

    if failed:
        print(f"\n[Research Agent] {len(failed)} URLs failed:")
        for failure in failed:
            print(f"  - [{failure['reason']}] {failure['url']}")
            print(f"    -> {failure['detail']}")

    print(
        f"\n[Research Agent] Usable sources: "
        f"{len(successful)}/{len(state['search_results'])}"
    )

    if not successful:
        return {
            "research_summary": "",
            "scraped_content": [],
            "final_status": "failed",
            "error": (
                f"All {len(failed)} URLs failed to scrape. "
                "Review the target sites, scraper network access, or retry later."
            ),
            "current_agent": "publisher_agent",
        }

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
1. KEY FINDINGS (5-7 bullet points of the most important facts)
2. MAIN THEMES (2-3 recurring themes across sources)
3. NOTABLE DATA POINTS (specific numbers, stats, quotes)
4. SOURCES USED (list URLs you found most useful)

Be specific. Use facts from the content. Do not make anything up."""

    print("[Research Agent] Synthesizing with LLM...")
    try:
        research_summary = invoke_prompt(prompt, temperature=0)
    except LLMInvocationError as exc:
        print(f"[Research Agent] LLM unavailable, using fallback summary: {exc}")
        research_summary = _fallback_summary(state["task"], successful)
    print(f"[Research Agent] Summary generated ({len(research_summary)} chars)")

    return {
        "scraped_content": [page["content"] for page in successful],
        "research_summary": research_summary,
        "current_agent": "publisher_agent",
        "error": None,
    }
