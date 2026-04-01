# agents/research_agent.py

import os
from langchain_ollama import ChatOllama
from orchestrator.state import PersonalAIState
from tools.scraper import scrape_url
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    temperature=0,      # research = precision, not creativity
)

def research_agent_node(state: PersonalAIState) -> dict:
    """
    Research Agent Node.
    
    Reads from State:  search_results (list of URLs + snippets)
    Writes to State:   scraped_content, research_summary
    """
    print(f"\n[Research Agent] Processing {len(state['search_results'])} URLs...")
    
    # ── Step 1: Scrape each URL ────────────────────────────────────
    scraped = []
    for i, result in enumerate(state["search_results"]):
        url = result["url"]
        print(f"[Research Agent] Scraping {i+1}/{len(state['search_results'])}: {url}")
        
        content = scrape_url(url)
        
        # Skip failed scrapes
        if content.startswith("[SCRAPE FAILED]"):
            print(f"[Research Agent] Skipped: {content}")
            continue
        
        # Store with source URL for citation
        scraped.append({
            "url":     url,
            "title":   result.get("title", ""),
            "content": content,
        })
    
    print(f"[Research Agent] Successfully scraped {len(scraped)} pages")
    
    # ── Step 2: Build context for LLM ─────────────────────────────
    # Combine all scraped content into one block
    # Format: SOURCE 1: [url]\n[content]\n\nSOURCE 2: ...
    combined = ""
    for i, page in enumerate(scraped, 1):
        combined += f"\nSOURCE {i}: {page['url']}\n"
        combined += f"{page['content']}\n"
        combined += "-" * 40 + "\n"
    
    # ── Step 3: Ask LLM to synthesize intelligence ─────────────────
    prompt = f"""You are a research analyst AI. Analyze the following web content 
and extract the most important facts, insights, and findings.

ORIGINAL TASK: {state['task']}

WEB CONTENT:
{combined}

Write a structured research summary with:
1. KEY FINDINGS (5-7 bullet points of the most important facts)
2. MAIN THEMES (2-3 recurring themes across sources)
3. NOTABLE QUOTES OR DATA POINTS (specific numbers, stats, quotes)
4. SOURCES USED (list the URLs you found most useful)

Be specific. Use facts from the content. Do not make anything up."""

    print("[Research Agent] Synthesizing with LLM...")
    response = llm.invoke(prompt)
    
    research_summary = response.content.strip()
    print(f"[Research Agent] Summary generated ({len(research_summary)} chars)")
    
    return {
        "scraped_content":  [p["content"] for p in scraped],
        "research_summary": research_summary,
        "current_agent":    "twitter_agent",
    }