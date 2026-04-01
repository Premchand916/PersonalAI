# orchestrator/state.py

from typing import TypedDict, Annotated, List, Optional
from operator import add                    # We'll explain this below

class PersonalAIState(TypedDict):
    # ── INPUT ──────────────────────────────────────────
    task: str                               # What the user asked for
    
    # ── SEARCH LAYER ───────────────────────────────────
    search_queries: List[str]               # Queries sent to Tavily
    search_results: List[dict]              # Raw URLs + snippets returned
    
    # ── RESEARCH LAYER ──────────────────────────────────
    scraped_content: List[str]              # Full text from each URL
    research_summary: str                   # Condensed facts by research agent
    
    # ── OUTPUT LAYER ───────────────────────────────────
    twitter_thread: List[str]               # Each tweet as a list item
    final_status: str                       # "completed" / "failed" / "in_progress"
    
    # ── SYSTEM ─────────────────────────────────────────
    messages: Annotated[List[dict], add]    # Full message history (special - explained below)
    error: Optional[str]                    # If something goes wrong, store why
    current_agent: str                      # Which agent is running right now