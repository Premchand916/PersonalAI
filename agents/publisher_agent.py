# agents/publisher_agent.py
# (rename from twitter_agent.py — publisher is generic, not Twitter-specific)

import os
import re
from langchain_ollama import ChatOllama
from orchestrator.state import PersonalAIState
from tools.telegram_tool import post_thread, PostSuccess, PostFailure
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    temperature=0.7,        # creative writing — higher temp is correct here
)

def publisher_agent_node(state: PersonalAIState) -> dict:
    """
    Publisher Agent Node.
    Formats research into a thread and posts to Telegram.

    Reads from State:  research_summary, task
    Writes to State:   twitter_thread, final_status, error
    """
    print(f"\n[Publisher Agent] Formatting research into thread...")

    # ── Guard: need research summary to proceed ────────────────────
    if not state.get("research_summary"):
        return {
            "final_status": "failed",
            "error":        "No research summary in State — research agent may have failed",
        }

    # ── Step 1: LLM formats research into thread ───────────────────
    prompt = f"""You are a social media expert writing a Telegram thread.

RESEARCH SUMMARY:
{state['research_summary']}

ORIGINAL TOPIC: {state['task']}

Write a thread with exactly 5 posts.

STRICT RULES:
- Each post MUST be under 270 characters
- Post 1: Hook — one powerful insight that makes people want to read more
- Post 2-4: One key finding per post with a specific fact or stat
- Post 5: Conclusion + one actionable takeaway
- Number each post: start with "1/" "2/" "3/" etc
- Write like a senior engineer sharing real knowledge
- Be direct, no fluff, no filler words

Return ONLY the 5 posts separated by "---". No other text.

Example format:
1/ Hook post here under 270 chars
---
2/ Second post here
---
3/ Third post here
---
4/ Fourth post here
---
5/ Final post here"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # ── Step 2: Parse posts from LLM output ───────────────────────
    posts = [p.strip() for p in raw.split("---") if p.strip()]

    # Fallback if LLM didn't use "---" separator
    if len(posts) < 3:
        posts = re.findall(r'\d/[^\n]*(?:\n(?!\d/)[^\n]*)*', raw)
        posts = [p.strip() for p in posts if p.strip()]

    # Last resort — use full response as single post
    if not posts:
        posts = [raw[:270]]

    print(f"[Publisher Agent] Generated {len(posts)} posts")

    # ── Step 3: Validate length ────────────────────────────────────
    validated = []
    for i, post in enumerate(posts):
        if len(post) > 280:
            print(f"  ⚠️  Post {i+1} too long ({len(post)} chars) — trimming")
            post = post[:277] + "..."
        validated.append(post)
        print(f"  Post {i+1} ({len(post)} chars): {post[:60]}...")

    # ── Step 4: Post to Telegram ───────────────────────────────────
    print(f"\n[Publisher Agent] Posting to Telegram...")
    result = post_thread(validated)

    if isinstance(result, PostSuccess):
        print(f"[Publisher Agent] ✅ Posted {result.count} messages")
        print(f"  Preview: {result.preview}...")
        return {
            "twitter_thread": validated,    # keeping field name for State compat
            "final_status":   "completed",
            "error":          None,
        }

    elif isinstance(result, PostFailure):
        print(f"[Publisher Agent] ❌ {result.reason}: {result.detail}")
        return {
            "twitter_thread": validated,    # save thread even if posting failed
            "final_status":   "post_failed",
            "error":          f"{result.reason}: {result.detail}",
        }