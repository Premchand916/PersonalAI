# agents/publisher_agent.py

import os
import re
from langchain_ollama import ChatOllama
from orchestrator.state import PersonalAIState
from tools.telegram_tool import post_thread, PostSuccess, PostFailure
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    temperature=0.7,
)

# ── NOW ASYNC ─────────────────────────────────────────────────────
async def publisher_agent_node(state: PersonalAIState) -> dict:
    print(f"\n[Publisher Agent] Formatting into thread...")

    if not state.get("research_summary"):
        return {
            "final_status": "failed",
            "error": "No research summary available",
        }

    prompt = f"""You are a social media expert writing a Telegram thread.

RESEARCH SUMMARY:
{state['research_summary']}

ORIGINAL TOPIC: {state['task']}

Write a thread with exactly 5 posts.

RULES:
- Each post MUST be under 270 characters
- Post 1: Hook — powerful opening insight
- Post 2-4: One key finding per post with specific fact
- Post 5: Conclusion + actionable takeaway
- Number each: "1/" "2/" "3/" etc
- No hashtags, no filler words

Return ONLY 5 posts separated by "---". Nothing else."""

    # ── TOKEN STREAMING ───────────────────────────────────────────
    print("[Publisher Agent] Streaming thread: ", end="", flush=True)

    raw = ""                                        # empty bucket

    async for token_chunk in llm.astream(prompt):  # each drip
        token = token_chunk.content
        raw += token                                # fill bucket
        print(token, end="", flush=True)            # show immediately

    print()  # newline when done

    # ── Parse posts ───────────────────────────────────────────────
    posts = [p.strip() for p in raw.split("---") if p.strip()]

    if len(posts) < 3:
        posts = re.findall(r'\d/[^\n]*(?:\n(?!\d/)[^\n]*)*', raw)
        posts = [p.strip() for p in posts if p.strip()]

    if not posts:
        posts = [raw[:270]]

    print(f"[Publisher Agent] Generated {len(posts)} posts")

    # ── Validate length ───────────────────────────────────────────
    validated = []
    for i, post in enumerate(posts):
        if len(post) > 280:
            post = post[:277] + "..."
        validated.append(post)
        print(f"  Post {i+1} ({len(post)} chars): {post[:60]}...")

    # ── Post to Telegram ──────────────────────────────────────────
    print(f"\n[Publisher Agent] Posting to Telegram...")
    result = post_thread(validated)

    if isinstance(result, PostSuccess):
        print(f"[Publisher Agent] ✅ Posted {result.count} messages")
        return {
            "thread":       validated,
            "final_status": "completed",
            "error":        None,
        }

    elif isinstance(result, PostFailure):
        print(f"[Publisher Agent] ❌ {result.reason}: {result.detail}")
        return {
            "thread":       validated,
            "final_status": "post_failed",
            "error":        f"{result.reason}: {result.detail}",
        }