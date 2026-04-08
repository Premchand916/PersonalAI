import re

from orchestrator.state import PersonalAIState
from tools.ollama_client import LLMInvocationError, invoke_prompt
from tools.telegram_tool import PostFailure, PostSuccess, post_thread

MAX_POST_LENGTH = 270


def _clip_post(index: int, text: str) -> str:
    prefix = f"{index}/ "
    available = MAX_POST_LENGTH - len(prefix)
    clipped = " ".join(text.split())
    if len(clipped) > available:
        clipped = clipped[: available - 3].rstrip() + "..."
    return prefix + clipped


def _fallback_posts(task: str, research_summary: str) -> list[str]:
    points = []

    for raw_line in research_summary.splitlines():
        line = re.sub(r"^[\-\*\d\.\)\s/]+", "", raw_line).strip()
        if not line:
            continue
        if line.isupper() or line.startswith("NOTE"):
            continue
        points.append(line)

    while len(points) < 3:
        points.append("Review the linked sources and validate the strongest claim before sharing.")

    return [
        _clip_post(1, f"{task}: here are the key takeaways worth paying attention to."),
        _clip_post(2, points[0]),
        _clip_post(3, points[1]),
        _clip_post(4, points[2]),
        _clip_post(5, "Next step: verify the top sources, refine the summary, and publish only the strongest facts."),
    ]


def _parse_posts(raw: str) -> list[str]:
    posts = [post.strip() for post in raw.split("---") if post.strip()]

    if len(posts) < 3:
        posts = re.findall(r"\d/[^\n]*(?:\n(?!\d/)[^\n]*)*", raw)
        posts = [post.strip() for post in posts if post.strip()]

    if not posts:
        posts = [raw[:MAX_POST_LENGTH]]

    return posts[:5]


def publisher_agent_node(state: PersonalAIState) -> dict:
    print("\n[Publisher Agent] Formatting research into thread...")

    if not state.get("research_summary"):
        return {
            "final_status": "failed",
            "error": state.get("error") or "No research summary in state.",
        }

    prompt = f"""You are a social media expert writing a Telegram thread.

RESEARCH SUMMARY:
{state['research_summary']}

ORIGINAL TOPIC: {state['task']}

Write a thread with exactly 5 posts.

STRICT RULES:
- Each post MUST be under {MAX_POST_LENGTH} characters
- Post 1: Hook - one powerful insight that makes people want to read more
- Post 2-4: One key finding per post with a specific fact or stat
- Post 5: Conclusion + one actionable takeaway
- Number each post: start with "1/" "2/" "3/" etc
- Write like a senior engineer sharing real knowledge
- Be direct, no fluff, no filler words

Return ONLY the 5 posts separated by "---". No other text.

Example format:
1/ Hook post here under {MAX_POST_LENGTH} chars
---
2/ Second post here
---
3/ Third post here
---
4/ Fourth post here
---
5/ Final post here"""

    try:
        raw = invoke_prompt(prompt, temperature=0.7)
        posts = _parse_posts(raw)
    except LLMInvocationError as exc:
        print(f"[Publisher Agent] LLM unavailable, using fallback thread: {exc}")
        posts = _fallback_posts(state["task"], state["research_summary"])

    print(f"[Publisher Agent] Generated {len(posts)} posts")

    validated = []
    for i, post in enumerate(posts, 1):
        if len(post) > MAX_POST_LENGTH:
            print(f"  [!] Post {i} too long ({len(post)} chars) - trimming")
            post = post[: MAX_POST_LENGTH - 3] + "..."
        validated.append(post)
        print(f"  Post {i} ({len(post)} chars): {post[:60]}...")

    if not state.get("post_to_telegram", True):
        print("[Publisher Agent] Posting skipped by request")
        return {
            "thread": validated,    # ← renamed
            "final_status": "completed",
            "error": None,
        }

    print("\n[Publisher Agent] Posting to Telegram...")
    result = post_thread(validated)

    if isinstance(result, PostSuccess):
        print(f"[Publisher Agent] [OK] Posted {result.count} messages")
        print(f"  Preview: {result.preview}...")
        return {
            "thread":  validated,    # ← renamed
            "final_status": "completed",
            "error": None,
        }

    if isinstance(result, PostFailure):
        print(f"[Publisher Agent] [X] {result.reason}: {result.detail}")
        return {
            "thread": validated,
            "final_status": "post_failed",
            "error": f"{result.reason}: {result.detail}",
        }

    return {
        "thread": validated,
        "final_status": "post_failed",
        "error": "Unexpected publisher result.",
    }
