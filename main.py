# main.py

from orchestrator.graph import build_graph


def run(task: str, post_to_telegram: bool = False):
    graph = build_graph()

    # Initial state — only fields defined in PersonalAIState
    # post_to_telegram is passed as function arg, not in state
    initial_state = {
        "task":             task,
        "search_queries":   [],
        "search_results":   [],
        "scraped_content":  [],
        "research_summary": "",
        "thread":           [],
        "final_status":     "in_progress",
        "messages":         [],
        "error":            None,
        "current_agent":    "search_agent",
    }

    print(f"\n🚀 Starting PersonalAI pipeline...")
    print(f"   Task: {task}")
    print(f"   Telegram: {'enabled' if post_to_telegram else 'disabled'}")

    result = graph.invoke(initial_state)

    # ── STATUS ────────────────────────────────────────────────────
    print("\n" + "="*50)
    status = result.get("final_status", "unknown")
    status_icon = "✅" if status == "completed" else "❌"
    print(f"{status_icon} STATUS: {status.upper()}")
    print("="*50)

    # ── ERROR (if any) ────────────────────────────────────────────
    if result.get("error"):
        print(f"\n⚠️  ERROR: {result['error']}")

    # ── SEARCH RESULTS ────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"🔍 SEARCH RESULTS ({len(result.get('search_results', []))} found):")
    print("="*50)

    if not result.get("search_results"):
        print("No search results returned.")
    else:
        for i, item in enumerate(result["search_results"], 1):
            print(f"\n{i}. {item.get('title', 'No title')}")
            print(f"   URL: {item.get('url', 'No URL')}")
            snippet = item.get('snippet', '')
            if snippet:
                print(f"   {snippet[:150]}...")

    # ── RESEARCH SUMMARY ──────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"📚 RESEARCH SUMMARY:")
    print("="*50)
    summary = result.get("research_summary", "")
    print(summary if summary else "No research summary was generated.")

    # ── THREAD POSTS ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    thread = result.get("thread", [])
    print(f"📨 THREAD ({len(thread)} posts):")
    print("="*50)

    if not thread:
        print("No thread was generated.")
    else:
        for i, post in enumerate(thread, 1):
            print(f"\nPost {i} ({len(post)} chars):")
            print(post)

    # ── SUMMARY STATS ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"📊 PIPELINE STATS:")
    print(f"   Sources found:  {len(result.get('search_results', []))}")
    print(f"   Sources scraped:{len(result.get('scraped_content', []))}")
    print(f"   Thread posts:   {len(thread)}")
    print("="*50)

    return result


if __name__ == "__main__":
    run(
        task="latest AI agent frameworks and tools 2026",
        post_to_telegram=False   # change to True when you want Telegram
    )