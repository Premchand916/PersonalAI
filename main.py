# main.py
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from tools.secrets_guard import validate_env
validate_env("main.py")
from orchestrator.graph import build_graph
from memory.store import init_db, save_run

def run(task: str, post_to_telegram: bool = False):
    graph  = build_graph()
    init_db()                                  # ← ADD THIS (safe to call every time)

    initial_state = {
        "task":             task,
        "post_to_telegram": post_to_telegram,
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

    result = graph.invoke(initial_state)

    # ── SAVE TO MEMORY ─────────────────────────────────────────────
    run_id = save_run(                         # ← ADD THIS BLOCK
        task=    task,
        summary= result.get("research_summary", ""),
        thread=  result.get("thread") or result.get("twitter_thread", []),
        status=  result.get("final_status", "unknown"),
        sources= len(result.get("scraped_content", [])),
    )
    print(f"\n[Memory] Run saved as #{run_id}")
   

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
    thread = result.get("thread") or result.get("twitter_thread", [])
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
