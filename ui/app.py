# ui/app.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import chainlit as cl
from orchestrator.graph import build_graph
from orchestrator.state import PersonalAIState

graph = build_graph()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("graph", graph)

    await cl.Message(
        content=(
            "👋 **Welcome to PersonalAI**\n\n"
            "Your personal Perplexity Computer — "
            "powered by LangGraph + Ollama.\n\n"
            "**What I can do:**\n"
            "- 🔍 Search the web for any topic\n"
            "- 📚 Read and synthesize multiple sources\n"
            "- 📨 Post a research thread to Telegram\n\n"
            "**Just type any research topic to begin.**\n"
            "Example: `latest developments in LangGraph 2026`"
        )
    ).send()


async def stream_text_to_ui(text: str, prefix: str = ""):
    """
    Takes completed text and streams it word by word into UI.
    Simulates token streaming from already-generated text.
    
    Why do this instead of passing llm directly to UI?
    → Agents must stay UI-agnostic (no Chainlit imports in agents)
    → Agent generates text, UI decides HOW to display it
    → Clean separation of concerns
    
    Real world analogy:
    Chef cooks the meal (agent).
    Waiter presents it beautifully (UI).
    Chef doesn't care about presentation.
    Waiter doesn't cook.
    """
    msg = cl.Message(content=prefix)
    await msg.send()                        # open message bubble

    # Stream word by word with small delay for effect
    words = text.split(" ")
    for word in words:
        await msg.stream_token(word + " ") # push word into bubble
        await asyncio.sleep(0.01)          # tiny delay = smooth effect

    await msg.update()                     # finalize message
    return msg


@cl.on_message
async def on_message(message: cl.Message):
    task = message.content.strip()

    if not task:
        await cl.Message(content="Please enter a research topic.").send()
        return

    # Initial state
    initial_state: PersonalAIState = {
        "task":             task,
        "post_to_telegram": False,
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

    # Kick off immediately
    await cl.Message(
        content=f"🚀 **Starting PersonalAI pipeline**\n> {task}"
    ).send()

    final_search_results = []
    final_thread         = []
    final_status         = "unknown"
    final_error          = None

    try:
        async for chunk in graph.astream(initial_state):

            # ── SEARCH AGENT CHUNK ─────────────────────────────────
            if "search_agent" in chunk:
                data    = chunk["search_agent"]
                results = data.get("search_results", [])
                queries = data.get("search_queries", [])

                final_search_results = results

                # Build sources list
                queries_text = "\n".join(
                    [f"  - `{q}`" for q in queries]
                )
                sources_text = "\n".join(
                    [
                        f"{i}. [{r.get('title','No title')}]({r.get('url','')})"
                        for i, r in enumerate(results, 1)
                    ]
                )

                await cl.Message(
                    content=(
                        f"✅ **Search Complete**\n\n"
                        f"**Queries:**\n{queries_text}\n\n"
                        f"**Sources ({len(results)} found):**\n"
                        f"{sources_text}"
                    )
                ).send()

            # ── RESEARCH AGENT CHUNK ───────────────────────────────
            elif "research_agent" in chunk:
                data    = chunk["research_agent"]
                summary = data.get("research_summary", "")
                scraped = data.get("scraped_content", [])
                error   = data.get("error")

                total         = len(final_search_results)
                scraped_count = len(scraped)
                blocked       = total - scraped_count

                # Show scrape stats — normal message
                await cl.Message(
                    content=(
                        f"✅ **Research Complete**\n\n"
                        f"| Metric | Value |\n"
                        f"|--------|-------|\n"
                        f"| URLs attempted | {total} |\n"
                        f"| Scraped | {scraped_count} |\n"
                        f"| Blocked | {blocked} |\n"
                        f"| Summary | {len(summary)} chars |\n"
                    )
                ).send()

                if error:
                    await cl.Message(
                        content=f"⚠️ **Warning:** {error}"
                    ).send()

                # ── STREAM SUMMARY WORD BY WORD INTO UI ───────────
                if summary:
                    await stream_text_to_ui(
                        text=summary,
                        prefix="### 📋 Research Summary\n\n"
                    )

            # ── PUBLISHER AGENT CHUNK ──────────────────────────────
            elif "publisher_agent" in chunk:
                data   = chunk["publisher_agent"]
                thread = data.get("thread") or data.get("twitter_thread", [])
                status = data.get("final_status", "unknown")
                error  = data.get("error")

                final_thread  = thread
                final_status  = status
                final_error   = error

                # Telegram status
                if status == "completed":
                    await cl.Message(
                        content="✅ **Published to Telegram**"
                    ).send()
                else:
                    await cl.Message(
                        content=f"⚠️ **Telegram failed:** {error}"
                    ).send()

                # ── STREAM EACH THREAD POST WORD BY WORD ──────────
                if thread:
                    await cl.Message(
                        content="### 📨 Your Research Thread"
                    ).send()

                    for i, post in enumerate(thread, 1):
                        # Each post streams independently
                        await stream_text_to_ui(
                            text=post,
                            prefix=f"**Post {i}** ({len(post)} chars)\n\n"
                        )
                        # Small gap between posts
                        await asyncio.sleep(0.3)

        # ── FINAL STATS ────────────────────────────────────────────
        status_icon = "✅" if final_status == "completed" else "❌"
        await cl.Message(
            content=(
                f"### 📊 Pipeline Complete\n\n"
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Status | {status_icon} {final_status} |\n"
                f"| Sources found | {len(final_search_results)} |\n"
                f"| Thread posts | {len(final_thread)} |\n"
                f"| Telegram | "
                f"{'✅ Posted' if final_status == 'completed' else '❌ Failed'} |\n"
            )
        ).send()

    except Exception as e:
        await cl.Message(
            content=(
                f"❌ **Pipeline failed**\n\n"
                f"```\n{str(e)}\n```"
            )
        ).send()
        import traceback
        traceback.print_exc()
