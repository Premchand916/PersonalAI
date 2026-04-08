# ui/app.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chainlit as cl
from orchestrator.graph import build_graph

# Build graph once at startup
graph = build_graph()


@cl.on_chat_start
async def on_chat_start():
    """Runs when user opens the chat."""
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


@cl.on_message
async def on_message(message: cl.Message):
    """
    Runs every time user sends a message.
    Uses graph.astream() to show progress after each agent.
    """
    task = message.content.strip()

    if not task:
        await cl.Message(content="Please enter a research topic.").send()
        return

    # ── Initial state ──────────────────────────────────────────────
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

    # ── Kick off with immediate feedback ──────────────────────────
    # User sees this instantly — no frozen screen
    await cl.Message(
        content=f"🚀 **Starting research pipeline...**\n`{task}`"
    ).send()

    # ── These accumulate as agents finish ─────────────────────────
    # We need them at the end to show final results
    final_search_results  = []
    final_summary         = ""
    final_thread          = []
    final_status          = "unknown"
    final_error           = None

    try:
        # ── THE KEY CHANGE: astream instead of invoke ─────────────
        # astream() is async — no need for asyncio.to_thread()
        # It yields one chunk per agent as each finishes
        async for chunk in graph.astream(initial_state):

            # ── CHUNK FROM SEARCH AGENT ───────────────────────────
            if "search_agent" in chunk:
                data    = chunk["search_agent"]
                results = data.get("search_results", [])
                queries = data.get("search_queries", [])

                # Save for final display
                final_search_results = results

                # Show immediately — user sees this after ~3 seconds
                queries_text = "\n".join(
                    [f"  - `{q}`" for q in queries]
                )
                sources_text = "\n".join(
                    [f"{i}. [{r.get('title','No title')}]({r.get('url','')})"
                     for i, r in enumerate(results, 1)]
                )

                await cl.Message(
                    content=(
                        f"✅ **Search Complete**\n\n"
                        f"**Queries used:**\n{queries_text}\n\n"
                        f"**Sources found ({len(results)}):**\n"
                        f"{sources_text}"
                    )
                ).send()

            # ── CHUNK FROM RESEARCH AGENT ─────────────────────────
            elif "research_agent" in chunk:
                data    = chunk["research_agent"]
                summary = data.get("research_summary", "")
                scraped = data.get("scraped_content", [])
                error   = data.get("error")

                # Save for stats
                final_summary = summary

                # Show scrape stats immediately
                total    = len(final_search_results)
                scraped_count = len(scraped)
                blocked  = total - scraped_count

                await cl.Message(
                    content=(
                        f"✅ **Research Complete**\n\n"
                        f"| Metric | Value |\n"
                        f"|--------|-------|\n"
                        f"| URLs attempted | {total} |\n"
                        f"| Successfully scraped | {scraped_count} |\n"
                        f"| Blocked (403) | {blocked} |\n"
                        f"| Summary length | {len(summary)} chars |\n"
                    )
                ).send()

                # Show error if research had issues
                if error:
                    await cl.Message(
                        content=f"⚠️ **Research warning:** {error}"
                    ).send()

                # Show full summary
                if summary:
                    await cl.Message(
                        content=f"### 📋 Research Summary\n\n{summary}"
                    ).send()

            # ── CHUNK FROM PUBLISHER AGENT ────────────────────────
            elif "publisher_agent" in chunk:
                data   = chunk["publisher_agent"]
                thread = data.get("thread", [])
                status = data.get("final_status", "unknown")
                error  = data.get("error")

                # Save for stats
                final_thread  = thread
                final_status  = status
                final_error   = error

                # Show publish status
                if status == "completed":
                    await cl.Message(
                        content="✅ **Published to Telegram successfully**"
                    ).send()
                elif status == "post_failed":
                    await cl.Message(
                        content=f"⚠️ **Telegram post failed:** {error}"
                    ).send()

                # Show the thread
                if thread:
                    thread_msg = "### 📨 Your Research Thread\n\n"
                    for i, post in enumerate(thread, 1):
                        thread_msg += (
                            f"**Post {i}** ({len(post)} chars)\n\n"
                            f"{post}\n\n"
                            f"---\n\n"
                        )
                    await cl.Message(content=thread_msg).send()

        # ── FINAL STATS — shown after all agents complete ──────────
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
                f"```\n{str(e)}\n```\n\n"
                f"Check terminal for full traceback."
            )
        ).send()
        # Also print to terminal for debugging
        import traceback
        traceback.print_exc()