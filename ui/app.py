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
    """
    Runs when user opens the chat.
    Sets up the session and shows welcome message.
    """
    # Store graph in user session
    # Why? Each user gets their own session — no state mixing between users
    cl.user_session.set("graph", graph)

    await cl.Message(
        content=(
            "👋 **Welcome to PersonalAI**\n\n"
            "Your personal Perplexity Computer — powered by LangGraph + Ollama.\n\n"
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
    Executes the full pipeline with real-time progress updates.
    """
    task = message.content.strip()

    if not task:
        await cl.Message(content="Please enter a research topic.").send()
        return

    # ── Step 1: Show immediate feedback ───────────────────────────
    # User sees this instantly — system doesn't look frozen
    thinking = await cl.Message(content="🔍 Starting research pipeline...").send()

    # ── Step 2: Build initial state ───────────────────────────────
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

    # ── Step 3: Run pipeline with step-by-step updates ────────────
    try:
        # Search step
        await cl.Message(content="🔍 **Step 1/3** — Searching the web...").send()

        # Run graph in thread (same pattern as FastAPI)
        import asyncio
        result = await asyncio.to_thread(graph.invoke, initial_state)

        # ── Step 4: Show results progressively ────────────────────

        # Search results
        search_results = result.get("search_results", [])
        search_msg = f"✅ **Step 1/3 Complete** — Found {len(search_results)} sources\n\n"
        for i, r in enumerate(search_results, 1):
            search_msg += f"{i}. [{r.get('title', 'No title')}]({r.get('url', '')})\n"
        await cl.Message(content=search_msg).send()

        # Research summary
        await cl.Message(content="📚 **Step 2/3 Complete** — Research synthesized\n").send()

        summary = result.get("research_summary", "")
        if summary:
            await cl.Message(content=f"### 📋 Research Summary\n\n{summary}").send()

        # Thread
        thread = result.get("thread", [])
        await cl.Message(
            content=f"✍️ **Step 3/3 Complete** — Thread generated & posted to Telegram"
        ).send()

        if thread:
            thread_msg = "### 📨 Your Research Thread\n\n"
            for i, post in enumerate(thread, 1):
                thread_msg += f"**Post {i}** ({len(post)} chars)\n{post}\n\n"
                thread_msg += "---\n"
            await cl.Message(content=thread_msg).send()

        # Stats
        scraped = result.get("scraped_content", [])
        status  = result.get("final_status", "unknown")
        error   = result.get("error")

        stats_msg = (
            f"### 📊 Pipeline Stats\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Status | {'✅ ' if status == 'completed' else '❌ '}{status} |\n"
            f"| Sources Found | {len(search_results)} |\n"
            f"| Sources Scraped | {len(scraped)} |\n"
            f"| Thread Posts | {len(thread)} |\n"
            f"| Telegram | {'✅ Posted' if status == 'completed' else '❌ Failed'} |\n"
        )

        if error:
            stats_msg += f"\n⚠️ **Error:** {error}"

        await cl.Message(content=stats_msg).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ **Pipeline failed**\n\n```\n{str(e)}\n```\n\nPlease try again."
        ).send()