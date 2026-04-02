# tools/telegram_tool.py

import os
import asyncio
import telegram
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# ── Result types — same pattern as scraper ─────────────────────────
@dataclass
class PostSuccess:
    message_ids: list[int]
    count:       int
    preview:     str        # first 100 chars of thread for confirmation

@dataclass
class PostFailure:
    reason: str
    detail: str


async def _send_thread(bot: telegram.Bot, chat_id: str, tweets: list[str]) -> PostSuccess:
    """
    Internal async function that sends each tweet as a separate message.
    Telegram doesn't have 'threads' like Twitter — we send them
    as sequential messages to simulate a thread feel.
    """
    message_ids = []

    # Send header first — tells user a thread is coming
    header = await bot.send_message(
        chat_id=chat_id,
        text="🤖 *PersonalAI Research Thread*",
        parse_mode="Markdown"
    )
    message_ids.append(header.message_id)

    # Send each tweet as a separate message
    for i, tweet_text in enumerate(tweets, 1):
        # Telegram limit is 4096 chars — much more generous than Twitter
        # But we keep our 280 char discipline for clean readable posts
        msg = await bot.send_message(
            chat_id=chat_id,
            text=tweet_text,
        )
        message_ids.append(msg.message_id)

    # Send footer
    footer = await bot.send_message(
        chat_id=chat_id,
        text=f"✅ *Thread complete — {len(tweets)} posts*\n_Powered by PersonalAI_",
        parse_mode="Markdown"
    )
    message_ids.append(footer.message_id)

    return PostSuccess(
        message_ids=message_ids,
        count=len(tweets),
        preview=tweets[0][:100] if tweets else ""
    )


def post_thread(tweets: list[str]) -> PostSuccess | PostFailure:
    """
    Posts a list of messages to Telegram as a thread.
    Synchronous wrapper around the async Telegram API.

    Why async wrapper?
    Telegram's Python library is async-first.
    Our LangGraph nodes are sync.
    asyncio.run() bridges the two worlds.
    """
    if not tweets:
        return PostFailure(reason="EMPTY", detail="No tweets to post")

    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    # Validate env vars exist before trying to connect
    if not token:
        return PostFailure(
            reason="CONFIG_ERROR",
            detail="TELEGRAM_BOT_TOKEN missing from .env"
        )
    if not chat_id:
        return PostFailure(
            reason="CONFIG_ERROR",
            detail="TELEGRAM_CHAT_ID missing from .env"
        )

    try:
        bot = telegram.Bot(token=token)

        # Run async function from sync context
        result = asyncio.run(_send_thread(bot, chat_id, tweets))
        return result

    except telegram.error.InvalidToken:
        return PostFailure(
            reason="AUTH_FAILED",
            detail="Bot token is invalid — check TELEGRAM_BOT_TOKEN in .env"
        )

    except telegram.error.ChatNotFound:
        return PostFailure(
            reason="CHAT_NOT_FOUND",
            detail="Chat ID not found — did you message your bot first?"
        )

    except telegram.error.RetryAfter as e:
        return PostFailure(
            reason="RATE_LIMITED",
            detail=f"Telegram says wait {e.retry_after} seconds before posting again"
        )

    except telegram.error.NetworkError as e:
        return PostFailure(
            reason="NETWORK_ERROR",
            detail=f"Connection failed: {str(e)}"
        )

    except Exception as e:
        return PostFailure(reason="UNKNOWN", detail=str(e))


# ── Quick test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_tweets = [
        "1/ Testing PersonalAI Telegram integration 🤖",
        "2/ This is the second message in the thread",
        "3/ If you see this — it works! ✅"
    ]
    result = post_thread(test_tweets)
    print(result)