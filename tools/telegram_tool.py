import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from dotenv import load_dotenv

try:
    import telegram as telegram_module
except ImportError:
    telegram_module = None

if TYPE_CHECKING:
    from telegram import Bot
else:
    Bot = Any

load_dotenv()


@dataclass
class PostSuccess:
    message_ids: list[int]
    count: int
    preview: str


@dataclass
class PostFailure:
    reason: str
    detail: str


async def _send_thread(bot: Bot, chat_id: str, posts: list[str]) -> PostSuccess:
    message_ids = []

    header = await bot.send_message(
        chat_id=chat_id,
        text="PersonalAI Research Thread",
    )
    message_ids.append(header.message_id)

    for post in posts:
        message = await bot.send_message(
            chat_id=chat_id,
            text=post,
        )
        message_ids.append(message.message_id)

    footer = await bot.send_message(
        chat_id=chat_id,
        text=f"Thread complete - {len(posts)} posts\nPowered by PersonalAI",
    )
    message_ids.append(footer.message_id)

    return PostSuccess(
        message_ids=message_ids,
        count=len(posts),
        preview=posts[0][:100] if posts else "",
    )


def post_thread(posts: list[str]) -> PostSuccess | PostFailure:
    if not posts:
        return PostFailure(reason="EMPTY", detail="No posts to publish.")

    if telegram_module is None:
        return PostFailure(
            reason="MISSING_DEPENDENCY",
            detail="python-telegram-bot is not installed.",
        )

    telegram = cast(Any, telegram_module)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token:
        return PostFailure(
            reason="CONFIG_ERROR",
            detail="TELEGRAM_BOT_TOKEN is not set.",
        )
    if not chat_id:
        return PostFailure(
            reason="CONFIG_ERROR",
            detail="TELEGRAM_CHAT_ID is not set.",
        )

    try:
        bot = telegram.Bot(token=token)
        return asyncio.run(_send_thread(bot, chat_id, posts))
    except telegram.error.InvalidToken:
        return PostFailure(
            reason="AUTH_FAILED",
            detail="Bot token is invalid. Check TELEGRAM_BOT_TOKEN.",
        )
    except telegram.error.ChatNotFound:
        return PostFailure(
            reason="CHAT_NOT_FOUND",
            detail="Chat ID not found. Message the bot first, then retry.",
        )
    except telegram.error.RetryAfter as exc:
        return PostFailure(
            reason="RATE_LIMITED",
            detail=f"Telegram asked to wait {exc.retry_after} seconds before retrying.",
        )
    except telegram.error.NetworkError as exc:
        return PostFailure(
            reason="NETWORK_ERROR",
            detail=f"Connection failed: {exc}",
        )
    except Exception as exc:
        return PostFailure(reason="UNKNOWN", detail=str(exc))


if __name__ == "__main__":
    test_posts = [
        "1/ Testing PersonalAI Telegram integration",
        "2/ This is the second message in the thread",
        "3/ If you see this, it works!",
    ]
    result = post_thread(test_posts)
    print(result)
