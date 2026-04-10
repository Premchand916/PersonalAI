# tools/secrets_guard.py
#
# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceOS — Security Foundation
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY THIS FILE EXISTS
# ─────────────────────
# PersonalAI v1 relied on each tool checking os.getenv() individually.
# If TAVILY_API_KEY was missing, you got a cryptic error deep in web_search.py.
# You had no single place to audit what secrets the system needed.
#
# IntelligenceOS runs AUTOMATICALLY at 9am with no human watching.
# If a secret is missing, you need to know at START-UP — not at 7:03am
# when the briefing fails silently and you wake up to nothing.
#
# This file is the security perimeter:
#   validate_env()  → called at every entry point  (hard stop if broken)
#   safe_log()      → replaces print() everywhere   (never leaks secrets)
#   redact()        → scrubs sensitive patterns      (before logging/storing)
#
# RULE: Every file that starts work must call validate_env() first.
#       No exceptions.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import sys

# ── Load .env before anything else ───────────────────────────────────────────
# python-dotenv reads .env file and populates os.environ
# safe=True means it won't crash if dotenv isn't installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Rely on system env vars if dotenv not installed


# ── Required environment variables ───────────────────────────────────────────
# Format: { "VAR_NAME": "human description of what it's used for" }
# Add here when IntelligenceOS needs a new secret.
REQUIRED_VARS: dict[str, str] = {
    "TAVILY_API_KEY":      "Web search — WatchlistAgent uses this to monitor topics",
    "OLLAMA_MODEL":        "Local LLM — all agents use this for summarisation",
    "TELEGRAM_BOT_TOKEN":  "Telegram delivery — AlertAgent + BriefingAgent",
    "TELEGRAM_CHAT_ID":    "Telegram destination — where to send briefs and alerts",
}

# ── Placeholder patterns that mean the var was NEVER filled in ───────────────
# If someone copies .env.example without editing it, we catch them here.
_PLACEHOLDER_PATTERNS: list[str] = [
    "your_",
    "YOUR_",
    "changeme",
    "CHANGEME",
    "example",
    "EXAMPLE",
    "<replace>",
    "xxx",
    "XXX",
]

# ── Regex patterns to scrub from logs and storage ────────────────────────────
# These match the SHAPE of real secrets, not the exact values.
_REDACT_PATTERNS: list[str] = [
    r"tvly-[A-Za-z0-9\-_]{20,}",       # Tavily API key  (tvly-xxxx)
    r"\d{8,10}:[A-Za-z0-9_\-]{30,}",   # Telegram bot token  (123456789:ABCxxx)
    r"sk-[A-Za-z0-9]{20,}",            # OpenAI-style keys   (sk-xxx)
    r"key-[A-Za-z0-9]{20,}",           # Generic key prefix
]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def validate_env(entry_point: str = "unknown") -> None:
    """
    Call this at the TOP of every entry point before any work begins:
        main.py, api/server.py, schedules/morning_brief.py, ui/app.py

    What it checks:
        1. Every REQUIRED_VAR is present in the environment
        2. None of them still hold placeholder text (e.g. "your_tavily_api_key")

    On failure → prints clear diagnosis and sys.exit(1)
    On success → logs one confirmation line and returns normally

    WHY sys.exit instead of raise?
        Because a raised exception can be caught accidentally by calling code.
        A missing API key is never recoverable at runtime — fail loudly, fail fast.

    Example usage:
        # top of main.py
        from tools.secrets_guard import validate_env
        validate_env("main.py")
    """
    missing: list[str] = []
    placeholder: list[str] = []

    for var, purpose in REQUIRED_VARS.items():
        value = os.getenv(var, "")

        if not value:
            missing.append(f"  - {var}  [{purpose}]")
            continue

        # Check for un-edited placeholder values
        if any(pattern in value for pattern in _PLACEHOLDER_PATTERNS):
            placeholder.append(
                f"  - {var}  still contains placeholder text: "
                f"'{value[:40]}...'"
            )

    if missing or placeholder:
        _print_utf8("\n" + "=" * 62)
        _print_utf8(f"  [SecretsGuard] VALIDATION FAILED — entry: {entry_point}")
        _print_utf8("=" * 62)

        if missing:
            _print_utf8("\n  MISSING variables (not set anywhere):")
            for m in missing:
                _print_utf8(m)

        if placeholder:
            _print_utf8("\n  PLACEHOLDER values (never filled in):")
            for p in placeholder:
                _print_utf8(p)

        _print_utf8(
            "\n  Fix:"
            "\n    1. Copy  .env.example → .env"
            "\n    2. Fill in real values for each variable above"
            "\n    3. Never commit your .env file (it's in .gitignore)"
            "\n"
        )
        _print_utf8("=" * 62 + "\n")
        sys.exit(1)

    safe_log(
        f"[SecretsGuard] OK — {len(REQUIRED_VARS)} vars validated "
        f"at entry point '{entry_point}'"
    )


def redact(text: str) -> str:
    """
    Scrub known secret patterns from a string before logging or storing.

    WHY: API keys sometimes end up in error messages, stack traces, or
    state fields. This function replaces them with [REDACTED] so they
    never appear in logs, SQLite rows, or Telegram messages.

    Example:
        redact("Tavily error with key tvly-abc123xyz456789012")
        → "Tavily error with key [REDACTED]"

    Called automatically by safe_log() — you rarely need to call it directly.
    Call it explicitly before writing to SQLite:
        cursor.execute("INSERT ... VALUES (?)", (redact(summary),))
    """
    if not isinstance(text, str):
        return str(text)

    result = text
    for pattern in _REDACT_PATTERNS:
        result = re.sub(pattern, "[REDACTED]", result)
    return result


def safe_log(message: str, level: str = "INFO") -> None:
    """
    Replacement for print() throughout IntelligenceOS.

    WHY not just print()?
        - print() can leak secrets if a key ends up in an error message
        - print() crashes on Windows if the message has emoji and the
          terminal is set to a narrow encoding

    What safe_log() does:
        1. Calls redact() on the message        → no secrets in output
        2. Adds a level prefix for [WARN]/[ERROR]
        3. Falls back to ASCII if UTF-8 fails   → no Windows crashes

    Levels: "INFO" (default), "WARN", "ERROR"

    Examples:
        safe_log("WatchlistAgent: 3 delta events found")
        safe_log("Tavily rate limited — skipping topic", level="WARN")
        safe_log("Ollama unreachable", level="ERROR")
    """
    clean = redact(str(message))

    if level == "INFO":
        line = clean
    else:
        line = f"[{level}] {clean}"

    _print_utf8(line)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_utf8(text: str) -> None:
    """
    Windows-safe print.
    On narrow terminals (cp1252, cp850) emoji causes UnicodeEncodeError.
    This falls back to ASCII replacement characters instead of crashing.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (python tools/secrets_guard.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test redact
    sample = "API call failed: tvly-abc123xyz456789012345 returned 429"
    redacted = redact(sample)
    print(f"Original : {sample}")
    print(f"Redacted : {redacted}")
    assert "[REDACTED]" in redacted, "redact() failed to catch Tavily key"
    print("redact() ... OK")

    # Test safe_log
    safe_log("System started — 4 agents loaded")
    safe_log("Watchlist empty, skipping delta check", level="WARN")
    safe_log("Ollama connection refused on port 11434", level="ERROR")
    print("safe_log() ... OK")

    # Test validate_env
    # (Will exit if .env is missing — that's expected behaviour)
    print("\nRunning validate_env() — ensure .env is populated:")
    validate_env("secrets_guard self-test")
    print("validate_env() ... OK")
