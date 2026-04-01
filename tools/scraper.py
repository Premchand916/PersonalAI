# tools/scraper.py

import re
import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass

# ── Structured result types ────────────────────────────────────────
@dataclass
class ScrapeSuccess:
    url:     str
    content: str
    chars:   int

@dataclass  
class ScrapeFailure:
    url:       str
    reason:    str          # TIMEOUT / BLOCKED / NOT_FOUND / UNKNOWN
    detail:    str          # exact error message
    retried:   bool = False # did we attempt a retry?

# Type alias — a scrape returns one or the other
ScrapeResult = ScrapeSuccess | ScrapeFailure


def scrape_url(url: str, max_chars: int = 3000, timeout: int = 10) -> ScrapeResult:
    """
    Fetches a URL and returns either ScrapeSuccess or ScrapeFailure.
    Never raises — always returns a typed result.
    Caller decides what to do with failures.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = httpx.get(
            url,
            headers=headers,
            timeout=timeout,
            follow_redirects=True
        )
        response.raise_for_status()

        # Parse and clean HTML
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text[:max_chars]

        return ScrapeSuccess(url=url, content=text, chars=len(text))

    except httpx.TimeoutException:
        return ScrapeFailure(
            url=url,
            reason="TIMEOUT",
            detail=f"No response within {timeout}s"
        )

    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 403:
            reason, detail = "BLOCKED", "Site returned 403 Forbidden — blocks scrapers"
        elif code == 404:
            reason, detail = "NOT_FOUND", "Page no longer exists (404)"
        elif code == 429:
            reason, detail = "RATE_LIMITED", "Too many requests (429) — slow down"
        else:
            reason, detail = "HTTP_ERROR", f"HTTP {code}: {str(e)}"

        return ScrapeFailure(url=url, reason=reason, detail=detail)

    except httpx.ConnectError:
        return ScrapeFailure(
            url=url,
            reason="CONNECT_ERROR",
            detail="Could not reach the server — DNS or network issue"
        )

    except Exception as e:
        return ScrapeFailure(
            url=url,
            reason="UNKNOWN",
            detail=str(e)
        )


def scrape_with_retry(url: str, max_chars: int = 3000) -> ScrapeResult:
    """
    Tries scraping once. If TIMEOUT, retries with 2x timeout.
    All other failures are returned immediately — no retry needed.
    
    Why only retry TIMEOUT?
    - BLOCKED: retrying won't change a 403
    - NOT_FOUND: page is gone, retrying wastes time
    - TIMEOUT: server was slow, more time might work
    """
    result = scrape_url(url, max_chars=max_chars, timeout=10)

    # Only retry timeouts
    if isinstance(result, ScrapeFailure) and result.reason == "TIMEOUT":
        print(f"[Scraper] TIMEOUT on {url} — retrying with 20s timeout...")
        result = scrape_url(url, max_chars=max_chars, timeout=20)
        result.retried = True   # mark that we already tried twice

    return result