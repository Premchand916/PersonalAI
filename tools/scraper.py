import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup


@dataclass
class ScrapeSuccess:
    url: str
    content: str
    chars: int


@dataclass
class ScrapeFailure:
    url: str
    reason: str
    detail: str
    retried: bool = False


ScrapeResult = ScrapeSuccess | ScrapeFailure


def scrape_url(url: str, max_chars: int = 3000, timeout: int = 10) -> ScrapeResult:
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
            follow_redirects=True,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        text = text[:max_chars]

        return ScrapeSuccess(url=url, content=text, chars=len(text))
    except httpx.TimeoutException:
        return ScrapeFailure(
            url=url,
            reason="TIMEOUT",
            detail=f"No response within {timeout}s",
        )
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 403:
            reason, detail = "BLOCKED", "Site returned 403 Forbidden and blocks scrapers."
        elif code == 404:
            reason, detail = "NOT_FOUND", "Page no longer exists (404)."
        elif code == 429:
            reason, detail = "RATE_LIMITED", "Too many requests (429)."
        else:
            reason, detail = "HTTP_ERROR", f"HTTP {code}: {exc}"

        return ScrapeFailure(url=url, reason=reason, detail=detail)
    except httpx.ConnectError:
        return ScrapeFailure(
            url=url,
            reason="CONNECT_ERROR",
            detail="Could not reach the server due to DNS or network issues.",
        )
    except Exception as exc:
        return ScrapeFailure(
            url=url,
            reason="UNKNOWN",
            detail=str(exc),
        )


def scrape_with_retry(url: str, max_chars: int = 3000) -> ScrapeResult:
    result = scrape_url(url, max_chars=max_chars, timeout=10)

    if isinstance(result, ScrapeFailure) and result.reason == "TIMEOUT":
        print(f"[Scraper] TIMEOUT on {url} - retrying with 20s timeout...")
        retry_result = scrape_url(url, max_chars=max_chars, timeout=20)
        if isinstance(retry_result, ScrapeFailure):
            retry_result.retried = True
        return retry_result

    return result
