# tools/scraper.py

import httpx
from bs4 import BeautifulSoup

def scrape_url(url: str, max_chars: int = 3000) -> str:
    """
    Fetches a URL and extracts clean readable text.
    Strips HTML tags, scripts, navigation — just the content.
    
    max_chars: We limit to 3000 chars per page.
    Why? Ollama has a context window limit.
    Sending 50,000 chars of HTML would overflow it.
    3000 chars = enough to understand the article.
    """
    try:
        # Pretend to be a browser — some sites block Python requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        
        # timeout=10: don't wait forever if site is slow
        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()     # raises error if 404, 500, etc.
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove junk tags — we don't want script/style/nav text
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()     # remove from tree entirely
        
        # Extract clean text
        text = soup.get_text(separator=" ", strip=True)
        
        # Collapse multiple spaces/newlines into single space
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Return only first max_chars
        return text[:max_chars]
    
    except httpx.TimeoutException:
        return f"[SCRAPE FAILED] Timeout: {url}"
    
    except Exception as e:
        return f"[SCRAPE FAILED] {url} — {str(e)}"


# Quick test
if __name__ == "__main__":
    text = scrape_url("https://www.ideas2it.com/blogs/ai-agent-frameworks")
    print(f"Extracted {len(text)} chars")
    print(text[:500])