"""Fetch a public bio page and turn it into a one-sentence kiosk welcome.

Used by the admin path when an employee's profile_url changes -- the
result is cached in employees.welcome_cache so the camera worker can
read it without doing any network calls during recognition.

The summarisation uses OpenAI's chat completions. If no API key is
configured the function returns "" and the caller falls back to the
random `messages.RECOGNIZED_GREETINGS` lines.
"""

from __future__ import annotations

import os
import re

import requests


_FETCH_TIMEOUT = 10
_MAX_PROFILE_CHARS = 2500

# Big corporate sites (infosys.com, microsoft.com, ...) refuse traffic
# from anything that doesn't look like a real browser, so we send a
# full Chrome-on-Mac UA + the headers a typical browser would.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


def fetch_profile_text(url: str) -> str:
    """Fetch `url` and return roughly the first 2.5 KB of visible text.

    Strips `<script>` and `<style>` blocks, then collapses tags +
    whitespace. Good enough for the kind of corporate bio pages this
    feature targets -- HTML JS-heavy SPAs that ship no server-rendered
    content will return an empty string.
    """
    r = requests.get(url, timeout=_FETCH_TIMEOUT, headers=_BROWSER_HEADERS)
    r.raise_for_status()
    html = r.text
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_PROFILE_CHARS]


def summarise_via_web_search(name: str, url: str) -> str:
    """Bypass scraper-blockers (Akamai etc.) by handing the URL to
    OpenAI's web_search tool. OpenAI's own crawler isn't blocked by
    the same fingerprinting rules that 403 `requests`, so this is the
    reliable path for sites like infosys.com / microsoft.com.

    Returns "" if OpenAI isn't configured, the Responses API isn't
    available on the installed openai SDK, or the LLM returned no
    text. Caller falls through to the manual paste-bio path.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        print(f"[profile] OpenAI client init failed: {exc!r}")
        return ""

    prompt = (
        f"Open this page: {url}\n\n"
        f"From the page content about {name}, write a single warm "
        f"welcome sentence (under 22 words) addressing them by name "
        f"and referencing ONE specific thing -- their current role, "
        f"a project they lead, or a known interest. Don't start with "
        f"'Hello' or 'Welcome' if you can avoid it. Sound natural."
    )

    # Try the Responses API with the web_search built-in tool first.
    # Falls through to chat.completions on older SDKs / when the tool
    # isn't enabled on the account.
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=prompt,
        )
        # The Responses API exposes the answer as .output_text on
        # recent SDKs.
        text = (getattr(resp, "output_text", "") or "").strip()
        if text:
            return re.sub(r"\s+", " ", text.strip('"').strip())
    except Exception as exc:  # noqa: BLE001
        print(f"[profile] responses.create(web_search) unavailable: {exc!r}")

    # No fallback that can fetch URLs reliably -- caller will see ""
    # and surface the manual-paste hint.
    return ""


def summarise_for_welcome(name: str, profile_text: str) -> str:
    """Ask the LLM for a single warm welcome sentence.

    Returns "" when no LLM is configured or the call fails. Callers
    should treat empty as "use the standard random greeting".
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return ""
    if not profile_text:
        return ""
    try:
        # Import lazily so the module loads even without openai installed.
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=80,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly office kiosk welcoming "
                        "returning visitors. Speak naturally, like a "
                        "person, not a brochure."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Write a single warm welcome sentence (under "
                        f"22 words) for {name}. Address them by name, "
                        f"and reference ONE specific thing from their "
                        f"bio below -- their current role, a project, "
                        f"or a known interest. Don't list multiple "
                        f"facts. Don't start with 'Hello' or 'Welcome' "
                        f"if you can avoid it.\n\n"
                        f"Bio (raw, may contain HTML cruft):\n"
                        f"{profile_text}"
                    ),
                },
            ],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[profile] LLM summarise failed: {exc!r}")
        return ""

    text = (resp.choices[0].message.content or "").strip().strip('"').strip()
    # Strip any trailing emoji + clean tabbed whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def regenerate(name: str, profile_input: str) -> tuple[str, str]:
    """Convenience wrapper. `profile_input` can be either:

    - an http(s):// URL -- fetched + stripped + summarised
    - any other non-empty string -- treated as already-pasted bio
      text (useful when the target site blocks scrapers, e.g.
      infosys.com returns 403)

    Returns (welcome_text, error_or_status_message). On success error
    is "" and welcome_text is non-empty.
    """
    text = (profile_input or "").strip()
    if not text:
        return ("", "no profile URL or text set")
    if text.lower().startswith(("http://", "https://")):
        # Path A: direct fetch. Works for sites that aren't behind
        # bot-management.
        body = ""
        direct_err = ""
        try:
            body = fetch_profile_text(text)
        except Exception as exc:  # noqa: BLE001
            direct_err = str(exc)
        if body:
            welcome = summarise_for_welcome(name, body)
            if welcome:
                return (welcome, "")

        # Path B: hand the URL to OpenAI's web_search tool. Their
        # crawler isn't blocked by the same Akamai / Cloudflare
        # fingerprinting that 403s a Python requests call -- so
        # infosys.com / microsoft.com / similar pages go through.
        web = summarise_via_web_search(name, text)
        if web:
            return (web, "")

        # Both paths gave up -- surface the original direct-fetch
        # error if there was one, plus the paste hint.
        if direct_err:
            return ("",
                    f"direct fetch failed ({direct_err}) and OpenAI "
                    "web_search returned nothing. The site likely has "
                    "bot protection (Akamai/Cloudflare). Paste the bio "
                    "text directly into the field instead of a URL.")
        return ("",
                "page returned no readable text and OpenAI web_search "
                "returned nothing. Try pasting the bio text directly "
                "into the field instead.")

    # Path C: input isn't a URL -- treat as pre-pasted bio text.
    welcome = summarise_for_welcome(name, text)
    if not welcome:
        return ("", "LLM returned empty (no OPENAI_API_KEY?)")
    return (welcome, "")
