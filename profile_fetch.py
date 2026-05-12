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


_FETCH_TIMEOUT = 8
_MAX_PROFILE_CHARS = 2500


def fetch_profile_text(url: str) -> str:
    """Fetch `url` and return roughly the first 2.5 KB of visible text.

    Strips `<script>` and `<style>` blocks, then collapses tags +
    whitespace. Good enough for the kind of corporate bio pages this
    feature targets -- HTML JS-heavy SPAs that ship no server-rendered
    content will return an empty string.
    """
    r = requests.get(
        url,
        timeout=_FETCH_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (ECHO SCOPE kiosk)"},
    )
    r.raise_for_status()
    html = r.text
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_PROFILE_CHARS]


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


def regenerate(name: str, profile_url: str) -> tuple[str, str]:
    """Convenience wrapper: fetch + summarise. Returns
    (welcome_text, error_or_status_message). On success error is "" and
    welcome_text is non-empty."""
    if not profile_url:
        return ("", "no profile_url set")
    try:
        body = fetch_profile_text(profile_url)
    except Exception as exc:  # noqa: BLE001
        return ("", f"fetch failed: {exc}")
    if not body:
        return ("", "page returned no readable text")
    welcome = summarise_for_welcome(name, body)
    if not welcome:
        return ("", "LLM returned empty (no OPENAI_API_KEY?)")
    return (welcome, "")
