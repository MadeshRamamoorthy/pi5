"""Match a free-text question against the solutions catalog.

Pure, local, deterministic keyword scoring -- no LLM, no embeddings. The
kiosk only has ~16 short rows to search, so a simple token-overlap score
(weighting name matches above description matches) is plenty to answer
"do you have something for container security?" with the right solution.

Used by main.py's chat path: when the user asks about a solution, we
score every catalog row and return the best match(es) above a threshold.
"""

from __future__ import annotations

import re

# Common words that carry no matching signal -- dropped from both the
# query and the catalog text so they don't create spurious overlaps.
_STOP = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with",
    "is", "are", "do", "you", "we", "have", "has", "any", "some", "that",
    "this", "it", "its", "your", "our", "can", "could", "would", "there",
    "about", "me", "tell", "show", "give", "want", "need", "looking", "got",
    "what", "which", "who", "how", "does", "did", "build", "built", "make",
    "made", "develop", "developed", "solution", "solutions", "tool", "tools",
    "platform", "platforms", "app", "apps", "application", "product",
    "products", "anything", "something", "please",
}

_MIN_SCORE = 2   # best match must clear this to count as a confident hit


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in _STOP
    }


def match_solutions(query: str, rows, limit: int = 3):
    """Score catalog `rows` against `query`.

    rows: iterable of (id, name, description, domain, link, ordering).
    Returns [(score, row), ...] sorted by score desc, only score > 0,
    capped at `limit`. Name-token hits are weighted 3x body hits.
    """
    qtokens = _tokens(query)
    if not qtokens:
        return []
    scored = []
    for row in rows:
        name, desc, domain = row[1], row[2], row[3]
        name_tokens = _tokens(name)
        body_tokens = _tokens(f"{desc} {domain}")
        score = 0
        for t in qtokens:
            if t in name_tokens:
                score += 3
            elif t in body_tokens:
                score += 1
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored[:limit]


def is_confident(matches) -> bool:
    return bool(matches) and matches[0][0] >= _MIN_SCORE


def summarize(description: str, max_chars: int = 220) -> str:
    """First sentence of `description`, trimmed to a spoken-friendly
    length at a word boundary."""
    desc = (description or "").strip()
    if not desc:
        return ""
    first = re.split(r"(?<=[.!?])\s+", desc, maxsplit=1)[0].strip()
    if len(first) > max_chars:
        first = first[:max_chars].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
    return first
