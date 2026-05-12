"""LLM chat with OpenAI -> Ollama fallback.

Two backends. `ChatClient.ask(emp_id, question)` enforces a per-session
question budget and dispatches to whichever backend in
config.CHAT_BACKEND_ORDER is currently reachable.

OpenAI: requires `OPENAI_API_KEY` env var and internet.
Ollama:  reaches a local server (default http://localhost:11434)
         configured per https://www.raspberrypi.com/documentation/computers/ai.html.
         Used as a backup so the kiosk can answer questions offline.

Calls run on a dedicated worker thread so the camera loop never blocks.
The caller submits a question via `submit(emp_id, question)` which
returns a `concurrent.futures.Future`; poll it from the main loop and
read `result()` once done.

`available()` results are cached for AVAILABILITY_TTL_SEC so the camera
loop's per-frame `status()` calls don't hammer the network.
"""

from __future__ import annotations

import concurrent.futures
import os
import socket
import time
from typing import Iterable

import requests

import config


BASE_SYSTEM_PROMPT = (
    "You are ECHO SCOPE, a friendly AI kiosk at the Infosys Calgary AI "
    "Club. Keep answers short -- two or three sentences -- and "
    "conversational. Sound like a person, not a brochure."
)


def build_system_prompt(db=None) -> str:
    """Return the system prompt with the current projects + next session
    folded in. When the user asks "what's on display today?" or "what's
    the next session?" the LLM can answer from this context instead of
    hallucinating. Other questions go through normally."""
    if db is None:
        return BASE_SYSTEM_PROMPT
    try:
        projects = db.list_projects() or []
        session_row = db.next_session()
    except Exception:
        return BASE_SYSTEM_PROMPT

    parts = [BASE_SYSTEM_PROMPT]
    if projects:
        lines = []
        for r in projects:
            title = (r[1] or "").strip()
            desc = (r[2] or "").strip()
            if not title:
                continue
            lines.append(f"  - {title}" + (f": {desc}" if desc else ""))
        if lines:
            parts.append(
                "Projects on display today (read these out if the user "
                "asks 'what's on display' / 'projects' / 'what can I see "
                "today'):\n" + "\n".join(lines)
            )
    if session_row:
        sid, title, starts_at, ends_at, notes = session_row
        when = (starts_at or "").replace("T", " ")
        if ends_at:
            when += f" - {ends_at}"
        ses = (
            f"Next upcoming session (read this out if the user asks "
            f"'what's the next session' / 'what's coming up' / 'when is "
            f"the next event'):\n  Title: {title}\n  When: {when}"
        )
        if notes:
            ses += f"\n  Notes: {notes}"
        parts.append(ses)
    parts.append(
        "Only use the lists above for questions about projects or "
        "sessions. For any other question, answer normally and "
        "conversationally."
    )
    return "\n\n".join(parts)


# Backwards compat for any external importer.
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

AVAILABILITY_TTL_SEC = 5.0


class ChatBudgetError(RuntimeError):
    pass


class ChatBackendError(RuntimeError):
    pass


# ---------------- backends ----------------


class _Backend:
    label = "?"

    def __init__(self):
        self._cache_ok: bool | None = None
        self._cache_at: float = 0.0

    def available(self) -> bool:
        now = time.time()
        if self._cache_ok is not None and now - self._cache_at < AVAILABILITY_TTL_SEC:
            return self._cache_ok
        ok = self._probe()
        self._cache_ok = ok
        self._cache_at = now
        return ok

    def invalidate(self) -> None:
        self._cache_ok = None

    def _probe(self) -> bool:
        raise NotImplementedError

    def complete(self, question: str, system: str) -> str:
        raise NotImplementedError


class OpenAIBackend(_Backend):
    label = "OpenAI"

    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.label = f"OpenAI · {model}"
        self._client = None

    def _probe(self) -> bool:
        if not self.api_key:
            return False
        try:
            with socket.create_connection(("api.openai.com", 443), timeout=1.5):
                return True
        except OSError:
            return False

    def complete(self, question: str, system: str) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()


class OllamaBackend(_Backend):
    label = "Ollama (local)"

    def __init__(self, url: str, model: str):
        super().__init__()
        self.url = url.rstrip("/")
        self.model = model
        self.label = f"Ollama (local) · {model}"

    def _probe(self) -> bool:
        # Cheap reachability probe via socket (no HTTP). Avoids parsing
        # /api/tags JSON, which Hailo-Ollama can return with a slightly
        # different schema. The actual model error (if any) surfaces
        # when complete() runs.
        try:
            from urllib.parse import urlparse
            u = urlparse(self.url)
            host = u.hostname or "localhost"
            port = u.port or (443 if u.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            return False

    def complete(self, question: str, system: str) -> str:
        r = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": question,
                "system": system,
                "stream": False,
                "options": {"num_predict": 200},
            },
            timeout=60,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()


# ---------------- client + budget ----------------


class ChatBudget:
    """Per-emp_id question count. Reset on session start."""

    def __init__(self, max_questions: int):
        self.max_questions = max_questions
        self._used: dict[str, int] = {}

    def reset(self) -> None:
        self._used.clear()

    def remaining(self, emp_id: str) -> int:
        return max(0, self.max_questions - self._used.get(emp_id, 0))

    def consume(self, emp_id: str) -> int:
        used = self._used.get(emp_id, 0)
        if used >= self.max_questions:
            raise ChatBudgetError(
                f"Lovely chatting! That's {self.max_questions} questions for "
                "now — come say hi again anytime."
            )
        self._used[emp_id] = used + 1
        return self._used[emp_id]


class ChatClient:
    def __init__(self, db=None):
        # When `db` is set, every chat call builds a system prompt that
        # includes the current projects + next session row -- so the
        # LLM can answer those questions from local truth instead of
        # hallucinating. Other questions go through normally.
        self.db = db
        self.budget = ChatBudget(config.CHAT_MAX_QUESTIONS_PER_SESSION)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="chat",
        )
        self._backends: dict[str, _Backend] = {
            "openai": OpenAIBackend(
                os.environ.get("OPENAI_API_KEY", ""),
                config.OPENAI_MODEL,
            ),
            "ollama": OllamaBackend(config.OLLAMA_URL, config.OLLAMA_MODEL),
        }
        self._last_label: str | None = None

    @property
    def last_backend_label(self) -> str | None:
        return self._last_label

    def pick_backend(self) -> _Backend | None:
        for name in config.CHAT_BACKEND_ORDER:
            b = self._backends.get(name)
            if b is not None and b.available():
                return b
        return None

    def status(self) -> str:
        b = self.pick_backend()
        return (b.label if b
                else "Chat is taking a quick break — back online shortly.")

    def submit(self, emp_id: str, question: str
               ) -> "concurrent.futures.Future[str]":
        return self._executor.submit(self._ask_blocking, emp_id, question)

    # internal -----------------------------------------------------------

    def _ask_blocking(self, emp_id: str, question: str) -> str:
        backend = self.pick_backend()
        if backend is None:
            raise ChatBackendError("No chat backend reachable.")
        # Reserve a slot before calling the backend so we don't get charged
        # for a network round-trip the user can't actually use.
        self.budget.consume(emp_id)
        system = build_system_prompt(self.db)
        try:
            answer = backend.complete(question, system)
        except Exception as exc:  # roll back budget on error
            self.budget._used[emp_id] = max(0, self.budget._used.get(emp_id, 1) - 1)
            raise ChatBackendError(f"{backend.label}: {exc}") from exc
        self._last_label = backend.label
        return answer

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
