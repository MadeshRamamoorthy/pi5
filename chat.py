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
"""

from __future__ import annotations

import concurrent.futures
import os
import socket
from typing import Iterable

import requests

import config


SYSTEM_PROMPT = (
    "You are Echo, a friendly receptionist at a small engineering studio. "
    "Keep answers short -- two or three sentences -- and conversational. "
    "If the user asks about projects on display, suggest they look at the "
    "screen on the right of the kiosk."
)


class ChatBudgetError(RuntimeError):
    pass


class ChatBackendError(RuntimeError):
    pass


# ---------------- backends ----------------


class _Backend:
    label = "?"

    def available(self) -> bool:
        raise NotImplementedError

    def complete(self, question: str, system: str) -> str:
        raise NotImplementedError


class OpenAIBackend(_Backend):
    label = "OpenAI"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.label = f"OpenAI · {model}"
        self._client = None

    def available(self) -> bool:
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
        self.url = url.rstrip("/")
        self.model = model
        self.label = f"Ollama (local) · {model}"

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=1.5)
            r.raise_for_status()
            tags = {m.get("name", "").split(":")[0] for m in r.json().get("models", [])}
            tags.update(m.get("name", "") for m in r.json().get("models", []))
            base = self.model.split(":")[0]
            return base in tags or self.model in tags
        except Exception:
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
                f"Question budget reached ({self.max_questions} max)."
            )
        self._used[emp_id] = used + 1
        return self._used[emp_id]


class ChatClient:
    def __init__(self):
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
        return b.label if b else "Chat unavailable — no backend reachable"

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
        try:
            answer = backend.complete(question, SYSTEM_PROMPT)
        except Exception as exc:  # roll back budget on error
            self.budget._used[emp_id] = max(0, self.budget._used.get(emp_id, 1) - 1)
            raise ChatBackendError(f"{backend.label}: {exc}") from exc
        self._last_label = backend.label
        return answer

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
