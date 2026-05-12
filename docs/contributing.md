# Contributing

## Workflow

- **Branches**: feature work happens on `claude/<topic>` branches
  (`claude/asr-lightweight`, `claude/echo-scope-ui`, etc.). Open a
  PR against the default branch when ready.
- **Hooks**: there are none — `git commit` runs straight through.
- **Tests**: there's no automated test suite yet. The validation
  surface is the kiosk itself + watching the backend log.
- **Style**: 4-space indent, type hints where useful, no docstrings
  on obvious functions. Comments explain "why" only — file paths,
  hidden constraints, subtle invariants.

## Local dev loop

```bash
git checkout -b claude/<topic>
# edit code
./stop.sh && ./start.sh
tail -F /tmp/echo-backend.log
# test in Chromium
git add -A
git commit -m "..."
git push -u origin claude/<topic>
```

For SPA-only changes (HTML/CSS/JS), no restart needed — **Ctrl+F5**
in Chromium hard-reloads.

## Commit messages

Follow the existing format: short summary, blank line, optional body
explaining the *why*. End with the session URL line for traceability.

```
fix-area: short imperative summary

Body explaining the why. What was broken, what changed, anything
non-obvious about the fix. Bullet points are fine for multi-issue
commits.

https://claude.ai/code/session_XXXX
```

## File touch rules

- **State changes**: must go through `state.StateBus.update()` so
  SSE subscribers see them. Direct mutation of `state._state` bypasses
  the diff broadcast.
- **Camera worker**: serialised by a single thread. New async work
  goes through a queue (`register_q`, `photo_register_q`, `chat_q`)
  rather than calling worker methods directly.
- **Database**: SQLite connection is opened with
  `check_same_thread=False` so any thread can use it. SQLite
  internally serialises writes; reads are fine concurrent.
- **TTS**: never call backend `speak()` directly — use `AsyncTTS`
  with an optional `on_start` callback for UI sync.
- **MJPEG frames**: `FrameStreamer.push()` is throttled to
  `MJPEG_MAX_FPS`. Don't push more than necessary.

## Adding a new ASR / TTS / Chat backend

Three subsystems, same pattern:

1. Subclass the abstract base (`ChatASR`, `_Backend`, `ChatClient`).
2. Add the new backend to the factory function (`make_chat_asr()`,
   `make_backend()`, `chat.create_client()`).
3. Add a new value to the corresponding `config.py` enum (e.g.
   `CHAT_ASR_BACKEND = "yourthing"`).
4. Document it in [speech](speech.md) and add a tuning row.

The `auto` selector in each factory tries backends in priority
order with explicit file-or-env checks. Don't add a new backend to
`auto` unless it's strictly better than the current next-best for
the same hardware.

## Adding an admin endpoint

In `web/app.py`:

```python
@app.route("/api/yourthing", methods=["POST"])
def api_yourthing():
    if not _check_admin_auth():
        return _request_admin_auth()
    body = request.get_json(silent=True) or {}
    # ... validate, enqueue, return 204
    return ("", 204)
```

If the endpoint needs work done by the camera worker, put it on a
queue rather than touching `hailo_infer` / camera state directly.

## Adding a UI element

`web/templates/index.html` is one big SPA shell with `data-bind=...`
attributes the JS controller mirrors state into. To add a new
field:

1. Add the field to `state.KioskState`.
2. Push to it from the camera worker via `state.update(...)`.
3. Add the HTML element with `data-bind="my-field"`.
4. (Optional) Add a renderer in `web/static/echo.js`'s `applyState`
   if the field needs formatting beyond plain text.

For new styling, add it to `web/static/echo.css`. The design
language is glassmorphism + cyan glow — keep new components
consistent.

## Releasing

There are no releases or tags today. Each merged PR to the default
branch is the latest "release". Pin a commit in your `start.sh`
deployment if you need stability.

## Reporting bugs

See [troubleshooting](troubleshooting.md) for what to include when
filing an issue.
