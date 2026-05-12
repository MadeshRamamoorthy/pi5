# State machine

The kiosk has two top-level states (IDLE / ACTIVE) plus three
sub-states the SPA reads to swap views and toggle overlays.

```
                           wake word
                           tap on dashboard
                       ┌───────────────────┐
                       │                   │
                       ▼                   │
                                           │
   ┌────────────┐                 ┌────────┴────────┐
   │   IDLE     │                 │     ACTIVE      │
   │            │                 │                 │
   │ dashboard  │ ◀───────────────┤ camera + boxes  │
   │ shown      │                 │                 │
   └────────────┘                 │ ┌─────────────┐ │
                                  │ │ chat busy   │ │
                                  │ │ - listening │ │
                                  │ │ - transcr.  │ │
                                  │ │ - pending   │ │
                                  │ └─────────────┘ │
                                  │                 │
                                  │ ┌─────────────┐ │
                                  │ │ register    │ │
                                  │ │ overlay open│ │
                                  │ └─────────────┘ │
                                  └─────────────────┘
                  ▲       ✕ tapped                  │
                  └─── idle 30s ─────────────────────┘
                  ▲                                 │
                  └─── active session 600s max ─────┘
```

## Transitions

| From | To | Trigger |
|---|---|---|
| IDLE | ACTIVE | Wake word fires (`hello echo scope` or aliases) |
| IDLE | ACTIVE | User taps anywhere on the dashboard SPA |
| ACTIVE | IDLE | User taps the ✕ button on the active header |
| ACTIVE | IDLE | `idle_for >= IDLE_AFTER_LAST_INTERACTION_SEC` (30 s) |
| ACTIVE | IDLE | `active_for >= ACTIVE_SESSION_MAX_SEC` (600 s hard cap) |
| ACTIVE | IDLE | "No thanks" tap on the register overlay |

## Engagement (resets the idle timer)

The idle countdown only fires while the user is *not* engaged. Any of
these conditions counts as engaged:

- `register_open` — registration form / pose capture is in progress
- `listening` — chat-voice is recording
- `chat_pending` — waiting for an LLM reply
- `chat_history` non-empty AND last chat message was within
  `CHAT_KEEPALIVE_SEC` (180 s)

The last condition is the "hold the screen while the user is having
a conversation" rule. As long as the user keeps asking questions
within 3 minutes of each other, the kiosk stays ACTIVE.

## Chat-busy detection skip

While `chat_pending` OR `listening` is true, the camera worker:

- skips `self.pipe.detect(...)` entirely (no NPU cycles)
- skips the liveness + recognition loop
- skips the bounding-box overlay draw
- resets `unknown_streak` so the register overlay can't pop
  mid-conversation

Frames still push to MJPEG so the SPA's camera image keeps moving
(just without boxes / labels until the chat exchange completes).
Recognition resumes on the very next loop iteration.

This means face recognition and Whisper inference never overlap on
the Hailo NPU — no chip contention even when both run on Hailo.

## go_active() and go_idle()

`state.StateBus.go_active()`:

- Sets `state="IDLE" -> "ACTIVE"` and updates `state_since`.
- The camera worker also: resets `liveness`, resets `greeter._last_greeted`,
  resets `chat.budget`, flushes pending TTS, pauses the wake-word
  listener, sets `chat_history=[]`, `chat_remaining=5`,
  `chat_pending=False`.
- Speaks "Hello! Lovely to see you." and prints
  `[state] IDLE -> ACTIVE (session XXXX)`.

`state.StateBus.go_idle()`:

- Sets `state="ACTIVE" -> "IDLE"` and clears:
  `listening`, `transcribing`, `register_open`, `chat_pending`,
  `chat_history=[]`, `person=None`, `toast=None`.
- The camera worker also: resets `liveness`, resets greeter
  cooldowns, flushes pending TTS, resumes the wake-word listener.
- Prints `[state] ACTIVE -> IDLE (<reason>, interactions=N)`. Reason
  is one of `"user closed"`, `"idle Ns"`.

The SPA's `applyState` handler hides `#active` and shows `#idle`
based on the `state` field — same code path regardless of which
trigger fired.

## Once-per-session greeting

The `seen_in_session` set (per-session, cleared on `go_active`)
prevents re-greeting. The greeter is skipped entirely for anyone in
the set — even if they walk out of frame for 90 seconds and come
back.

```python
for emp_id, name, _det, _score in pending_greets:
    if emp_id in seen_in_session:
        continue                  # already said hi this session
    if self.greeter.greet(emp_id, name):
        seen_in_session.add(emp_id)
        # ... record interaction, push metrics, etc.
```

## Watching transitions in real time

```bash
tail -F /tmp/echo-backend.log | grep '\[state\]'
```

Output looks like:

```
[state] IDLE -> ACTIVE (session 2c4c54cddc77)
[state] ACTIVE -> IDLE (idle 31s, interactions=2)
[state] IDLE -> ACTIVE (session 4e9ab1f30c2a)
[state] ACTIVE -> IDLE (user closed, interactions=1)
```
