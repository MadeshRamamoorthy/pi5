# Running the kiosk

## `start.sh`

```bash
./start.sh
```

What it does, in order:

1. Sources `.env` if present (`set -a` so every key=value becomes an
   exported env var).
2. Activates the venv.
3. Skips Hailo-Ollama when `OPENAI_API_KEY` is set (unless
   `PRELOAD_OLLAMA=1`).
4. Generates a random admin password if `KIOSK_ADMIN_PASS` is unset.
5. Warms up the Whisper model in the background.
6. Starts `main.py` (backend) in the background. Output logs to
   `/tmp/echo-backend.log`.
7. Polls `/api/state` until the Flask app is up (up to 30 s).
8. Execs Chromium in `--kiosk --app=http://127.0.0.1:8090`.
9. Traps Ctrl+C: kills the Python child cleanly.

The startup banner shows everything important:

```
---------------------------------------------------------------
 ECHO SCOPE kiosk
   project    : /home/echo/Documents/code/pi5
   config     : .env loaded
   openai     : key set (***k3xL)
   mic        : sounddevice index 1
   speaker    : plughw:CARD=PowerConf,DEV=0
   web URL    : http://127.0.0.1:8090
   admin user : admin
   admin pass : kP3qN-tw    (generated -- set KIOSK_ADMIN_PASS to pin)
---------------------------------------------------------------
   backend    : up (pid 12345)
   browser    : chromium (kiosk mode)
```

If the backend doesn't come up:

```
Error: backend didn't come up -- see /tmp/echo-backend.log
```

Tail the log; the cause is almost always a missing model file or a
HailoRT version mismatch.

## `stop.sh`

```bash
./stop.sh                # graceful TERM, then KILL after ~3 s
FORCE=1 ./stop.sh        # immediate KILL
./stop.sh --ollama       # also stop the hailo-ollama daemon
```

What it does:

- `pkill -TERM` everything matching `python.* main\.py`
- `pkill -TERM` everything matching `chromium.*--kiosk`
- Waits up to 3 s; escalates to `KILL`
- Releases the kiosk port via `fuser -k 8090/tcp` if anything's
  still bound
- With `--ollama`, also TERMs the `hailo-ollama` daemon

## Live logs

```bash
tail -F /tmp/echo-backend.log                          # everything
tail -F /tmp/echo-backend.log | grep -E 'wake-word|state|GREET|asr'
```

The backend emits a partial + final transcript for every Vosk
hypothesis so you can see exactly what it heard:

```
[wake-word] partial: 'hello'
[wake-word] partial: 'hello echo'
[wake-word] FINAL  : 'hello echo'
[state] IDLE -> ACTIVE (session 2c4c54cddc77)
[GREET] Welcome, Parakh!
```

State transitions log the *reason*:

```
[state] ACTIVE -> IDLE (idle 30s, interactions=2)
[state] ACTIVE -> IDLE (user closed, interactions=1)
```

## Auto-start on boot

Add a user systemd unit at `~/.config/systemd/user/echo-scope.service`:

```ini
[Unit]
Description=ECHO SCOPE kiosk
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=/home/echo/Documents/code/pi5
ExecStart=/home/echo/Documents/code/pi5/start.sh
Restart=on-failure
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now echo-scope
loginctl enable-linger echo                       # so it survives logout
```

Status / logs:

```bash
systemctl --user status echo-scope
journalctl --user -u echo-scope -f
```

## Restarting on code changes

There's no auto-reload. Stop and restart:

```bash
./stop.sh && ./start.sh
```

For SPA (HTML/CSS/JS) iteration only, **Ctrl+F5** in Chromium hard-
reloads the assets without restarting the backend.
