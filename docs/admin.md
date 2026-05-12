# Admin UI

`http://<pi>:8090/admin` — HTTP basic auth, credentials from the
startup banner (or whatever you pinned in `.env` as
`KIOSK_ADMIN_USER` / `KIOSK_ADMIN_PASS`).

## Tabs

| Tab | What's there |
|---|---|
| **Projects** | The list shown on the idle dashboard. Add / delete |
| **Sessions** | Upcoming session card (next row from this table). Datetime pickers for start / end |
| **Employees** | Registered faces — name, sample count, registration date, rename, delete |
| **Register from photo** | Multipart upload — 1+ images per person, silent registration |

A metrics strip above the tabs summarises total / today / this week /
best-day interaction counts.

## HTTP API

All paths relative to `http://<pi>:8090`. Endpoints marked
**[admin]** require basic auth.

### State + media

```
GET    /                                SPA shell (HTML)
GET    /events                          SSE diff stream
GET    /api/state                       full kiosk state snapshot (JSON)
GET    /camera.mjpg                     MJPEG multipart stream
```

### Session control

```
POST   /api/wake                        force IDLE -> ACTIVE
POST   /api/idle                        force ACTIVE -> IDLE
POST   /api/listen/start                start chat-voice recording
POST   /api/listen/stop                 stop chat-voice recording
POST   /api/chat                        {question: "..."} - submit chat
```

### Registration

```
POST   /api/register                    {emp_id, name}  - kiosk live capture
POST   /api/register/photo              multipart       - admin photo upload  [admin]
POST   /api/register/skip               decline + 120 s cooldown
```

### CRUD

```
GET    /api/projects                    list projects
POST   /api/projects                    {title, description, ordering}     [admin]
POST   /api/projects/<id>               update                              [admin]
DELETE /api/projects/<id>               delete                              [admin]

GET    /api/sessions                    list sessions
POST   /api/sessions                    {title, starts_at, ends_at, notes}  [admin]
POST   /api/sessions/<id>               update                              [admin]
DELETE /api/sessions/<id>               delete                              [admin]

GET    /api/employees                   list registered faces               [admin]
POST   /api/employees/<emp_id>          {name} - rename                     [admin]
DELETE /api/employees/<emp_id>          delete (also removes embeddings)    [admin]
```

### Metrics

```
GET    /api/metrics                     {total, today, week, best_day, best_count}
```

## Authentication

The auth check is permissive when `KIOSK_ADMIN_PASS` is unset: the
admin endpoints **refuse** rather than fall through to "open" mode.
`start.sh` generates a random `KIOSK_ADMIN_PASS` at every launch when
the user hasn't pinned one, so admin is always behind auth.

To pin a stable password:

```bash
# In .env:
KIOSK_ADMIN_USER=admin
KIOSK_ADMIN_PASS=your-stable-password-here
```

## Examples

```bash
# Browse projects without auth
curl http://localhost:8090/api/projects

# Add a project (with auth)
curl -u admin:kP3qN-tw -X POST http://localhost:8090/api/projects \
     -H 'Content-Type: application/json' \
     -d '{"title": "New Project", "description": "...", "ordering": 0}'

# Upload a photo for registration
curl -u admin:kP3qN-tw -X POST http://localhost:8090/api/register/photo \
     -F emp_id=E001 -F name="Alex Doe" \
     -F photos=@alex-1.jpg -F photos=@alex-2.jpg

# Force the kiosk back to IDLE
curl -X POST http://localhost:8090/api/idle
```

## Legacy admin (port 8081)

The original `admin_web.py` still ships in the repo. Run it
separately if you want a second admin surface:

```bash
python admin_web.py
# serves on http://<pi>:8081
```

It shares the same `faces.db`. New admin work happens on the `/admin`
route inside the main app; the legacy app is kept for compatibility.
