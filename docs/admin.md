# Admin UI

The admin app runs as a **separate Flask process** from the kiosk on
port **8081**. Either side can run without the other — they share
`faces.db` (in WAL mode), so projects/sessions/employees updates
flow between them via the database.

```
http://<pi>:8081/        admin UI + admin API
http://<pi>:8090/        kiosk SPA + kiosk API
```

Same `.env`, same `KIOSK_ADMIN_USER` / `KIOSK_ADMIN_PASS` credentials.

## Launching

```bash
./start_admin.sh         # only the admin process
./start.sh               # only the kiosk process
# Run both side-by-side -- they share the chip via group_id="SHARED"
```

Each launcher loads `.env`, generates a random `KIOSK_ADMIN_PASS` if
none is set (printed in the banner), and runs its Flask app.

## Tabs

| Tab | What's there |
|---|---|
| **Projects** | The list shown on the idle dashboard. Add / edit / delete |
| **Sessions** | Upcoming session card (next row from this table). Datetime pickers |
| **Employees** | Registered faces — name, sample count, profile URL, custom welcome, rename, delete |
| **Register from photo** | Multipart upload — 1+ images per person |

A metrics strip above the tabs summarises total / today / this week /
best-day interaction counts.

## HTTP API

All paths relative to `http://<pi>:8081`. Endpoints marked **[admin]**
require HTTP basic auth.

### Read

```
GET    /                                admin HTML  [admin]
GET    /api/projects                    list projects
GET    /api/sessions                    list sessions
GET    /api/employees                   list registered faces  [admin]
GET    /api/metrics                     {total, today, week, best_day, best_count}
```

### Mutate

```
POST   /api/projects                    {title, description, ordering}     [admin]
POST   /api/projects/<id>               update                              [admin]
DELETE /api/projects/<id>               delete                              [admin]

POST   /api/sessions                    {title, starts_at, ends_at, notes}  [admin]
POST   /api/sessions/<id>               update                              [admin]
DELETE /api/sessions/<id>               delete                              [admin]

POST   /api/employees/<emp_id>          {name, profile_url, custom_welcome} [admin]
DELETE /api/employees/<emp_id>          delete (cascades to face_embeddings)[admin]
POST   /api/employees/<emp_id>/welcome  regenerate welcome from profile URL [admin]

POST   /api/register/photo              multipart (emp_id, name, photos[])  [admin]
```

The kiosk app (port 8090) does NOT serve any of these. `GET /admin`
on the kiosk port returns 404.

## Authentication

`KIOSK_ADMIN_PASS` must be set. If it's not (neither in `.env` nor in
the environment), the launcher generates a random one per launch and
prints it. To pin:

```bash
# .env
KIOSK_ADMIN_USER=admin
KIOSK_ADMIN_PASS=your-stable-password-here
```

The auth check refuses rather than falls through, so a missing
password never accidentally opens the admin endpoints.

## Examples

```bash
# Browse projects without auth (read-only is public)
curl http://localhost:8081/api/projects

# Add a project (with auth)
curl -u admin:kP3qN-tw -X POST http://localhost:8081/api/projects \
     -H 'Content-Type: application/json' \
     -d '{"title": "New Project", "description": "...", "ordering": 0}'

# Upload a photo for registration
curl -u admin:kP3qN-tw -X POST http://localhost:8081/api/register/photo \
     -F emp_id=E001 -F name="Alex Doe" \
     -F photos=@alex-1.jpg -F photos=@alex-2.jpg
```

## Photo registration without the kiosk

Photo register works whether or not the kiosk is running. The admin
process opens its own `HailoFacePipeline` per upload, runs SCRFD +
ArcFace on the photos, persists the embeddings, then closes the
pipeline. Cost: ~3 s of NPU acquire + HEF load per upload. Idle
memory cost: 0 (pipeline only lives for the duration of one request).

If you're doing batch onboarding (10+ photos), see the comment in
`web/admin_app.py:register_photo` for how to switch to a
process-lifetime pipeline (~300 MB always resident in exchange for
zero per-upload latency).

## Sharing the NPU when both run

The kiosk holds a long-running `HailoFacePipeline` for face
recognition. The admin process spins up a second one on each photo
upload. Both use `scheduling_algorithm = ROUND_ROBIN` and
`group_id = "SHARED"` so HailoRT multiplexes the chip. The same
mechanism handles Whisper-on-NPU + face-recognition concurrency.

Practical impact: a photo upload during active kiosk recognition adds
a few hundred ms to one frame's recognition latency. No visible
stall in the kiosk's wake/greet flow.

## Database concurrency

`database.py` opens SQLite with `PRAGMA journal_mode = WAL` and
`busy_timeout = 2000`. WAL allows concurrent readers + a single
writer without blocking. `busy_timeout` retries silently for up to 2
seconds if both processes try to write at exactly the same instant.

This makes the file safe to share between the kiosk's silent learner
(writing `face_embeddings` rows) and the admin's CRUD writes
(`projects` / `sessions` / `employees`).

Backups: WAL creates a `faces.db-wal` sidecar. Either include both
files in your backup or run `sqlite3 faces.db "PRAGMA wal_checkpoint(TRUNCATE);"`
first to merge the WAL back into the main DB.
