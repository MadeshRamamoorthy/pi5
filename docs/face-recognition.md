# Face recognition

## Pipeline

```
SCRFD (face detect, NPU) →
   biggest_quality_face filter (size + landmarks + frontality) →
      LivenessChecker (optional) →
         align_face + ArcFace embed (NPU) →
            cosine match against gallery →
               score >= 0.38 (loose) → recognised
                                      → margin >= 0.15 → silent learner adds embedding
               else → bounding box "Unknown"
```

Bounding boxes are drawn server-side onto each MJPEG frame:

| Colour | Meaning | Label |
|---|---|---|
| 🟢 Green | Recognised | `Name 0.71` |
| 🟡 Yellow | Unknown / low-confidence | `Unknown` |
| 🟠 Orange | Low quality detection | `low quality` |
| 🔵 Amber-cyan | Liveness gathering frames | `checking liveness…` |

## Registration flow

### Live capture (kiosk)

1. Unknown face stays in frame for `UNKNOWN_FRAMES_BEFORE_REGISTER`
   (~15 frames, ~1 s) and `--auto-register` is enabled (default in
   `EXTRA_ARGS` in `start.sh`).
2. Registration form appears with three actions:

| Button | Behaviour |
|---|---|
| **Register** | Form closes, pose capture begins. The camera screen shows a `Pose 1/5` glass card while it walks through "look straight" → "turn slightly left" → "turn slightly right" → "tilt up" → "tilt down". Captures one ArcFace embedding per pose. |
| **No thanks** | Kiosk speaks a random `decline_registration` line from `messages.py` ("No worries! You can still ask me anything.") and drops back to the IDLE dashboard. 120 s cooldown before the prompt can auto-pop again. |
| **✕ close** | Same as No thanks. |

3. Greeting fires **once per ACTIVE session per emp_id** — even if
   you leave and re-enter frame.

### Photo upload (admin)

`/admin` → "Register from photo" tab. Upload 1–N images of the same
person; each one becomes an embedding. Useful for back-office
onboarding without the live capture flow.

Admin uploads run silently — no TTS announcement on the kiosk
speaker. The outcome still logs to `/tmp/echo-backend.log` as
`[photo-register] ...` so you can verify success.

### CLI pre-enrol

```bash
python enroll.py --emp-id E001 --name "Alex Doe" --photo ./photos/alex-*.jpg
```

Same code path as photo upload, just with a CLI front-end.

## Silent learning

A successful greeting opportunistically adds the face's embedding
to that person's gallery — but only if **all** the following hold:

- match score ≥ `SILENT_LEARN_MIN_SCORE` (default 0.70)
- runner-up score is ≥ `SILENT_LEARN_MIN_MARGIN` (default 0.15)
  below the best — closes the two-similar-people drift trap
- new sample isn't a near-duplicate
  (cos ≤ `SILENT_LEARN_MAX_SIMILARITY`, default 0.92)
- at most one new sample per minute per person
- cap at `SILENT_LEARN_MAX_SAMPLES_PER_PERSON` (default 30) total —
  oldest non-enrolment samples drop first

Set `SILENT_LEARN_ENABLED = False` in `config.py` to disable
entirely.

## Liveness (optional)

Disabled by default (`LIVENESS_ENABLED = False`). When on, the
checker runs a sliding-window analysis of:

- **Texture variance** (`LIVENESS_MIN_TEXTURE_VAR`) — printed
  photos have lower variance than real skin.
- **Specular highlights** (`LIVENESS_MAX_SPECULAR_RATIO`) — glossy
  screens have more.
- **Pixel jitter** (`LIVENESS_PIXEL_JITTER_MIN`) — frame-to-frame
  natural micro-motion.
- **Relative motion** (`LIVENESS_REL_MOTION_MIN`) — landmark
  displacement over the window.
- **Active blink challenge** (`LIVENESS_REQUIRE_BLINK`) — kiosk
  asks the user to blink and watches for eye-state change.

Effective against printed photos and most phone-screen replays;
**not bulletproof**. A high-quality video on a large monitor with
natural ambient motion will bypass the passive checks. The blink
challenge raises the bar.

For high-stakes deployments, combine with depth sensing (3D
structured-light or stereo) or a dedicated liveness model.

## Cleanup / re-registration

If silent learning drifted a gallery (the kiosk starts mis-matching
people), inspect and clean:

```bash
sqlite3 faces.db "SELECT emp_id, name, COUNT(*)
                  FROM employees JOIN face_embeddings USING(emp_id)
                  GROUP BY emp_id;"
```

Drop a problematic gallery and re-register from photos:

```bash
sqlite3 faces.db "DELETE FROM face_embeddings WHERE emp_id='E001';"
# Then re-upload photos via /admin -> Register from photo
```

Or use the Employees admin tab → Delete → re-register.

## Confidence tuning

If recognition is too loose (false positives), raise
`COSINE_MATCH_THRESHOLD` from 0.38 → 0.42 or 0.45.

If recognition is too strict (you're registered but treated as
unknown), lower it to 0.32.

Default 0.38 is a reasonable compromise for ArcFace
MobileFaceNet.
