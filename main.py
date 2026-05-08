"""Live face recognition with wake-word activation, passive liveness,
voice-guided enrolment, and re-registration verification.

Press keys in the OpenCV window:
  q  quit
  r  force registration of the largest visible face (only in ACTIVE state)
"""

from __future__ import annotations

# Suppress the harmless Qt font warning from opencv-python's bundled Qt
# before cv2 is imported. (See README troubleshooting.)
import os
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

import argparse
import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2

import config
from database import FaceDB
from hailo_infer import HailoFacePipeline, align_face
from liveness import LivenessChecker
from quality import (
    is_quality_face,
    landmark_anchor,
    landmarks_drift,
    shift_matches_direction,
)
from blink import BlinkChecker
from tts import make_backend
from wake_word import WakeWordError, WakeWordListener


# ---------- helpers ---------------------------------------------------------


def cosine_match(query: np.ndarray, matrix: np.ndarray):
    if matrix.shape[0] == 0:
        return -1, 0.0
    sims = matrix @ query
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


def best_self_match(embeddings, db_ids, db_matrix, emp_id) -> float:
    mask = np.array([eid == emp_id for eid in db_ids], dtype=bool)
    if not mask.any():
        return -1.0
    sub = db_matrix[mask]
    return max(float((sub @ e).max()) for e in embeddings)


class SilentLearner:
    """Quietly appends new embeddings to a recognised person's gallery so
    recognition gets more robust over time. Rate-limited per person and
    capped per person to keep the DB bounded."""

    def __init__(self, db: FaceDB):
        self.db = db
        self._last_added: dict[str, float] = {}

    def maybe_add(self, emp_id: str, name: str,
                  embedding: np.ndarray, score: float) -> bool:
        if not config.SILENT_LEARN_ENABLED:
            return False
        if score < config.SILENT_LEARN_MIN_SCORE:
            return False
        now = time.time()
        if now - self._last_added.get(emp_id, 0) < config.SILENT_LEARN_MIN_INTERVAL_SEC:
            return False

        existing = self.db.get_embeddings(emp_id)
        if existing.shape[0] > 0:
            sims = existing @ embedding
            if float(sims.max()) > config.SILENT_LEARN_MAX_SIMILARITY:
                # Already have an essentially-identical sample. Skip.
                return False

        self.db.add_embedding(emp_id, embedding)
        self._last_added[emp_id] = now

        # Cap per-person sample count by trimming the oldest extras.
        cap = config.SILENT_LEARN_MAX_SAMPLES_PER_PERSON
        n = self.db.count_embeddings(emp_id)
        if n > cap:
            dropped = self.db.trim_embeddings(emp_id, cap)
            print(f"[silent-learn] +1 sample for {emp_id} ({name}); "
                  f"capped at {cap}, dropped {dropped} older.")
        else:
            print(f"[silent-learn] +1 sample for {emp_id} ({name}); now {n}.")
        return True


def largest_detection(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))


# ---------- voice ----------------------------------------------------------


class Greeter:
    """Speaks each recognised face once per cooldown, regardless of how
    many people are in frame. Keeps a per-emp_id timestamp so the same
    person isn't re-greeted while they linger."""

    def __init__(self):
        self.backend = make_backend()
        self._last_greeted: dict[str, float] = {}
        if config.AUDIO_OUTPUT_DEVICE:
            print(f"[TTS] routing audio to ALSA device: {config.AUDIO_OUTPUT_DEVICE}")

    def greet(self, emp_id: str, name: str) -> bool:
        """Returns True iff we actually spoke."""
        now = time.time()
        if now - self._last_greeted.get(emp_id, 0.0) < config.GREET_COOLDOWN_SEC:
            return False
        self._last_greeted[emp_id] = now
        msg = f"Hello {name}, welcome!"
        print(f"[GREET] {msg}")
        self.backend.speak(msg)
        return True

    def say(self, text: str):
        print(f"[TTS] {text}")
        self.backend.speak(text)

    def reset_last(self):
        self._last_greeted.clear()


# ---------- camera ---------------------------------------------------------


def open_camera() -> Picamera2:
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)
    return cam


def grab_frame(cam: Picamera2) -> np.ndarray:
    return cam.capture_array()


# ---------- HUD ------------------------------------------------------------


def draw_hud(frame, dets, label_for, *, banner=None, prompt=None,
             progress=None, status=None):
    if banner:
        # Big banner across the top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame, banner, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = label_for(d)
        if label:
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y = 80 if banner else 30
    if prompt:
        head = f"[{progress[0]}/{progress[1]}] {prompt}" if progress else prompt
        cv2.putText(frame, head, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        y += 30
    if status:
        cv2.putText(frame, status, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ---------- voice-guided pose capture --------------------------------------


def capture_with_prompts(
    cam: Picamera2,
    pipe: HailoFacePipeline,
    greeter: Greeter,
    show_preview: bool,
) -> list[np.ndarray]:
    embeddings: list[np.ndarray] = []
    prompts = config.POSE_PROMPTS
    baseline_anchor = None
    baseline_eye_dist = 1.0

    for i, (prompt, direction) in enumerate(prompts, start=1):
        greeter.say(prompt)
        hold_until = time.time() + config.POSE_HOLD_SEC
        deadline = time.time() + config.POSE_HOLD_SEC + config.POSE_CAPTURE_TIMEOUT_SEC
        stable_since = None
        last_lms = None
        captured = False
        status = "waiting for pose change..." if direction else "waiting..."

        while not captured and time.time() < deadline:
            frame = grab_frame(cam)
            dets = pipe.detect(frame, config.DETECTOR_SCORE_THRESHOLD,
                               config.DETECTOR_NMS_IOU)
            det = largest_detection(dets)
            ok = False
            reason = "no face"
            if det is not None:
                ok, reason = is_quality_face(det, frame.shape)

            if ok and time.time() >= hold_until:
                anchor, eye_dist = landmark_anchor(det)
                shift_ok = (
                    direction is None
                    or baseline_anchor is None
                    or shift_matches_direction(
                        anchor, baseline_anchor, baseline_eye_dist, direction
                    )
                )
                if shift_ok:
                    if last_lms is not None and landmarks_drift(det.landmarks, last_lms) <= config.POSE_STABLE_PIXEL_TOL:
                        if stable_since is None:
                            stable_since = time.time()
                            status = "hold still..."
                        elif time.time() - stable_since >= config.POSE_STABLE_SEC:
                            aligned = align_face(frame, det.landmarks)
                            embeddings.append(pipe.embed(aligned))
                            if direction is None and baseline_anchor is None:
                                baseline_anchor = anchor
                                baseline_eye_dist = eye_dist
                            captured = True
                            status = "captured"
                    else:
                        stable_since = None
                        status = "hold still..."
                    last_lms = det.landmarks
                else:
                    status = f"please move {direction}"
                    stable_since = None
                    last_lms = det.landmarks
            else:
                status = reason
                stable_since = None
                last_lms = None

            if show_preview:
                draw_hud(frame, dets, lambda d: "", prompt=prompt,
                         progress=(len(embeddings), len(prompts)),
                         status=status)
                cv2.imshow("Face Recognition", frame)
                cv2.waitKey(20)

        if not captured:
            greeter.say("Skipping this pose. Let's continue.")
    return embeddings


# ---------- registration ---------------------------------------------------


def prompt_registration(
    db: FaceDB,
    greeter: Greeter,
    cam: Picamera2,
    pipe: HailoFacePipeline,
    show_preview: bool,
) -> bool:
    greeter.say("I do not recognise you. Please register at the console.")
    print("\n--- New face detected ---")
    emp_id = input("Employee ID: ").strip()
    if not emp_id:
        print("Skipped.")
        return False

    is_existing = db.employee_exists(emp_id)
    if is_existing:
        name = db.get_name(emp_id) or ""
        print(f"Employee {emp_id} already exists ({name}). Will verify and append.")
    else:
        name = input("Name: ").strip()
        if not name:
            print("Skipped.")
            return False

    embeddings = capture_with_prompts(cam, pipe, greeter, show_preview)
    if not embeddings:
        greeter.say("Registration failed. No face captured.")
        return False

    if is_existing:
        emp_ids, _, matrix = db.load_all()
        score = best_self_match(embeddings, emp_ids, matrix, emp_id)
        if score < config.REREGISTER_MATCH_THRESHOLD:
            greeter.say(
                "This face does not match the existing employee. Registration refused."
            )
            print(f"Refused: best self-match {score:.2f} < {config.REREGISTER_MATCH_THRESHOLD:.2f}")
            return False
        for emb in embeddings:
            db.add_embedding(emp_id, emb)
        greeter.say(f"Added {len(embeddings)} new samples for {name}.")
        print(f"Re-registered {name} ({emp_id}): +{len(embeddings)} samples (self-match {score:.2f}).\n")
    else:
        db.add_employee(emp_id, name, embeddings)
        greeter.say(f"Thank you {name}, you are now registered.")
        print(f"Registered {name} ({emp_id}) with {len(embeddings)} samples.\n")

    greeter.reset_last()
    return True


# ---------- main loop ------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--auto-register", action="store_true")
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Skip Vosk and start in ACTIVE state immediately.",
    )
    args = parser.parse_args()
    show_preview = not args.no_display

    pipe = HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)
    db = FaceDB()
    greeter = Greeter()
    cam = open_camera()
    liveness = LivenessChecker()
    learner = SilentLearner(db)
    blinker = BlinkChecker(pipe, grab_frame, draw_hud)

    listener: WakeWordListener | None = None
    state = "ACTIVE" if args.no_wake_word else "IDLE"
    if not args.no_wake_word:
        try:
            listener = WakeWordListener(
                config.VOSK_MODEL_DIR,
                config.WAKE_WORD,
                samplerate=config.WAKE_WORD_SAMPLERATE,
                blocksize=config.WAKE_WORD_BLOCKSIZE,
            )
            listener.start()
            print(f"[wake-word] listening for: '{config.WAKE_WORD}'")
        except WakeWordError as exc:
            print(f"[wake-word] disabled: {exc}")
            print("[wake-word] starting in ACTIVE state. Use --no-wake-word to silence this.")
            state = "ACTIVE"

    emp_ids, names, matrix = db.load_all()
    print(f"Loaded {matrix.shape[0]} embeddings for {len(set(emp_ids))} employees.")
    print(f"Initial state: {state}")

    unknown_streak = 0
    # Resets on any "new" event (different person greeted, an unknown face is
    # in frame, registration completed). Same recognised person standing in
    # frame does NOT count as new -- we still time out and go IDLE.
    last_interaction_at = 0.0
    activated_at = 0.0
    # Per-session blink confirmation. Cleared on each ACTIVE entry.
    blink_confirmed: set[str] = set()
    banner_idle = f"Say '{config.WAKE_WORD}' to start recognition"

    try:
        while True:
            frame = grab_frame(cam)

            # ---- wake-word transition (IDLE -> ACTIVE) -----------------
            if state == "IDLE" and listener is not None and listener.is_activated():
                state = "ACTIVE"
                activated_at = time.time()
                last_interaction_at = activated_at
                listener.deactivate()
                greeter.reset_last()
                liveness.reset()
                blink_confirmed.clear()
                greeter.say("Hello. I am ready.")
                print("[state] IDLE -> ACTIVE")

            dets = pipe.detect(frame, config.DETECTOR_SCORE_THRESHOLD,
                               config.DETECTOR_NMS_IOU)
            biggest = largest_detection(dets)
            labels: dict[int, str] = {}

            biggest_quality_unknown = False
            biggest_is_live = False

            if state == "ACTIVE":
                # Temporal liveness only tracked on the largest face.
                biggest_is_live = liveness.update(frame, biggest)

                # Faces we want to greet this frame; collected first, then
                # the blink challenge runs once on the largest of them.
                pending_greets: list[tuple[int, str, str, np.ndarray, float]] = []

                for i, det in enumerate(dets):
                    ok, reason = is_quality_face(det, frame.shape)
                    if not ok:
                        labels[i] = f"low quality: {reason}"
                        continue
                    if det is biggest:
                        if not biggest_is_live:
                            labels[i] = f"checking liveness... ({liveness.last_reason})"
                            continue
                    else:
                        # Secondary faces only get the cheap screen-attack
                        # gates (specular + texture). No temporal window.
                        sf_ok, sf_reason = LivenessChecker.single_frame_check(frame, det)
                        if not sf_ok:
                            labels[i] = f"liveness: {sf_reason}"
                            continue

                    aligned = align_face(frame, det.landmarks)
                    emb = pipe.embed(aligned)
                    idx, score = cosine_match(emb, matrix)
                    if idx >= 0 and score >= config.COSINE_MATCH_THRESHOLD:
                        labels[i] = f"{names[idx]} ({score:.2f})"
                        # Silent learning on confident, live, recognised faces.
                        if learner.maybe_add(emp_ids[idx], names[idx], emb, score):
                            emp_ids, names, matrix = db.load_all()
                        pending_greets.append(
                            (i, emp_ids[idx], names[idx], emb, score)
                        )
                    else:
                        labels[i] = f"unknown ({score:.2f})"
                        if det is biggest:
                            biggest_quality_unknown = True

                # ---- blink challenge gate -------------------------------
                # If any pending greet's emp_id hasn't been blink-confirmed
                # this session, run the challenge once. While the challenge
                # is running, the loop yields control to it.
                need_blink = [
                    g for g in pending_greets if g[1] not in blink_confirmed
                ]
                if config.LIVENESS_REQUIRE_BLINK and need_blink:
                    if blinker.run(cam, greeter, show_preview):
                        for _, eid, _, _, _ in need_blink:
                            blink_confirmed.add(eid)
                        last_interaction_at = time.time()
                    else:
                        greeter.say("Blink not detected. Please try again.")
                        # Don't greet the unconfirmed ones this round.
                        pending_greets = [
                            g for g in pending_greets if g[1] in blink_confirmed
                        ]

                for _, emp_id, name, _, _ in pending_greets:
                    if greeter.greet(emp_id, name):
                        last_interaction_at = time.time()

                if biggest_quality_unknown:
                    # Someone unfamiliar is in frame -- keep awake while we
                    # build up to the registration trigger.
                    last_interaction_at = time.time()
                    unknown_streak += 1
                else:
                    unknown_streak = 0

                if (
                    args.auto_register
                    and unknown_streak >= config.UNKNOWN_FRAMES_BEFORE_REGISTER
                ):
                    unknown_streak = 0
                    if prompt_registration(db, greeter, cam, pipe, show_preview):
                        emp_ids, names, matrix = db.load_all()
                    last_interaction_at = time.time()

                # ---- ACTIVE -> IDLE timeouts ---------------------------
                # IDLE_AFTER_LAST_INTERACTION_SEC after the last *new* event
                # -- a recognised person who just stands there does NOT keep
                # the system awake.
                idle_for = (time.time() - last_interaction_at) if last_interaction_at else 0
                if (
                    idle_for >= config.IDLE_AFTER_LAST_INTERACTION_SEC
                    or (activated_at and time.time() - activated_at >= config.ACTIVE_SESSION_MAX_SEC)
                ):
                    if listener is not None:
                        state = "IDLE"
                        liveness.reset()
                        greeter.reset_last()
                        listener.deactivate()
                        print(f"[state] ACTIVE -> IDLE (idle for {idle_for:.0f}s)")

            # ---- HUD ---------------------------------------------------
            if show_preview:
                if state == "IDLE":
                    # Just show detections, but don't recognise.
                    for i, _ in enumerate(dets):
                        labels[i] = "(idle)"
                    banner = banner_idle
                    status = None
                else:
                    banner = None
                    if biggest is not None and not biggest_is_live:
                        status = (
                            f"liveness: {liveness.last_reason}  "
                            f"motion={liveness.last_relative_motion:.2f}/"
                            f"{config.LIVENESS_REL_MOTION_MIN:.2f}  "
                            f"jitter={liveness.last_pixel_jitter:.1f}/"
                            f"{config.LIVENESS_PIXEL_JITTER_MIN:.1f}  "
                            f"glare={liveness.last_specular_ratio:.0%}/"
                            f"{config.LIVENESS_MAX_SPECULAR_RATIO:.0%}  "
                            f"tex={liveness.last_texture_var:.0f}/"
                            f"{config.LIVENESS_MIN_TEXTURE_VAR:.0f}"
                        )
                    elif biggest_quality_unknown:
                        status = (f"unknown streak {unknown_streak}/"
                                  f"{config.UNKNOWN_FRAMES_BEFORE_REGISTER}")
                    else:
                        status = None

                draw_hud(frame, dets, lambda d: labels.get(dets.index(d), ""),
                         banner=banner, status=status)
                cv2.imshow("Face Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r") and state == "ACTIVE":
                    if prompt_registration(db, greeter, cam, pipe, show_preview):
                        emp_ids, names, matrix = db.load_all()
                    last_interaction_at = time.time()
    finally:
        if listener is not None:
            listener.stop()
        cam.stop()
        pipe.close()
        db.close()
        if show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
