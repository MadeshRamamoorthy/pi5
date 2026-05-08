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
import subprocess
import tempfile
import time

import cv2
import numpy as np
import pyttsx3
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


def largest_detection(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))


# ---------- voice ----------------------------------------------------------


class Greeter:
    """Speaks once per person change. Same emp_id back-to-back stays silent.

    If config.AUDIO_OUTPUT_DEVICE is set, TTS audio is routed through
    `aplay -D <device>` so the greeting plays on the chosen sink (e.g. the
    Anker) instead of the system default (typically HDMI on the Pi).
    """

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)
        self._last_greeted: str | None = None
        self._device = config.AUDIO_OUTPUT_DEVICE
        if self._device:
            print(f"[TTS] routing audio to ALSA device: {self._device}")

    def _speak(self, text: str) -> None:
        if not self._device:
            self.engine.say(text)
            self.engine.runAndWait()
            return
        # Synth to a WAV, then play it on the chosen device. Bypasses
        # whatever the system default sink is.
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(fd)
        try:
            self.engine.save_to_file(text, path)
            self.engine.runAndWait()
            subprocess.run(
                ["aplay", "-q", "-D", self._device, path],
                check=False,
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def greet(self, emp_id: str, name: str) -> bool:
        """Returns True iff we actually spoke (i.e. emp_id changed)."""
        if emp_id == self._last_greeted:
            return False
        self._last_greeted = emp_id
        msg = f"Hello {name}, welcome!"
        print(f"[GREET] {msg}")
        self._speak(msg)
        return True

    def say(self, text: str):
        print(f"[TTS] {text}")
        self._speak(text)

    def reset_last(self):
        self._last_greeted = None


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
                greeter.say("Hello. I am ready.")
                print("[state] IDLE -> ACTIVE")

            dets = pipe.detect(frame, config.DETECTOR_SCORE_THRESHOLD,
                               config.DETECTOR_NMS_IOU)
            biggest = largest_detection(dets)
            labels: dict[int, str] = {}

            biggest_quality_unknown = False
            biggest_is_live = False

            if state == "ACTIVE":
                # Liveness updates only on the largest face's track.
                biggest_is_live = liveness.update(frame, biggest)

                for i, det in enumerate(dets):
                    ok, reason = is_quality_face(det, frame.shape)
                    if not ok:
                        labels[i] = f"low quality: {reason}"
                        continue
                    if det is biggest and not biggest_is_live:
                        labels[i] = f"checking liveness... ({liveness.last_reason})"
                        continue

                    aligned = align_face(frame, det.landmarks)
                    emb = pipe.embed(aligned)
                    idx, score = cosine_match(emb, matrix)
                    if idx >= 0 and score >= config.COSINE_MATCH_THRESHOLD:
                        labels[i] = f"{names[idx]} ({score:.2f})"
                        if det is biggest:
                            # greet() returns True only on a *new* person.
                            if greeter.greet(emp_ids[idx], names[idx]):
                                last_interaction_at = time.time()
                    else:
                        labels[i] = f"unknown ({score:.2f})"
                        if det is biggest:
                            biggest_quality_unknown = True

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
                # 30s after the last *new* event -- a recognised person who
                # just stands there does NOT keep the system awake.
                idle_for = (time.time() - last_interaction_at) if last_interaction_at else 0
                if (
                    idle_for >= config.SLEEP_AFTER_NO_LIVE_FACE_SEC
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
                        status = (f"liveness: {liveness.last_reason}  "
                                  f"motion={liveness.last_relative_motion:.2f}  "
                                  f"jitter={liveness.last_pixel_jitter:.1f}")
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
