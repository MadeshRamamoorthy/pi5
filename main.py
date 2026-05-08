"""Live face recognition + voice-guided auto-registration.

Press keys in the OpenCV window:
  q  quit
  r  force registration of the largest visible face
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pyttsx3
from picamera2 import Picamera2

import config
from database import FaceDB
from hailo_infer import HailoFacePipeline, align_face
from quality import (
    is_quality_face,
    landmark_anchor,
    landmarks_drift,
    shift_matches_direction,
)


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
    """Speaks once per person change. Same emp_id back-to-back stays silent."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)
        self._last_greeted: str | None = None

    def greet(self, emp_id: str, name: str):
        if emp_id == self._last_greeted:
            return
        self._last_greeted = emp_id
        msg = f"Hello {name}, welcome!"
        print(f"[GREET] {msg}")
        self.engine.say(msg)
        self.engine.runAndWait()

    def say(self, text: str):
        print(f"[TTS] {text}")
        self.engine.say(text)
        self.engine.runAndWait()

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
    """picamera2 'RGB888' delivers BGR-ordered bytes on libcamera; the array
    already matches what cv2 / our model wrappers expect."""
    return cam.capture_array()


# ---------- HUD ------------------------------------------------------------


def draw_hud(frame, dets, label_for, prompt: str | None = None,
             progress: tuple[int, int] | None = None,
             status: str | None = None):
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = label_for(d)
        if label:
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if prompt:
        head = f"[{progress[0]}/{progress[1]}] {prompt}" if progress else prompt
        cv2.putText(frame, head, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    if status:
        cv2.putText(frame, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ---------- voice-guided pose capture --------------------------------------


def capture_with_prompts(
    cam: Picamera2,
    pipe: HailoFacePipeline,
    greeter: Greeter,
    show_preview: bool,
) -> list[np.ndarray]:
    """Walks through config.POSE_PROMPTS. Each pose only captures once we see:
        1. the user has actually shifted their face in the asked direction, AND
        2. the face has been a high-quality, stable detection for POSE_STABLE_SEC.
    """
    embeddings: list[np.ndarray] = []
    prompts = config.POSE_PROMPTS
    baseline_anchor = None    # set after the first ("look straight") capture
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
                draw_hud(frame, dets, lambda d: "", prompt, (len(embeddings), len(prompts)), status)
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
    parser.add_argument("--auto-register", action="store_true",
                        help="Prompt to register every unknown face automatically.")
    args = parser.parse_args()
    show_preview = not args.no_display

    pipe = HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)
    db = FaceDB()
    greeter = Greeter()
    cam = open_camera()

    emp_ids, names, matrix = db.load_all()
    print(f"Loaded {matrix.shape[0]} embeddings for {len(set(emp_ids))} employees.")

    unknown_streak = 0  # consecutive frames where the largest face is good but unknown

    try:
        while True:
            frame = grab_frame(cam)
            dets = pipe.detect(frame, config.DETECTOR_SCORE_THRESHOLD,
                               config.DETECTOR_NMS_IOU)

            labels: dict[int, str] = {}
            biggest = largest_detection(dets)
            biggest_quality_unknown = False

            for i, det in enumerate(dets):
                ok, reason = is_quality_face(det, frame.shape)
                if not ok:
                    labels[i] = f"low quality: {reason}"
                    continue
                aligned = align_face(frame, det.landmarks)
                emb = pipe.embed(aligned)
                idx, score = cosine_match(emb, matrix)
                if idx >= 0 and score >= config.COSINE_MATCH_THRESHOLD:
                    labels[i] = f"{names[idx]} ({score:.2f})"
                    if det is biggest:
                        greeter.greet(emp_ids[idx], names[idx])
                else:
                    labels[i] = f"unknown ({score:.2f})"
                    if det is biggest:
                        biggest_quality_unknown = True

            if biggest_quality_unknown:
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

            if show_preview:
                status = (f"unknown streak {unknown_streak}/{config.UNKNOWN_FRAMES_BEFORE_REGISTER}"
                          if biggest_quality_unknown else None)
                draw_hud(frame, dets, lambda d: labels.get(dets.index(d), ""), status=status)
                cv2.imshow("Face Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    if prompt_registration(db, greeter, cam, pipe, show_preview):
                        emp_ids, names, matrix = db.load_all()
    finally:
        cam.stop()
        pipe.close()
        db.close()
        if show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
