"""Live face recognition + auto-registration loop.

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


def cosine_match(query: np.ndarray, matrix: np.ndarray):
    """Return (best_index, best_score) or (-1, 0) if matrix is empty."""
    if matrix.shape[0] == 0:
        return -1, 0.0
    sims = matrix @ query
    idx = int(np.argmax(sims))
    return idx, float(sims[idx])


class Greeter:
    """Speaks once per person change. Same emp_id back-to-back stays silent."""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)
        self._last_greeted_emp_id: str | None = None

    def greet(self, emp_id: str, name: str):
        if emp_id == self._last_greeted_emp_id:
            return
        self._last_greeted_emp_id = emp_id
        msg = f"Hello {name}, welcome!"
        print(f"[GREET] {msg}")
        self.engine.say(msg)
        self.engine.runAndWait()

    def say(self, text: str):
        print(f"[TTS] {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def reset_last(self):
        self._last_greeted_emp_id = None


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
    """picamera2 'RGB888' actually delivers BGR-ordered bytes on libcamera, so
    the array already matches what cv2 / our model wrappers expect.
    """
    return cam.capture_array()


def largest_detection(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))


def _annotate(frame, dets, prompt: str, captured: int, total: int):
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(
        frame,
        f"[{captured}/{total}] {prompt}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )


def capture_with_prompts(
    cam: Picamera2,
    pipe: HailoFacePipeline,
    greeter: Greeter,
    show_preview: bool,
) -> list[np.ndarray]:
    """Voice-guided multi-pose capture. One embedding per prompt."""
    embeddings: list[np.ndarray] = []
    prompts = config.POSE_PROMPTS
    for i, prompt in enumerate(prompts, start=1):
        greeter.say(prompt)
        # Live preview while the user moves into the new pose.
        settle_until = time.time() + config.POSE_HOLD_SEC
        while time.time() < settle_until:
            frame = grab_frame(cam)
            if show_preview:
                dets_preview = pipe.detect(
                    frame,
                    config.DETECTOR_SCORE_THRESHOLD,
                    config.DETECTOR_NMS_IOU,
                )
                _annotate(frame, dets_preview, prompt, len(embeddings), len(prompts))
                cv2.imshow("Face Recognition", frame)
                cv2.waitKey(30)

        # Now try to grab one good face within the timeout.
        deadline = time.time() + config.POSE_CAPTURE_TIMEOUT_SEC
        captured = False
        while not captured and time.time() < deadline:
            frame = grab_frame(cam)
            dets = pipe.detect(
                frame,
                config.DETECTOR_SCORE_THRESHOLD,
                config.DETECTOR_NMS_IOU,
            )
            det = largest_detection(dets)
            if det is not None:
                aligned = align_face(frame, det.landmarks)
                embeddings.append(pipe.embed(aligned))
                captured = True
            if show_preview:
                _annotate(frame, dets, prompt, len(embeddings), len(prompts))
                cv2.imshow("Face Recognition", frame)
                cv2.waitKey(30)
        if not captured:
            greeter.say("I could not see your face clearly. Skipping this one.")
    return embeddings


def best_self_match(embeddings, db_ids, db_matrix, emp_id):
    """Return the best cosine score of `embeddings` against the rows of
    `db_matrix` whose emp_id == emp_id. -1 if that emp_id has no rows."""
    mask = np.array([eid == emp_id for eid in db_ids], dtype=bool)
    if not mask.any():
        return -1.0
    sub = db_matrix[mask]
    best = -1.0
    for emb in embeddings:
        sims = sub @ emb
        best = max(best, float(sims.max()))
    return best


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
                "This face does not match the existing employee. "
                "Registration refused."
            )
            print(
                f"Refused: best self-match {score:.2f} < "
                f"{config.REREGISTER_MATCH_THRESHOLD:.2f}"
            )
            return False
        for emb in embeddings:
            db.add_embedding(emp_id, emb)
        greeter.say(f"Added {len(embeddings)} new samples for {name}.")
        print(
            f"Re-registered {name} ({emp_id}): +{len(embeddings)} samples "
            f"(self-match {score:.2f}).\n"
        )
    else:
        db.add_employee(emp_id, name, embeddings)
        greeter.say(f"Thank you {name}, you are now registered.")
        print(f"Registered {name} ({emp_id}) with {len(embeddings)} samples.\n")

    # The next live frame might match this newly-enrolled person — make sure
    # we actually greet them rather than treating it as "same as last".
    greeter.reset_last()
    return True


def draw(frame, dets, label_for):
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = label_for(d)
        if label:
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--auto-register",
        action="store_true",
        help="Prompt to register every unknown face automatically.",
    )
    args = parser.parse_args()
    show_preview = not args.no_display

    pipe = HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)
    db = FaceDB()
    greeter = Greeter()
    cam = open_camera()

    emp_ids, names, matrix = db.load_all()
    print(f"Loaded {matrix.shape[0]} embeddings for {len(set(emp_ids))} employees.")

    last_unknown_prompt = 0.0

    try:
        while True:
            frame = grab_frame(cam)
            dets = pipe.detect(
                frame,
                config.DETECTOR_SCORE_THRESHOLD,
                config.DETECTOR_NMS_IOU,
            )

            labels: dict[int, str] = {}
            biggest = largest_detection(dets)

            for i, det in enumerate(dets):
                aligned = align_face(frame, det.landmarks)
                emb = pipe.embed(aligned)
                idx, score = cosine_match(emb, matrix)
                if idx >= 0 and score >= config.COSINE_MATCH_THRESHOLD:
                    labels[i] = f"{names[idx]} ({score:.2f})"
                    greeter.greet(emp_ids[idx], names[idx])
                else:
                    labels[i] = f"unknown ({score:.2f})"
                    if (
                        det is biggest
                        and args.auto_register
                        and time.time() - last_unknown_prompt > 5
                    ):
                        last_unknown_prompt = time.time()
                        if prompt_registration(db, greeter, cam, pipe, show_preview):
                            emp_ids, names, matrix = db.load_all()

            if show_preview:
                draw(frame, dets, lambda d: labels.get(dets.index(d), ""))
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
