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
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)
        self._last = {}  # emp_id -> timestamp

    def greet(self, emp_id: str, name: str):
        now = time.time()
        if now - self._last.get(emp_id, 0) < config.GREET_COOLDOWN_SEC:
            return
        self._last[emp_id] = now
        msg = f"Hello {name}, welcome!"
        print(f"[GREET] {msg}")
        self.engine.say(msg)
        self.engine.runAndWait()

    def say(self, text: str):
        print(f"[TTS] {text}")
        self.engine.say(text)
        self.engine.runAndWait()


def open_camera() -> Picamera2:
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)  # let AE/AWB settle
    return cam


def grab_frame(cam: Picamera2) -> np.ndarray:
    rgb = cam.capture_array()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def largest_detection(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))


def prompt_registration(db: FaceDB, greeter: Greeter, embedding: np.ndarray) -> bool:
    """Ask via stdin for emp_id + name and persist the embedding."""
    greeter.say("I do not recognise you. Please register at the console.")
    print("\n--- New face detected ---")
    emp_id = input("Employee ID: ").strip()
    if not emp_id:
        print("Skipped.")
        return False
    name = input("Name: ").strip()
    if not name:
        print("Skipped.")
        return False
    if db.employee_exists(emp_id):
        db.add_embedding(emp_id, embedding)
    else:
        db.add_employee(emp_id, name, [embedding])
    greeter.say(f"Thank you {name}, you are now registered.")
    print(f"Registered {name} ({emp_id}).\n")
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
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless (no preview window).",
    )
    parser.add_argument(
        "--auto-register",
        action="store_true",
        help="Prompt to register every unknown face automatically.",
    )
    args = parser.parse_args()

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
                        if prompt_registration(db, greeter, emb):
                            emp_ids, names, matrix = db.load_all()

            if not args.no_display:
                draw(frame, dets, lambda d: labels.get(dets.index(d), ""))
                cv2.imshow("Face Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r") and biggest is not None:
                    aligned = align_face(frame, biggest.landmarks)
                    emb = pipe.embed(aligned)
                    if prompt_registration(db, greeter, emb):
                        emp_ids, names, matrix = db.load_all()
    finally:
        cam.stop()
        pipe.close()
        db.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
