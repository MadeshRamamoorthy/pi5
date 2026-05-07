"""Offline enrollment helper.

Captures N frames of the largest face from the AI Camera, computes embeddings,
and stores them in the database under the given emp_id / name.

Usage:
    python enroll.py --emp-id E123 --name "Jane Doe" --frames 5
"""

from __future__ import annotations

import argparse
import time

import cv2
from picamera2 import Picamera2

import config
from database import FaceDB
from hailo_infer import HailoFacePipeline, align_face
from main import grab_frame, largest_detection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emp-id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--frames", type=int, default=5)
    args = p.parse_args()

    pipe = HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)
    db = FaceDB()

    cam = Picamera2()
    cam.configure(
        cam.create_preview_configuration(
            main={"size": config.CAMERA_RESOLUTION, "format": "RGB888"}
        )
    )
    cam.start()
    time.sleep(1.0)

    embeddings = []
    print(f"Capturing {args.frames} samples. Look at the camera.")
    while len(embeddings) < args.frames:
        frame = grab_frame(cam)
        dets = pipe.detect(
            frame,
            config.DETECTOR_SCORE_THRESHOLD,
            config.DETECTOR_NMS_IOU,
        )
        det = largest_detection(dets)
        if det is None:
            cv2.imshow("Enroll", frame)
            cv2.waitKey(1)
            continue
        aligned = align_face(frame, det.landmarks)
        embeddings.append(pipe.embed(aligned))
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{len(embeddings)}/{args.frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Enroll", frame)
        cv2.waitKey(200)

    if db.employee_exists(args.emp_id):
        for e in embeddings:
            db.add_embedding(args.emp_id, e)
    else:
        db.add_employee(args.emp_id, args.name, embeddings)

    cam.stop()
    pipe.close()
    db.close()
    cv2.destroyAllWindows()
    print(f"Enrolled {args.name} ({args.emp_id}) with {len(embeddings)} embeddings.")


if __name__ == "__main__":
    main()
