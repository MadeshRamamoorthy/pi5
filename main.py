"""ECHO SCOPE kiosk — orchestrator.

Spawns the camera worker, the weather poller, the fun-fact rotator,
the wake-word listener, the chat-voice capture, and the Flask app
that serves the browser SPA. Each component talks through the
StateBus, which Flask exposes to the browser over Server-Sent Events.

There is no more cv2 window -- the UI is a Chromium kiosk pointed at
http://127.0.0.1:8080. start.sh launches the browser after this
process is up.
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import threading
import time
import uuid

import numpy as np
from picamera2 import Picamera2

import config
import messages
from asr import make_chat_asr
from async_tts import AsyncTTS
from chat import ChatBackendError, ChatBudgetError, ChatClient
from chat_voice import ChatVoiceCapture, ChatVoiceCaptureError
from database import FaceDB
from frame_streamer import FrameStreamer
from fun_facts import FunFactRotator
from hailo_infer import HailoFacePipeline, align_face
from liveness import LivenessChecker
from quality import (is_quality_face, landmark_anchor, landmarks_drift,
                     shift_matches_direction)
from state import StateBus
from tts import make_backend
from wake_word import WakeWordError, WakeWordListener
from weather import WeatherPoller
from web import create_app


# ---------- helpers --------------------------------------------------------


def cosine_match(query: np.ndarray, matrix: np.ndarray):
    """Return (best_idx, best_score, second_best_score). When fewer
    than two embeddings exist, second_best is -1.0 so margin checks
    against it always pass."""
    if matrix.shape[0] == 0:
        return -1, 0.0, -1.0
    sims = matrix @ query
    order = np.argsort(sims)[::-1]
    best_idx = int(order[0])
    best = float(sims[best_idx])
    second = float(sims[order[1]]) if order.size > 1 else -1.0
    return best_idx, best, second


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


def _draw_overlays(frame, face_labels):
    """Annotate `frame` in-place with rectangles + name labels.

    face_labels: list of (det, label_text, bgr_color) tuples produced
    by the recognition loop. Drawing happens server-side -- the
    browser just decodes the resulting JPEG -- so the user gets the
    "boxes around faces" visual that the cv2 build had, without
    spending bandwidth on per-detection JSON pushes."""
    import cv2
    for det, text, color in face_labels:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if not text:
            continue
        (tw, th), _bl = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1,
        )
        # Label background bar just above the box.
        ly2 = max(0, y1 - th - 10)
        cv2.rectangle(frame, (x1, ly2), (x1 + tw + 12, y1), color, -1)
        cv2.putText(
            frame, text, (x1 + 6, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )


# ---------- silent learning -----------------------------------------------


class SilentLearner:
    def __init__(self, db: FaceDB):
        self.db = db
        self._last_added: dict[str, float] = {}

    def maybe_add(self, emp_id, name, embedding, score,
                  runner_up: float = -1.0) -> bool:
        if not config.SILENT_LEARN_ENABLED:
            return False
        if score < config.SILENT_LEARN_MIN_SCORE:
            return False
        # The classic drift trap: two similar-looking faces both score
        # 0.65-ish against the same person. Both look "comfortable" in
        # isolation, but the small margin says the model isn't actually
        # sure which one this is. Refuse to learn when the runner-up is
        # too close.
        if runner_up >= 0.0 and (score - runner_up) < config.SILENT_LEARN_MIN_MARGIN:
            return False
        now = time.time()
        if now - self._last_added.get(emp_id, 0) < config.SILENT_LEARN_MIN_INTERVAL_SEC:
            return False
        existing = self.db.get_embeddings(emp_id)
        if existing.shape[0] > 0:
            sims = existing @ embedding
            if float(sims.max()) > config.SILENT_LEARN_MAX_SIMILARITY:
                return False
        self.db.add_embedding(emp_id, embedding)
        self._last_added[emp_id] = now
        cap = config.SILENT_LEARN_MAX_SAMPLES_PER_PERSON
        if self.db.count_embeddings(emp_id) > cap:
            self.db.trim_embeddings(emp_id, cap)
        return True


# ---------- voice ---------------------------------------------------------


class Greeter:
    """Async TTS-backed greeter with per-emp_id cooldown + a toast push
    to the SPA so the user can see who was greeted."""

    def __init__(self, tts: AsyncTTS, state: StateBus):
        self._tts = tts
        self._state = state
        self._last_greeted: dict[str, float] = {}

    def greet(self, emp_id: str, name: str) -> bool:
        now = time.time()
        if now - self._last_greeted.get(emp_id, 0.0) < config.GREET_COOLDOWN_SEC:
            return False
        self._last_greeted[emp_id] = now
        msg = messages.random_recognized_greeting(name)
        print(f"[GREET] {msg}")
        # Push the toast + person state from the TTS worker thread so
        # the UI update lands exactly when audio starts playing, not
        # N queued utterances earlier.
        def on_start():
            self._state.update(
                person={"emp_id": emp_id, "name": name},
                toast={"text": msg, "since": time.time()},
            )
        self._tts.speak(msg, on_start=on_start)
        return True

    def say(self, text: str) -> None:
        print(f"[TTS] {text}")
        def on_start():
            self._state.update(toast={"text": text, "since": time.time()})
        self._tts.speak(text, on_start=on_start)

    def reset_last(self) -> None:
        self._last_greeted.clear()


# ---------- camera --------------------------------------------------------


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


# ---------- camera worker thread ------------------------------------------


class CameraWorker(threading.Thread):
    """Owns the camera, runs detection / recognition / registration, and
    publishes the latest frame to FrameStreamer and state changes to
    the StateBus."""

    daemon = True

    def __init__(self, db, state, frames, tts, chat, greeter, learner,
                 register_queue, photo_register_queue, chat_queue, wake_event,
                 idle_event, args, listener=None):
        super().__init__(name="camera-worker")
        self.db = db
        self.state = state
        self.frames = frames
        self.tts = tts
        self.chat = chat
        self.greeter = greeter
        self.learner = learner
        self.register_q = register_queue
        self.photo_register_q = photo_register_queue
        self.chat_q = chat_queue
        self.wake_event = wake_event
        self.idle_event = idle_event
        self.listener = listener
        self.args = args
        self.cam = None
        self.pipe = None
        self.liveness = LivenessChecker()
        self._stop = threading.Event()
        # When the user explicitly says "no thanks" to registration, we
        # don't re-pop the overlay for this long. Otherwise the unknown-
        # face streak would re-open it immediately.
        self._register_declined_at = 0.0

    def stop(self):
        self._stop.set()

    # ---- the loop ----

    def run(self):
        try:
            self.pipe = HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)
            self.cam = open_camera()
        except Exception as exc:
            print(f"[camera-worker] init failed: {exc!r}")
            return
        emp_ids, names, matrix = self.db.load_all()
        print(f"Loaded {matrix.shape[0]} embeddings for {len(set(emp_ids))} employees.")
        self._refresh_idle_data()

        # State machine local vars.
        kiosk_state = "ACTIVE" if self.args.no_wake_word else "IDLE"
        self.state.update(state=kiosk_state, state_since=time.time())
        unknown_streak = 0
        last_interaction_at = time.time() if kiosk_state == "ACTIVE" else 0.0
        activated_at = time.time() if kiosk_state == "ACTIVE" else 0.0
        seen_in_session: set[str] = set()
        session_id = uuid.uuid4().hex[:12] if kiosk_state == "ACTIVE" else ""
        last_known_emp_id = "anon"
        last_idle_refresh = 0.0

        while not self._stop.is_set():
            frame = grab_frame(self.cam)
            self.state.set_now()
            # Detection overlays are drawn onto `frame` further down,
            # then we push to MJPEG. While IDLE we push immediately so
            # the dashboard's (unused) camera socket stays warm.
            if kiosk_state == "IDLE":
                self.frames.push(frame)

            # ---- IDLE -> ACTIVE on wake word OR touch -------
            if kiosk_state == "IDLE" and self.wake_event.is_set():
                self.wake_event.clear()
                kiosk_state = "ACTIVE"
                activated_at = time.time()
                last_interaction_at = activated_at
                seen_in_session = set()
                session_id = uuid.uuid4().hex[:12]
                self.greeter.reset_last()
                self.liveness.reset()
                self.chat.budget.reset()
                self.tts.flush()
                self.state.go_active()
                self.state.update(
                    person=None,
                    chat_history=[],
                    chat_remaining=config.CHAT_MAX_QUESTIONS_PER_SESSION,
                    chat_pending=False,
                )
                # Pause the wake-word listener while ACTIVE -- there's no
                # point burning CPU / contending for the mic when the
                # user is already engaged with the kiosk.
                if self.listener is not None:
                    self.listener.pause()
                self.greeter.say("Hello! Lovely to see you.")
                print(f"[state] IDLE -> ACTIVE (session {session_id})")

            # Refresh idle data periodically (projects / metrics /
            # session card) -- once every 5 s is plenty.
            if time.time() - last_idle_refresh > 5:
                last_idle_refresh = time.time()
                self._refresh_idle_data()

            # ---- process register requests (from web POST) ---
            try:
                while True:
                    req = self.register_q.get_nowait()
                    # "Skip" sentinel: user declined. Speak one of the
                    # decline_registration lines from messages.py, close
                    # the overlay, and drop straight back to the
                    # dashboard (IDLE) -- no point hanging around.
                    if isinstance(req, dict) and req.get("action") == "skip":
                        self._register_declined_at = time.time()
                        self.state.update(register_open=False,
                                          register_pose=None)
                        self.greeter.say(messages.random_registration_prompt(
                            "decline_registration",
                        ))
                        self.idle_event.set()
                        continue
                    if kiosk_state != "ACTIVE":
                        continue
                    self._run_registration(req)
                    emp_ids, names, matrix = self.db.load_all()
                    last_interaction_at = time.time()
            except queue.Empty:
                pass

            # ---- process photo-based registration (from web upload) -
            try:
                while True:
                    req = self.photo_register_q.get_nowait()
                    self._run_photo_registration(req)
                    emp_ids, names, matrix = self.db.load_all()
                    last_interaction_at = time.time()
            except queue.Empty:
                pass

            # ---- process chat-voice transcripts --------------
            try:
                while True:
                    spoken = self.chat_q.get_nowait()
                    if kiosk_state != "ACTIVE" or not spoken:
                        continue
                    self._submit_chat(last_known_emp_id, spoken)
                    last_interaction_at = time.time()
            except queue.Empty:
                pass

            # ---- recognition (only in ACTIVE) ----------------
            biggest_quality_unknown = False
            dets = self.pipe.detect(frame, config.DETECTOR_SCORE_THRESHOLD,
                                    config.DETECTOR_NMS_IOU)
            biggest = largest_detection(dets)
            # label entries: (det, text, color_bgr)
            face_labels: list[tuple] = []

            if kiosk_state == "ACTIVE":
                # Liveness can be turned off entirely via config; when
                # disabled we treat every quality face as live and skip
                # both the sliding-window analysis and the per-face
                # single-frame check.
                liveness_on = getattr(config, "LIVENESS_ENABLED", True)
                biggest_is_live = (
                    self.liveness.update(frame, biggest)
                    if liveness_on else True
                )
                pending_greets = []
                for i, det in enumerate(dets):
                    ok, _ = is_quality_face(det, frame.shape)
                    if not ok:
                        face_labels.append((det, "low quality", (0, 165, 255)))
                        continue
                    if liveness_on:
                        if det is biggest:
                            if not biggest_is_live:
                                face_labels.append((det, "checking liveness…", (0, 200, 255)))
                                continue
                        else:
                            sf_ok, _ = LivenessChecker.single_frame_check(frame, det)
                            if not sf_ok:
                                face_labels.append((det, "checking liveness…", (0, 200, 255)))
                                continue
                    aligned = align_face(frame, det.landmarks)
                    emb = self.pipe.embed(aligned)
                    idx, score, runner_up = cosine_match(emb, matrix)
                    if idx >= 0 and score >= config.COSINE_MATCH_THRESHOLD:
                        if self.learner.maybe_add(
                            emp_ids[idx], names[idx], emb, score, runner_up,
                        ):
                            emp_ids, names, matrix = self.db.load_all()
                        pending_greets.append(
                            (emp_ids[idx], names[idx], det, score)
                        )
                        face_labels.append((
                            det, f"{names[idx]}  {score:.2f}", (0, 255, 0),
                        ))
                    else:
                        face_labels.append((det, "Unknown", (0, 255, 255)))
                        if det is biggest:
                            biggest_quality_unknown = True

            # Draw boxes + names on the frame, then publish to MJPEG.
            if kiosk_state == "ACTIVE":
                _draw_overlays(frame, face_labels)
                self.frames.push(frame)

                for emp_id, name, _det, _score in pending_greets:
                    if self.greeter.greet(emp_id, name):
                        last_interaction_at = time.time()
                        if emp_id not in seen_in_session:
                            seen_in_session.add(emp_id)
                            self.db.record_interaction(emp_id, session_id)
                            self._push_metrics()
                        last_known_emp_id = emp_id

                # Auto-open the register overlay for new visitors.
                snap = self.state.snapshot()
                if biggest_quality_unknown:
                    last_interaction_at = time.time()
                    unknown_streak += 1
                    declined_recently = (
                        time.time() - self._register_declined_at
                    ) < config.REGISTER_DECLINE_COOLDOWN_SEC
                    if (self.args.auto_register
                            and unknown_streak >= config.UNKNOWN_FRAMES_BEFORE_REGISTER
                            and not snap.get("register_open")
                            and not snap.get("chat_pending")
                            and not declined_recently):
                        unknown_streak = 0
                        self.state.update(
                            register_open=True,
                            register_step="form",
                            register_message=(
                                "Hi! Pop your details below and tap Register."
                            ),
                        )
                        self.greeter.say(messages.random_unrecognized_greeting())
                else:
                    unknown_streak = 0

                # Idle timeout (keep alive while user is engaged).
                # "Engaged" includes:
                #   - register overlay is open
                #   - chat is waiting for a reply
                #   - chat-voice mic is recording
                #   - a chat exchange happened recently (chat_history
                #     non-empty AND within CHAT_KEEPALIVE_SEC of last
                #     chat activity) -- gives the user time to read
                #     the answer and ask a follow-up.
                chat_active = (
                    snap.get("chat_pending")
                    or bool(snap.get("chat_history"))
                    and (time.time() - getattr(self, "_last_chat_at", 0))
                        <= config.CHAT_KEEPALIVE_SEC
                )
                user_engaged = (
                    snap.get("register_open")
                    or snap.get("listening")
                    or chat_active
                )
                if user_engaged:
                    last_interaction_at = time.time()
                idle_for = (time.time() - last_interaction_at) if last_interaction_at else 0
                force_idle = self.idle_event.is_set()
                if (force_idle
                        or idle_for >= config.IDLE_AFTER_LAST_INTERACTION_SEC
                        or (activated_at and time.time() - activated_at >= config.ACTIVE_SESSION_MAX_SEC)):
                    if force_idle:
                        self.idle_event.clear()
                    kiosk_state = "IDLE"
                    self.liveness.reset()
                    self.greeter.reset_last()
                    self.tts.flush()
                    self.state.go_idle()
                    # Resume the wake-word listener -- the user is gone,
                    # we need to be listening for the next "hello echo".
                    if self.listener is not None:
                        self.listener.resume()
                    reason = "user closed" if force_idle else f"idle {idle_for:.0f}s"
                    print(f"[state] ACTIVE -> IDLE ({reason}, "
                          f"interactions={len(seen_in_session)})")

                # Poll the chat future.
                fut = getattr(self, "_chat_future", None)
                if fut is not None and fut.done():
                    self._chat_future = None
                    try:
                        answer = fut.result()
                    except ChatBudgetError as exc:
                        answer = str(exc)
                    except ChatBackendError as exc:
                        answer = ("I'm having a little trouble reaching the "
                                  f"chat service right now ({exc}).")
                    except Exception as exc:  # noqa: BLE001
                        answer = f"Hmm, something went sideways: {exc}"
                    self.state.update(chat_pending=False)
                    self._append_chat("assistant", answer)
                    self.greeter.say(answer)
                    self._last_chat_at = time.time()
                    last_interaction_at = time.time()

    # ---- registration flow (driven by /api/register) -----------------

    def _run_registration(self, req: dict):
        emp_id = req["emp_id"]
        name = req["name"]
        # Close the form overlay immediately so the user can see the
        # camera feed during pose capture. Pose progress now renders
        # as a card on the active screen via state.register_pose.
        self.state.update(
            register_open=False,
            register_step="capturing",
            register_message=f"Capturing your photo, {name}!",
        )

        is_existing = self.db.employee_exists(emp_id)
        if is_existing:
            existing_name = self.db.get_name(emp_id) or ""
            self.greeter.say(
                f"Welcome back, {existing_name}! Confirming it's you."
            )

        embeddings = self._capture_with_prompts()
        if not embeddings:
            self.state.update(
                register_step="form",
                register_message=("Hmm, we couldn't capture a clear photo. "
                                   "Take a step closer and try again."),
                register_pose=None,
            )
            self.greeter.say("Let's give that another go.")
            return

        if is_existing:
            emp_ids, _, matrix = self.db.load_all()
            score = best_self_match(embeddings, emp_ids, matrix, emp_id)
            if score < config.REREGISTER_MATCH_THRESHOLD:
                self.state.update(
                    register_step="form",
                    register_message="Looks like a fresh face! Try a new Employee ID.",
                    register_pose=None,
                )
                self.greeter.say("Let's set you up with a new profile.")
                return
            for e in embeddings:
                self.db.add_embedding(emp_id, e)
            confirm = messages.random_registration_prompt(
                "confirm_registration", name=name,
            )
            self.greeter.say(confirm)
            self.state.update(
                register_open=False,
                register_pose=None,
                register_message=f"Welcome back, {name}!",
            )
        else:
            self.db.add_employee(emp_id, name, embeddings)
            confirm = messages.random_registration_prompt(
                "confirm_registration", name=name,
            )
            self.greeter.say(confirm)
            self.state.update(
                register_open=False,
                register_pose=None,
                register_message=f"Welcome, {name}!",
            )

    def _run_photo_registration(self, req: dict):
        """Register from one or more uploaded photos -- no pose capture.

        req = {"emp_id": str, "name": str, "photos": [bytes, ...]}
        """
        import cv2
        emp_id = req["emp_id"]
        name = req["name"]
        photos = req.get("photos") or []
        if not photos:
            return

        embeddings = []
        for blob in photos:
            arr = np.frombuffer(blob, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            # Mirror Picamera2 colour order: BGR (cv2) -> RGB.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = self.pipe.detect(
                img, config.DETECTOR_SCORE_THRESHOLD, config.DETECTOR_NMS_IOU,
            )
            det = largest_detection(dets)
            if det is None:
                continue
            aligned = align_face(img, det.landmarks)
            embeddings.append(self.pipe.embed(aligned))

        self.state.update(register_open=False, register_pose=None)
        if not embeddings:
            self.greeter.say(
                "Hmm, I couldn't see a clear face in those photos. "
                "Want to try with a different one?"
            )
            return

        is_existing = self.db.employee_exists(emp_id)
        if is_existing:
            for e in embeddings:
                self.db.add_embedding(emp_id, e)
        else:
            self.db.add_employee(emp_id, name, embeddings)
        confirm = messages.random_registration_prompt(
            "confirm_registration", name=name,
        )
        self.greeter.say(confirm)
        self.state.update(register_message=f"Welcome, {name}!")

    def _capture_with_prompts(self):
        embeddings = []
        prompts = config.POSE_PROMPTS
        baseline_anchor = None
        baseline_eye_dist = 1.0

        for i, (prompt, direction) in enumerate(prompts, start=1):
            self.greeter.say(prompt)
            self.state.update(register_pose={
                "idx": i, "total": len(prompts),
                "prompt": prompt, "status": "speaking..."
            })
            # Render frames while the prompt plays.
            wait_deadline = time.time() + 8
            while (not self.tts.wait_idle(timeout=0.05)
                   and time.time() < wait_deadline):
                self.frames.push(grab_frame(self.cam))

            hold_until = time.time() + config.POSE_HOLD_SEC
            deadline = (time.time() + config.POSE_HOLD_SEC
                        + config.POSE_CAPTURE_TIMEOUT_SEC)
            stable_since = None
            last_lms = None
            captured = False

            while not captured and time.time() < deadline:
                frame = grab_frame(self.cam)
                self.frames.push(frame)
                dets = self.pipe.detect(
                    frame, config.DETECTOR_SCORE_THRESHOLD,
                    config.DETECTOR_NMS_IOU,
                )
                det = largest_detection(dets)
                ok = False
                if det is not None:
                    ok, _ = is_quality_face(det, frame.shape)
                if det is None:
                    status = "step into frame"
                elif not ok:
                    status = "move closer / face the camera"
                elif time.time() < hold_until:
                    status = "hold steady..."
                else:
                    status = "capturing — hold the pose"
                self.state.patch("register_pose", status=status)

                if ok and time.time() >= hold_until:
                    anchor, eye_dist = landmark_anchor(det)
                    shift_ok = (
                        direction is None
                        or baseline_anchor is None
                        or shift_matches_direction(
                            anchor, baseline_anchor, baseline_eye_dist, direction,
                        )
                    )
                    if shift_ok:
                        if (last_lms is not None
                                and landmarks_drift(det.landmarks, last_lms)
                                <= config.POSE_STABLE_PIXEL_TOL):
                            if stable_since is None:
                                stable_since = time.time()
                            elif time.time() - stable_since >= config.POSE_STABLE_SEC:
                                aligned = align_face(frame, det.landmarks)
                                embeddings.append(self.pipe.embed(aligned))
                                if direction is None and baseline_anchor is None:
                                    baseline_anchor = anchor
                                    baseline_eye_dist = eye_dist
                                captured = True
                        else:
                            stable_since = None
                        last_lms = det.landmarks
                    else:
                        stable_since = None
                        last_lms = det.landmarks
                else:
                    stable_since = None
                    last_lms = None

            if not captured:
                self.greeter.say("That's okay! Let's try the next one.")
                self.tts.wait_idle(timeout=3)
        self.state.update(register_pose=None)
        return embeddings

    # ---- chat ---------------------------------------------------------

    def _submit_chat(self, emp_id: str, question: str):
        self._append_chat("user", question)
        if self.chat.budget.remaining(emp_id) <= 0:
            answer = (
                f"Lovely chatting! That's {config.CHAT_MAX_QUESTIONS_PER_SESSION} "
                "questions for now — come say hi again anytime."
            )
            self._append_chat("assistant", answer)
            self.greeter.say(answer)
            return
        self.state.update(chat_pending=True)
        self._last_chat_at = time.time()
        self._chat_future = self.chat.submit(emp_id, question)

    def _append_chat(self, role: str, text: str):
        snap = self.state.snapshot()
        history = list(snap.get("chat_history") or [])
        history.append([role, text])
        history = history[-12:]    # cap so SSE diffs stay small
        emp_id = (snap.get("person") or {}).get("emp_id", "anon")
        self.state.update(
            chat_history=history,
            chat_remaining=self.chat.budget.remaining(emp_id),
        )

    # ---- idle dashboard data ------------------------------------------

    def _refresh_idle_data(self):
        # Projects.
        projects = [
            {"id": r[0], "title": r[1], "description": r[2],
             "ordering": r[3]}
            for r in self.db.list_projects()
        ]
        # Sessions: next upcoming.
        nxt = self.db.next_session()
        next_session = None
        if nxt is not None:
            next_session = {
                "id": nxt[0], "title": nxt[1],
                "starts_at": nxt[2], "ends_at": nxt[3], "notes": nxt[4],
            }
        # Metrics.
        best_day, best_count = self.db.interaction_best_day()
        metrics = {
            "total":    self.db.interaction_count_total(),
            "today":    self.db.interaction_count_today(),
            "week":     self.db.interaction_count_this_week(),
            "best_day": best_day,
            "best_count": best_count,
        }
        self.state.update(projects=projects,
                          next_session=next_session,
                          metrics=metrics)

    def _push_metrics(self):
        best_day, best_count = self.db.interaction_best_day()
        self.state.update(metrics={
            "total":    self.db.interaction_count_total(),
            "today":    self.db.interaction_count_today(),
            "week":     self.db.interaction_count_this_week(),
            "best_day": best_day,
            "best_count": best_count,
        })


# ---------- main ----------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-register", action="store_true")
    parser.add_argument("--no-wake-word", action="store_true")
    parser.add_argument("--no-display", action="store_true",
                        help="(compat) ignored -- the kiosk has no cv2 window.")
    args = parser.parse_args()

    db = FaceDB()
    state = StateBus()
    state.update(
        brand=getattr(config, "BRAND_NAME", "ECHO SCOPE"),
        wake_phrase=config.WAKE_WORD,
        contact_email=getattr(config, "CONTACT_EMAIL", ""),
    )

    weather = WeatherPoller()
    weather.start()
    state_weather_pump = _start_weather_pump(weather, state)

    fun_facts = FunFactRotator(state)
    fun_facts.start()

    tts = AsyncTTS(make_backend())
    chat = ChatClient()
    state.update(chat_backend=chat.status(),
                 chat_remaining=config.CHAT_MAX_QUESTIONS_PER_SESSION)

    greeter = Greeter(tts, state)
    learner = SilentLearner(db)
    frames = FrameStreamer()

    wake_event = threading.Event()
    idle_event = threading.Event()
    register_q: queue.Queue = queue.Queue()
    photo_register_q: queue.Queue = queue.Queue()
    chat_q: queue.Queue = queue.Queue()

    # Wake-word listener.
    listener: WakeWordListener | None = None
    if not args.no_wake_word:
        try:
            # Print every Vosk hypothesis so it's obvious WHY the wake
            # word isn't firing -- accent / phrasing / mic positioning
            # are usually the cause and were previously invisible.
            _last_partial = [""]

            def _on_partial(text: str):
                if text and text != _last_partial[0]:
                    _last_partial[0] = text
                    print(f"[wake-word] partial: {text!r}", flush=True)

            def _on_final(text: str):
                print(f"[wake-word] FINAL  : {text!r}", flush=True)
                _last_partial[0] = ""

            listener = WakeWordListener(
                config.VOSK_MODEL_DIR, config.WAKE_WORD,
                samplerate=config.WAKE_WORD_SAMPLERATE,
                on_partial=_on_partial,
                on_final=_on_final,
            )
            listener.start()
            # Poll thread: convert listener.is_activated() to wake_event.
            threading.Thread(target=_poll_wake, args=(listener, wake_event),
                              daemon=True).start()
            phrases = ", ".join(repr(p) for p in listener._phrases)
            print(f"[wake-word] listening for any of: {phrases}")
        except WakeWordError as exc:
            print(f"[wake-word] disabled: {exc}")

    # Chat voice.
    chat_voice: ChatVoiceCapture | None = None
    try:
        chat_voice = ChatVoiceCapture(
            asr=make_chat_asr(),
            out_queue=chat_q,
            wake_listener=listener,
            on_listening_changed=lambda on: state.update(listening=on),
        )
        print(f"[chat-voice] using {config.CHAT_ASR_BACKEND}")
    except ChatVoiceCaptureError as exc:
        print(f"[chat-voice] disabled: {exc}")

    # Worker thread.
    worker = CameraWorker(db, state, frames, tts, chat, greeter, learner,
                          register_q, photo_register_q, chat_q,
                          wake_event, idle_event, args, listener=listener)
    worker.start()

    # Flask app -- callbacks bridge the browser to the worker.
    def request_wake():
        wake_event.set()

    def request_idle():
        idle_event.set()

    def request_register(payload):
        register_q.put(payload)

    def request_register_photo(payload):
        photo_register_q.put(payload)

    def request_register_skip():
        register_q.put({"action": "skip"})

    def request_chat(question):
        # Route chat questions through the same chat_q the chat-voice
        # capture uses so the worker only has one entrypoint.
        chat_q.put(question)

    def listen_start():
        if chat_voice is not None:
            chat_voice.start()

    def listen_stop():
        if chat_voice is not None:
            chat_voice.stop()

    app = create_app(state, frames, db,
                     request_wake=request_wake,
                     request_idle=request_idle,
                     request_register=request_register,
                     request_register_photo=request_register_photo,
                     request_register_skip=request_register_skip,
                     request_chat=request_chat,
                     listen_start=listen_start,
                     listen_stop=listen_stop)

    host = getattr(config, "KIOSK_HOST", "127.0.0.1")
    port = int(os.environ.get("KIOSK_PORT", getattr(config, "KIOSK_PORT", 8080)))
    print(f"echo-scope serving on http://{host}:{port}")
    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        worker.stop()
        if listener is not None:
            listener.stop()
        weather.stop()
        fun_facts.stop()
        chat.shutdown()
        tts.stop()


def _poll_wake(listener: WakeWordListener, ev: threading.Event):
    while True:
        if listener.is_activated():
            ev.set()
            listener.deactivate()
        time.sleep(0.1)


def _start_weather_pump(weather: WeatherPoller, state: StateBus):
    """Mirror WeatherPoller.get() into state.weather every 30 s."""
    def pump():
        while True:
            w = weather.get()
            state.update(weather={
                "ok": w.get("ok", False),
                "temp_c": w.get("temp_c"),
                "label": w.get("label"),
                "city": w.get("city"),
                "humidity": w.get("humidity"),
                "icon": w.get("icon", "🌡️"),
            })
            time.sleep(30)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    main()
